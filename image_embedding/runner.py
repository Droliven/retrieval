#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : runner.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-26 15:56
'''
from tqdm import tqdm
import json
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from image_embedding.nets.embedding_net import Embedding
from image_embedding.datas.dataset import ImgDataset
from image_embedding.config import Config
from torch.utils.data import DataLoader

LOG_EVERY = 40


class EmbeddingLoss(nn.Module):
    '''

    :param pred: B, D
    :param gt:
    :return:
    '''

    def __init__(self):
        super(EmbeddingLoss, self).__init__()

    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        gt = torch.sigmoid(gt)

        loss = -torch.mean(torch.mul(gt, torch.log(pred)) + torch.mul((1 - gt), torch.log(1 - pred)))
        return loss


def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Runner:
    def __init__(self):
        self.cfg = Config()
        self.best_eval_loss = 1e8
        self.global_step = 1

        self.lr = self.cfg.lr

        self.writter = SummaryWriter(log_dir=self.cfg.ckpt_dir)
        with open(osp.join(self.cfg.ckpt_dir, "config.json"), "w") as fp:
            fp.write(json.dumps(self.cfg.__dict__))

        self.model = Embedding(self.cfg.embedding_dim, self.cfg.backbone_type)
        self.losser = EmbeddingLoss()

        if self.cfg.device != "cpu":
            self.model = nn.DataParallel(self.model).to(self.cfg.device)
            self.losser = nn.DataParallel(self.losser).to(self.cfg.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.sgd_momentum)

        self.train_set = ImgDataset(self.cfg.ds_root, split="train")
        self.valid_set = ImgDataset(self.cfg.ds_root, split="val")
        # self.test_set = ImgDataset(self.cfg.img_base_dir, self.cfg.txt_embedding_base_dir, self.cfg.test_path)
        self.train_dataloader = DataLoader(self.train_set, batch_size=self.cfg.train_batch_size,
                                           num_workers=self.cfg.num_workers, shuffle=True, pin_memory=True,
                                           drop_last=True)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=self.cfg.valid_batch_size,
                                           num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True,
                                           drop_last=True)
        # self.test_dataloader = DataLoader(self.test_set, batch_size=self.cfg.test_batch_size, num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True, drop_last=True)

        print(f"train len: {self.train_set.__len__()}")
        print(f"valid len: {self.valid_set.__len__()}")
        # print(f"test len: {self.test_set.__len__()}")

    def run(self):
        for epo in range(1, self.cfg.epochs + 1):
            # train
            self.model.train()
            for img, txt_emb in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
                if self.cfg.device != "cpu":
                    img = img.float().to(self.cfg.device)
                    txt_emb = txt_emb.float().to(self.cfg.device)

                out_emb = self.model(img)
                loss = self.losser(out_emb, txt_emb)

                self.optimizer.zero_grad()
                loss.sum().backward()
                self.optimizer.step()

                loss = loss.sum().cpu().data.numpy()
                self.global_step += 1

                # TODO summary
                if self.global_step % LOG_EVERY == 0:
                    self.writter.add_scalar('Loss/train', loss, self.global_step)

            # ???????????? forward  validation set
            self.model.eval()
            avg_eval_loss = 0
            for idx, (img, txt_emb) in enumerate(tqdm(self.valid_dataloader, total=len(self.valid_dataloader))):
                if self.cfg.device != "cpu":
                    img = img.float().to(self.cfg.device)
                    txt_emb = txt_emb.float().to(self.cfg.device)

                with torch.no_grad():
                    out_emb = self.model(img)

                    loss_eval = self.losser(out_emb, txt_emb)
                    avg_eval_loss += loss_eval.cpu().sum().data.numpy()

            avg_eval_loss = avg_eval_loss / idx
            self.writter.add_scalar('Loss/val', avg_eval_loss, epo)

            if avg_eval_loss < self.best_eval_loss:
                self.best_eval_loss = avg_eval_loss
                print(f">>> epoch {epo} / {self.cfg.epochs}, lr: {self.lr}, "
                      f"avg_eval_loss: {avg_eval_loss},"
                      f" best_eval_loss: {self.best_eval_loss}")
                self.save(epo, self.best_eval_loss, avg_eval_loss, is_best=True)

            if epo % self.cfg.lr_descent_every == 0:
                # descent lr
                self.lr = lr_decay(self.optimizer, self.lr, self.cfg.lr_descent_rate)

        self.save(self.cfg.epochs, -1, -1, is_best=False)

    def save(self, epoch, best_loss, curr_loss, is_best=False):
        print(f"Saved epoch: {epoch}, currloss: {curr_loss}, bestloss: {best_loss} to {self.cfg.ckpt_dir}.")
        save_name = f"epoch{epoch}_currloss{curr_loss}_bestloss{best_loss}.pth.tar" if not is_best \
            else "best.pth.tar"
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(), 'curr_loss': curr_loss},
                   os.path.join(self.cfg.ckpt_dir, save_name))


if __name__ == '__main__':
    runner = Runner()
    runner.run()

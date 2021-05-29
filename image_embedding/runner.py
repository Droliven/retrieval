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
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from nets import Embedding
from datas import ImgDataset
from config import Config
from torch.utils.data import DataLoader

def embedding_loss(pred, gt):
    '''

    :param pred: B, D
    :param gt:
    :return:
    '''
    loss = -torch.mean(torch.mul(gt, torch.log(pred)) + torch.mul((1-gt), torch.log(1-pred)))
    return loss

def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class Runner():
    def __init__(self):
        self.cfg = Config()
        self.best_eval_loss = 1e8
        self.global_step = 1

        self.lr = self.cfg.lr

        self.writter = SummaryWriter(log_dir=self.cfg.ckpt_dir)

        self.model = Embedding(self.cfg.embedding_dim)
        if self.cfg.device != "cpu":
            self.model = self.model.to(self.cfg.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.sgd_momentum)

        self.train_set = ImgDataset(self.cfg.train_img_path, self.cfg.train_txt_embed_path)
        self.valid_set = ImgDataset(self.cfg.valid_img_path, self.cfg.valid_txt_embed_path)
        # self.test_set = ImgDataset(self.cfg.test_img_path, self.cfg.test_txt_embed_path)
        self.train_dataloader = DataLoader(self.train_set, batch_size=self.cfg.train_batch_size, num_workers=self.cfg.num_workers, shuffle=True, pin_memory=True, drop_last=True)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=self.cfg.valid_batch_size, num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True, drop_last=True)
        # self.test_dataloader = DataLoader(self.test_set, batch_size=self.cfg.test_batch_size, num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True, drop_last=True)

        print(f"train len: {self.train_set.__len__()}")
        print(f"valid len: {self.valid_set.__len__()}")
        # print(f"test len: {self.test_set.__len__()}")

    def run(self):
        for epo in range(1, self.cfg.epochs + 1):
            # train
            self.model.train()
            for img, txt_emb in self.train_dataloader:
                if self.cfg.device != "cpu":
                    img = img.float().to(self.cfg.device)
                    txt_emb = txt_emb.float().to(self.cfg.device)

                out_emb = self.model(img)
                out_emb = nn.Sigmoid()(out_emb)
                txt_emb = nn.Sigmoid()(txt_emb)

                loss = embedding_loss(out_emb, txt_emb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.cpu().data.numpy()
                # TODO summary
                self.writter.add_scalar('Loss/train', loss, self.global_step)
                self.global_step += 1

            if epo % self.cfg.eval_iters_evary == 0 or epo == self.cfg.epochs:
                # eval
                self.model.eval()
                avg_eval_loss = 0
                for idx, (img, txt_emb) in enumerate(self.valid_dataloader):
                    if self.cfg.device != "cpu":
                        img = img.float().to(self.cfg.device)
                        txt_emb = txt_emb.float().to(self.cfg.device)

                    with torch.no_grad():
                        out_emb = self.model(img)
                        out_emb = nn.Sigmoid()(out_emb)
                        txt_emb = nn.Sigmoid()(txt_emb)

                        loss_eval = embedding_loss(out_emb, txt_emb)
                        avg_eval_loss += loss_eval.cpu().data.numpy()

                avg_eval_loss = avg_eval_loss / (idx)
                self.writter.add_scalar('Loss/val', avg_eval_loss, epo)

                if avg_eval_loss < self.best_eval_loss:
                    self.best_eval_loss = avg_eval_loss

                print(f">>> epoch {epo} / {self.cfg.epochs}, lr: {self.lr}, avg_eval_loss: {avg_eval_loss}, best_eval_loss: {self.best_eval_loss}")
                self.save(epo, self.best_eval_loss, avg_eval_loss)

            if epo % self.cfg.lr_descent_every == 0:
                # descent lr
                self.lr = lr_decay(self.optimizer, self.lr, self.cfg.lr_descent_rate)


    def test(self):
        # eval
        self.model.eval()
        avg_test_loss = 0
        for idx, (img, txt_emb) in enumerate(self.test_dataloader):
            if self.cfg.device != "cpu":
                img = img.float().to(self.cfg.device)
                txt_emb = txt_emb.float().to(self.cfg.device)

            with torch.no_grad():
                out_emb = self.model(img)
                out_emb = nn.Sigmoid()(out_emb)
                txt_emb = nn.Sigmoid()(txt_emb)

                loss_test = embedding_loss(out_emb, txt_emb)
                avg_test_loss += loss_test.cpu().data.numpy()

        avg_test_loss = avg_test_loss / (idx)
        print(f"avg_test_loss: {avg_test_loss}")


    def save(self, epoch, best_loss, curr_loss):
        print(f"Saved epoch: {epoch}, currloss: {curr_loss}, bestloss: {best_loss} to {self.cfg.ckpt_dir}.")
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(), 'curr_loss': curr_loss},
                   os.path.join(self.cfg.ckpt_dir, f"epoch{epoch}_currloss{curr_loss}_bestloss{best_loss}.pth.tar"))


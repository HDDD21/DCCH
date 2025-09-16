from utils.tools import *
import itertools
from scipy.linalg import hadamard
from network import *
import pdb
import os
import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from torch.nn.utils.clip_grad import clip_grad_norm_
import math


parser = argparse.ArgumentParser(description='manual to this script')
def str2bool(str):
    return True if str.lower() == 'true' else False
parser.add_argument('--gpus', type = str, default = '0')
parser.add_argument('--hash_dim', type = int, default = 32)
parser.add_argument('--noise_rate', type = float, default = 0.2)
parser.add_argument('--dataset', type = str, default = 'flickr')
parser.add_argument('--Lambda', type = float, default = 0.9)
parser.add_argument('--num_gradual', type = int, default = 100)

# parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--beta", type=float, default=0.05)
parser.add_argument("--gamma", type=float, default=3)
parser.add_argument("--lamda_", type=float, default=0.5) 
parser.add_argument("--q", type=float, default=0.01)
parser.add_argument("--tau_", type=float, default=1.0) #temperature

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

bit_len = args.hash_dim
noise_rate = args.noise_rate
dataset = args.dataset
Lambda=args.Lambda
num_gradual =  args.num_gradual

beta = args.beta
gamma = args.gamma
lamda_ = args.lamda_
tp= args.tp
q= args.q
tau_=args.tau_

if dataset == 'flickr':
    train_size = 10000
elif dataset == 'ms-coco':
    train_size = 10000
elif dataset == 'nuswide21':
    train_size = 10500
elif dataset == 'iapr':
    train_size = 10000
n_class = 0
tag_len = 0
torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        # "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 1e-4}},
        # "txt_optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 1e-4}}, 
        # "p_optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-4, "weight_decay": 1e-4}},
        
        "optimizer": {"type": optim.Adam , "optim_params": {"lr": 5e-5, "weight_decay": 1e-4}},
        "txt_optimizer": {"type": optim.Adam , "optim_params": {"lr": 5e-5, "weight_decay": 1e-4}}, 
        "p_optimizer": {"type": optim.Adam , "optim_params": {"lr": 1e-5, "weight_decay": 1e-4}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size":128,
        "dataset": dataset,
        # "epoch": 100,
        "epoch": 100,
        "device": torch.device("cuda:0"),
        "bit_len": bit_len,
        "noise_type": 'symmetric',
        "noise_rate": noise_rate,
        "random_state": 1,
        "n_class": n_class,
        "lambda":Lambda,
        "tag_len":tag_len,
        "train_size": train_size,

        "beta": args.beta,
        "gamma": args.gamma,
        "lamda_": args.lamda_,
        "tp": args.tp,
        "q": args.q,
        "tau_": args.tau_,
    }
    return config    

class Robust_Loss(torch.nn.Module):
    def __init__(self, config, bit):
        super(Robust_Loss, self).__init__()
        self.shift = 1
        self.margin =.2
        # self.tau = 1.
        self.tau = 1.
      
    def forward(self, u, v, y, config, w=1):
        u = u.tanh()
        v = v.tanh()
        T = self.calc_neighbor (y,y)
        T.diagonal().fill_(0)
        S = u.mm(v.t())
        #pdb.set_trace()
        d = S.diag().view(v.size(0), 1)
        d1 = d.expand_as(S)
        d2 = d.t().expand_as(S)

        mask_te = (S >= (d1 - self.margin)).float().detach()
        cost_te = S * mask_te + (1. - mask_te) * (S - self.shift)

        cost_te_max = torch.zeros_like(cost_te)
        cost_te_max.copy_(cost_te)
        identity_matrix_te = torch.eye(cost_te_max.size(0), cost_te_max.size(1), device=cost_te_max.device, dtype=cost_te_max.dtype)
        diagonal_te = torch.diag(cost_te_max).clamp(min=0)
        modified_diagonal_matrix_te = torch.diag_embed(diagonal_te)
        cost_te_max = cost_te_max * (1 - identity_matrix_te) + modified_diagonal_matrix_te 
 
        mask_im = (S>= (d2 - self.margin)).float().detach()
        cost_im = S * mask_im + (1. - mask_im) * (S - self.shift)

        cost_im_max = torch.zeros_like(cost_im)
        cost_im_max.copy_(cost_im)
        identity_matrix_im = torch.eye(cost_im_max.size(0), cost_im_max.size(1), device=cost_im_max.device, dtype=cost_im_max.dtype)
        diagonal_im = torch.diag(cost_im_max).clamp(min=0)
        modified_diagonal_matrix_im = torch.diag_embed(diagonal_im)
        cost_im_max = cost_im_max * (1 - identity_matrix_im) + modified_diagonal_matrix_im 

        loss_r = (-cost_te.diag()+self.tau * ((cost_te_max / self.tau*(1-T))).exp().sum(1).log() + self.margin) +(-cost_im.diag()+self.tau * ((cost_te_max / self.tau*(1-T))).exp().sum(1).log() + self.margin)
        Q_loss = (u.abs() - 1 / np.sqrt(u.shape[1])).pow(2).mean(axis = 1) + (v.abs() - 1 / np.sqrt(u.shape[1])).pow(2).mean(axis = 1)        
        loss = config["lambda"] *loss_r + (1-config["lambda"])*Q_loss
        
        loss = loss * w 
        final_loss = torch.mean(loss)
        return final_loss
    
    def calc_neighbor(self,label1, label2):
        # calculate the similar matrix
        label1 = label1.type(torch.cuda.FloatTensor)
        label2 = label2.type(torch.cuda.FloatTensor)
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
        return Sim



class PrototypeModule(torch.nn.Module):
    def __init__(self, n_class, bit_len):
        super(PrototypeModule, self).__init__()
        W = torch.Tensor(n_class, bit_len)
        W = torch.nn.init.orthogonal_(W, gain=1)
        W = torch.tensor(W, requires_grad= True).cuda()
        W = torch.nn.Parameter(W)
        self.n_class = n_class
        self.W = W
        self.loss_type = 'RSHNL'
     
    def forward_persample_loss(self, img, txt, label): 
        label_ = (label - 0.5) * 2  #   [-1,1]
        u_sims = img @ self.W.tanh().t()   # N X C  [-1, 1]
        v_sims = txt @ self.W.tanh().t()   # N X C  [-1, 1]
        loss_ = (label_ - u_sims)**2
        loss_ += (label_ - v_sims)**2                 
        loss = (loss_ * label).max(1)[0]
        indices_u = torch.argmax(u_sims, dim=1)
        indices_v = torch.argmax(v_sims, dim=1)
        bs = indices_u.size(0)
        acc = (label[torch.arange(bs),indices_u]  + label[torch.arange(bs),indices_v])/2
       
        return loss, acc.sum()/bs

    def forward_soft_pred(self, img, txt, last_label, alpha = 0.9):
        tau = 0.1
        u_logits = img.detach() @ self.W.tanh().t().detach() / tau
        v_logits = txt.detach() @ self.W.tanh().t().detach() / tau
        u_pred = F.softmax(u_logits, dim=1)
        v_pred = F.softmax(v_logits, dim=1)
        new_label = alpha * last_label + (1 - alpha) * (u_pred + v_pred)/2
        return new_label
    
    # Prototype Loss
    def forward(self, features1, features2,config, labels=None, epoch=0, isclean=None):
        # w=0.9
        gamma = config["gamma"]
        beta = config["beta"]
        lamda_ = config["lamda_"]
        predict1 = F.softmax(features1.view([features1.shape[0], -1]).mm(self.W.T), dim=1)
        predict2 = F.softmax(features2.view([features2.shape[0], -1]).mm(self.W.T), dim=1)
        label_ = (labels - 0.5) * 2  #   [-1,1]
        u_sims = features1 @ self.W.tanh().t() # N X C  [-1, 1]
        v_sims = features2 @ self.W.tanh().t() # N X C  [-1, 1]
        q = config["q"]
        tmp1 = (1 - q) * (1. - torch.sum(labels.float() * predict1, dim=1) ** q).div(q) + q * (1 - torch.sum(labels.float() * predict1, dim=1))
        tmp2 = (1 - q) * (1. - torch.sum(labels.float() * predict2, dim=1) ** q).div(q) + q * (1 - torch.sum(labels.float() * predict2, dim=1))
        term1 = tmp1 + tmp2

        term3 = cross_modal_contrastive_ctriterion_q([features1, features2], tau_=config["tau_"], q=q)
        loss = (label_ - u_sims)**2
        loss += (label_ - v_sims)**2
        loss = loss.mean() 
        labels = (torch.eye(self.n_class).cuda() - 0.5) * 2
        p_loss = (self.W.tanh() @ self.W.tanh().t() - labels)**2
        loss += p_loss.mean()
        return (lamda_ * term1.mean() + beta * term3)+loss
    
def cross_modal_contrastive_ctriterion_q(fea, tau_=1., q = 1):
        # q = opt.q
        n_view = 2
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())
        sim = (sim / tau_).exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        p1 = diag1 / sim.sum(1)
        loss1 = (1 - q) * (1. - (p1) ** q).div(q) + q * (1 - p1)
        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        p2 = diag2 / sim.sum(1)
        loss2 = (1 - q) * (1. - (p2) ** q).div(q) + q * (1 - p2)
        return loss1.mean() + loss2.mean()
 
def build_c():
    net = ImgModule(y_dim=4096, bit=bit, mid_num1=2048, hiden_layer=4).to('cuda')
    txt_net = TxtModule(y_dim=tag_len, bit=bit, mid_num1=2048, hiden_layer=4).to('cuda')
    p_net = PrototypeModule(n_class, bit_len).to('cuda')
 
    params_list =  [
            {'params':  net.parameters(), 'lr': config["optimizer"]["optim_params"]['lr'], 'weight_decay':config["optimizer"]["optim_params"]['weight_decay']},
            {'params':  txt_net.parameters(), 'lr': config["txt_optimizer"]["optim_params"]['lr'], 'weight_decay':config["optimizer"]["optim_params"]['weight_decay']},
            {'params':  p_net.parameters(), 'lr': config["p_optimizer"]["optim_params"]['lr'], 'weight_decay':config["optimizer"]["optim_params"]['weight_decay']},
        ]
    optimizer = torch.optim.AdamW(params_list, lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.2, last_epoch=-1, verbose='deprecated')

    return net, txt_net, p_net, optimizer, scheduler

def train(config, bit):
    device = config["device"]
    train_loader,  test_loader, dataset_loader, num_train,  num_test, num_dataset = get_data(config)

    net, txt_net, p_net, optimizer, scheduler = build_c()
    init_epoches = [21, 41]

    criterion = Robust_Loss(config, bit)

    i2t_mAP_list = []
    t2i_mAP_list = []
    epoch_list = []
    precision_list = []
    bestt2i=0
    besti2t=0
    n=0
    os.makedirs('./checkpoint', exist_ok=True)
    
    soft_labels1 = torch.zeros((len(train_loader.dataset), n_class)).long()
    soft_labels2 = torch.zeros((len(train_loader.dataset), n_class)).long()
    gt_labels = torch.zeros((len(train_loader.dataset), n_class)).long()
    for image, tag, tlabel, label, ind in train_loader:  
        soft_labels1[ind] = label 
        soft_labels2[ind] = label 
        gt_labels[ind] = tlabel 
    soft_labels1 = soft_labels1.float()
    soft_labels2 = soft_labels2.float()
    gt_labels = gt_labels.float() 
    
    warm_epoch = 2
    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")  

        if (epoch+1) %20 == 0:
            net.eval()
            txt_net.eval()
            p_net.eval() 
            print("calculating test binary code......")
            img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
            print("calculating dataset binary code.......")
            img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
            txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
            txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
            print("calculating map.......")
            t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(), txt_tst_label.numpy())
            i2t_mAP = calc_map_k(txt_trn_binary.numpy(),img_tst_binary.numpy(), txt_trn_label.numpy(), img_tst_label.numpy())
            if t2i_mAP+i2t_mAP> bestt2i+besti2t:
                bestt2i=t2i_mAP
                besti2t=i2t_mAP
                torch.save({
                    'net_state_dict': net.state_dict(),
                    'txt_net_state_dict': txt_net.state_dict(),
                    'p_net_state_dict': p_net.state_dict(),
                }, './checkpoint/best_model.pth') 
            t2i_mAP_list.append(t2i_mAP.item())
            i2t_mAP_list.append(i2t_mAP.item())
            epoch_list.append(epoch)
            print("%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], config["noise_rate"],t2i_mAP, i2t_mAP))
        
        net.train()
        txt_net.train()
        p_net.train()
        train_loss = 0

        for image, tag, tlabel, label, ind in train_loader:
            image = image.to('cuda').float() 
            tag = tag.to('cuda').float() 
            label = label.to('cuda') 
            tlabel = tlabel.to('cuda') 
            optimizer.zero_grad()
            
            loss = 0
            u1, u2 = net(image, True)
            v1, v2 = txt_net(tag, True) 

            soft_pred1 = p_net.forward_soft_pred(u1, v1, soft_labels1[ind].to('cuda')).detach()
            soft_labels1[ind] = soft_pred1.clone().cpu()

            soft_pred2 = p_net.forward_soft_pred(u2, v2, soft_labels2[ind].to('cuda')).detach()
            soft_labels2[ind] = soft_pred2.clone().cpu()

            # confident div
            indices_1 = torch.argmax(soft_labels1[ind], dim=1)
            indices_2 = torch.argmax(soft_labels2[ind], dim=1)
            
            d_l = label.size(0)
            div_metric = label[torch.arange(d_l), indices_1] + label[torch.arange(d_l), indices_2]
            div_label = (div_metric > 0.).float().view(d_l, 1) # 0 1 2 

            r_label = torch.zeros_like(label).to(label.device)
            indices_es = torch.argmax(soft_labels1[ind] + soft_labels2[ind], dim=1)
            r_label[torch.arange(d_l), indices_es] += 1


            # print(div_label)
            r_label = r_label * (1. - div_label) + label * div_label

            loss += criterion(u1, v1, r_label.float(), config)
            loss += criterion(u2, v2, r_label.float(), config)
            # loss += criterion(u, v, label.float(), config)
            loss += p_net.forward(u1, v1, config, r_label.float()) 
            loss += p_net.forward(u2, v2, config, r_label.float())

            train_loss += loss 
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Calculate the accuracy of the correction
        indices_u = torch.argmax(soft_labels1, dim=1)
        indices_v = torch.argmax(soft_labels1, dim=1)
        d_l = gt_labels.size(0)
        print(sum(gt_labels[torch.arange(d_l), indices_u])/d_l, sum(gt_labels[torch.arange(d_l), indices_v])/d_l)
        indices_u = torch.argmax(soft_labels2, dim=1)
        indices_v = torch.argmax(soft_labels2, dim=1)
        d_l = gt_labels.size(0)
        print(sum(gt_labels[torch.arange(d_l), indices_u])/d_l, sum(gt_labels[torch.arange(d_l), indices_v])/d_l)
 
       
        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        # precision_list.append(precision)
        print("%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f" % (
                config["info"], epoch + 1, bit, config["dataset"], config["noise_rate"]))
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
    
def test(config, bit, model_path='./checkpoint/best_model.pth'):
    global df_results  # 使用全局 DataFrame
    device = config["device"]
    _, test_loader, dataset_loader, _, _, _ = get_data(config)
    net, txt_net, p_net, optimizer, scheduler = build_c()
 
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    txt_net.load_state_dict(checkpoint['txt_net_state_dict'])
    p_net.load_state_dict(checkpoint['p_net_state_dict'])
    net.eval()
    txt_net.eval()
    p_net.eval()
    print("calculating test binary code......")
    print("calculating test binary code......")
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
    txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
    txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
    print("calculating map.......")
    t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(), txt_tst_label.numpy())
    i2t_mAP = calc_map_k(txt_trn_binary.numpy(),img_tst_binary.numpy(), txt_trn_label.numpy(), img_tst_label.numpy())
    print("Test Results: t2i_mAP: %.3f, i2t_mAP: %.3f" % (t2i_mAP, i2t_mAP))

if __name__ == "__main__":
    data_name_list = ['flickr','nuswide21','iapr','ms-coco']
    bit_list=[16,32,64,128]
    noise_rate_list = [0.2,0.5,0.8]
    
    for data_name in data_name_list:
        for g in gamma_list:
            for rate in noise_rate_list:
                for bit in bit_list:
                
                    bit_len = bit
                    noise_rate = rate
                    dataset = data_name
                    args.gamma=g
                    gamma=g
                    if dataset == 'nuswide21':
                        n_class = 21
                        tag_len = 1000
                    elif dataset == 'flickr':
                        n_class = 24
                        tag_len = 1386                        
                    elif dataset == 'ms-coco':
                        n_class = 80
                        tag_len = 300
                    elif dataset == 'iapr':
                        n_class = 255
                        tag_len = 2912
                    config = get_config()
                    print(config)
                    train(config, bit)
                    test(config, bit)

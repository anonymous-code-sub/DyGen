import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle
import os
import torch.nn.functional as F
from torch.autograd import Variable
from network import *
from dataloader import load_dataset
from utils import *
from utils2 import *

def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)

class text_NPC:
    def __init__(self, args, grad_scaler, train_z0_dataloader, val_z0_dataloader, test_z0_dataloader, len_train, len_val, len_test, emb_dim):
        self.args = args
        self.grad_scaler = grad_scaler
        self.time = time.time()

        self.train_loader = train_z0_dataloader
        self.val_loader = val_z0_dataloader
        self.test_loader = test_z0_dataloader
        self.len_train = len_train
        self.len_val = len_val
        self.len_test = len_test

        print('\n===> AE Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.emb_dim = emb_dim
        self.AE = text_CVAE(self.args, self.emb_dim)
        self.AE.to(self.device)
        self.optimizer = optim.Adam(self.AE.parameters(), lr=self.args.lr)

    def generate_alpha_from_original_proba(self, y_prior):
        if self.args.knn_mode=='onehot':
            # if self.args.stage1 == 'base':
            proba = torch.zeros(len(y_prior), self.args.num_classes).to(self.device)
            return proba.scatter(1,y_prior.view(-1, 1), self.args.prior_norm)+1.0+1/self.args.num_classes
            # elif self.args.stage1 == 'cr':
            #     proba_list = []
            #     for i in range(y_prior.shape[-1]):
            #         y_prior_temp = y_prior[:, i].squeeze()
            #         proba = torch.zeros(len(y_prior_temp), self.args.num_classes).to(self.device)
            #         proba = proba.scatter(1,y_prior_temp.view(-1, 1), self.args.prior_norm)+1.0+1/self.args.num_classes
            #         proba_list.append(proba)
            #     # proba_list = torch.stack(proba_list, 0)
            #     return proba_list
        elif self.args.knn_mode=='proba': # proba
            if self.args.selected_class == self.args.num_classes:
                return self.args.prior_norm * y_prior + 1.0 + 1 / self.args.num_classes
            else: # topk
                values, indices = torch.topk(y_prior, k=int(self.args.selected_class), dim=1)
                y_prior_temp = Variable(torch.zeros_like(y_prior)).to(self.device)
                return self.args.prior_norm * y_prior_temp.scatter(1, indices, values) + 1.0 + 1 / self.args.num_classes

    def loss_function(self, y_tilde_data, y_tilde_recon, alpha_prior, alpha_infer):
        
        recon_loss = nn.BCEWithLogitsLoss(reduction='sum')(y_tilde_recon.float(), y_tilde_data.float())

        KL = torch.sum(torch.lgamma(alpha_prior) - torch.lgamma(alpha_infer) + (alpha_infer - alpha_prior) *
                         torch.digamma(alpha_infer), dim=1)

        return recon_loss, torch.sum(KL)

    def update_model(self):
        self.AE.train()
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
        batch = 0

        for step, batch in enumerate(self.train_loader):
            batch = tuple(t.to(self.args.device) for t in batch)
            b_inputs, b_priors, b_labels = batch
            label_one_hot = F.one_hot(b_labels, self.args.num_classes)
            # label_one_hot = label_one_hot.scatter(1, self.y_hat[indexes].view(-1, 1), 1).to(self.device)
            alpha_prior = self.generate_alpha_from_original_proba(b_priors[:,-1].squeeze())
            # if self.args.stage1 == 'cr':
            #     alpha_prior = alpha_prior[-1]
            # if self.args.stage2 == 'base':
            #     alpha_prior = alpha_prior[-1]

            with torch.cuda.amp.autocast():
                y_recon, alpha_infer = self.AE(b_inputs[:, -1, :].squeeze(), label_one_hot)
                recon_loss, kl_loss = self.loss_function(label_one_hot, y_recon, alpha_prior, alpha_infer)
                loss = recon_loss + self.args.beta * kl_loss

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

            # print(step, loss.item())

        time_elapse = time.time() - self.time

        return epoch_loss, epoch_recon, epoch_kl, time_elapse

    def save_result(self, train_loss_total, train_loss_recon, train_loss_kl):
        self.train_loss.append(train_loss_total/self.len_train)
        self.train_recon_loss.append(train_loss_recon/self.len_train)
        self.train_kl_loss.append(train_loss_kl/self.len_train)

        print('Train', train_loss_total/self.len_train, train_loss_recon/self.len_train)

        return

    def key_name(self, k):
        if self.args.model == 'CNN_MNIST':
            last_layer = 'linear3'
        elif self.args.model == 'CNN_CIFAR':
            last_layer = 'l_c1'
        elif self.args.model == 'Resnet50Pre':
            last_layer = 'model'
        else:
            last_layer = None

        res = k.split('.')

        if self.args.class_method == 'causalNL':
            return res[0] == 'y_encoder' and (res[1] != last_layer if 'CNN' in self.args.model else res[1] == last_layer)
        else:
            return res[0] != last_layer if 'CNN' in self.args.model else res[0] == last_layer

    def run(self, func, test_z_dataloader, best_model):
        # initialize
        self.train_loss, self.train_recon_loss, self.train_kl_loss = [], [], []

        # update feature extractor
        # old_dict = torch.load(self.args.model_dir + 'classifier.pk')
        # change_dict = self.AE.state_dict()

        # change_dict.update(old_dict)
        # self.AE.load_state_dict(change_dict)

        # train model using train dataset
        self.accuracy_list = []
        for epoch in range(self.args.total_iter):
            train_loss, train_recon, train_kl, time_train = self.update_model()
            # print result loss
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train)
            self.save_result(train_loss, train_recon, train_kl)
            # if epoch % 5 == 0:
            accuracy_pre, accuracy = func.merge_classifier_and_autoencoder(test_z_dataloader, best_model, self.AE, self.args.vae_batch_size)
            self.accuracy_list.append(accuracy)
            print("The test performance after NPC:", accuracy)
        # plot_(self.args.gen_dir, self.train_loss, 'train_loss')
        self.train_loss = torch.tensor(self.train_loss).numpy()
        self.train_recon_loss = torch.tensor(self.train_recon_loss).numpy()
        self.train_kl_loss = torch.tensor(self.train_kl_loss).numpy()
        self.accuracy_list = torch.tensor(self.accuracy_list).numpy()
        np.save('./train_loss.npy', self.train_loss)
        np.save('./train_recon_loss.npy', self.train_recon_loss)
        np.save('./train_kl_loss.npy', self.train_kl_loss)
        np.save(self.args.save_path, self.accuracy_list)
        # plot_(self.args.gen_dir, self.train_recon_loss, 'train_recon_loss')
        # plot_(self.args.gen_dir, self.train_kl_loss, 'train_kl_loss')

        # torch.save(self.AE.state_dict(), self.args.gen_model_dir+'_AE.pk')
        # results = self.evaluate()
        return self.AE

    def evaluate(self):
        logits_list = None
        test_acc = 0
        for step, batch in enumerate(self.test_loader):
            batch = tuple(t.to(self.args.device) for t in batch)
            b_inputs, b_masks, b_labels, b_z0 = batch
            label_one_hot = F.one_hot(b_labels, self.args.num_classes)
            y_recon, alpha_infer = self.AE(b_inputs, label_one_hot)
            if logits_list == None:
                logits_list = y_recon
            else:
                logits_list = torch.cat((logits_list, y_recon), 0)
        return logits_list

    def accurate_nb(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)

class text_CRNPC:
    def __init__(self, args, grad_scaler, train_z0_dataloader, val_z0_dataloader, test_z0_dataloader, len_train, len_val, len_test, emb_dim):
        self.args = args
        self.grad_scaler = grad_scaler
        self.time = time.time()

        self.train_loader = train_z0_dataloader
        self.val_loader = val_z0_dataloader
        self.test_loader = test_z0_dataloader
        self.len_train = len_train
        self.len_val = len_val
        self.len_test = len_test

        print('\n===> AE Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.emb_dim = emb_dim
        self.AEs = nn.ModuleList()
        for i in range(args.n_model):
            AE = text_CVAE(self.args, self.emb_dim)
            AE.to(self.device)
            self.AEs.append(AE)

        self.optimizer = optim.Adam(self.AEs.parameters(), lr=self.args.lr)

    def generate_alpha_from_original_proba(self, y_prior):
        if self.args.knn_mode=='onehot':
            proba = torch.zeros(len(y_prior), self.args.num_classes).to(self.device)
            return proba.scatter(1,y_prior.view(-1, 1), self.args.prior_norm)+1.0+1/self.args.num_classes
        elif self.args.knn_mode=='proba': # proba
            if self.args.selected_class == self.args.num_classes:
                return self.args.prior_norm * y_prior + 1.0 + 1 / self.args.num_classes
            else: # topk
                values, indices = torch.topk(y_prior, k=int(self.args.selected_class), dim=1)
                y_prior_temp = Variable(torch.zeros_like(y_prior)).to(self.device)
                return self.args.prior_norm * y_prior_temp.scatter(1, indices, values) + 1.0 + 1 / self.args.num_classes

    def loss_function(self, y_tilde_data, y_tilde_recon, alpha_prior, alpha_infer):
        
        num_models = len(self.AEs)
        recon_loss_b = 0
        KL_loss_b = 0
        reg_loss_b = 0
        alpha = alpha_infer - 1
        alpha = alpha / torch.sum(alpha, dim=-1).unsqueeze(-1)
        avg_alpha = torch.mean(alpha, 0)
        for i in range(num_models):
            recon_loss = nn.BCEWithLogitsLoss(reduction='sum')(y_tilde_recon[i].float().squeeze(), y_tilde_data.float())

            KL = torch.sum(torch.lgamma(alpha_prior[i]) - torch.lgamma(alpha_infer[i]) + (alpha_infer[i] - alpha_prior[i]) *
                            torch.digamma(alpha_infer[i]), dim=1)

            recon_loss_b += recon_loss
            KL_loss_b += torch.sum(KL)
            temp_alpha = alpha_infer[i] - 1
            temp_alpha = temp_alpha / torch.sum(temp_alpha, dim=-1).unsqueeze(-1)
            # print(avg_alpha.shape, temp_alpha.shape)
            reg_loss = kl_div(avg_alpha.squeeze(), temp_alpha.squeeze())
            reg_loss_b += torch.sum(reg_loss)
            # print(recon_loss, torch.sum(KL), reg_loss)
        #     print('=========>', recon_loss.device, KL.device, reg_loss.device)
        # recon_loss_list = torch.tensor(recon_loss_list)
        # KL_loss_list = torch.tensor(KL_loss_list)
        # reg_loss_list = torch.tensor(reg_loss_list)
        # print('=====^^^====>', recon_loss_list.device, KL_loss_list.device, reg_loss_list.device)
        return recon_loss_b/num_models, KL_loss_b/num_models, reg_loss_b/num_models

    def update_model(self):
        self.AEs.train()
        num_models = len(self.AEs)
        epoch_loss, epoch_recon, epoch_kl, epoch_reg = 0, 0, 0, 0
        batch = 0

        for step, batch in enumerate(self.train_loader):
            batch = tuple(t.to(self.args.device) for t in batch)
            b_inputs, b_priors, b_labels = batch
            label_one_hot = F.one_hot(b_labels, self.args.num_classes)
            # label_one_hot = label_one_hot.scatter(1, self.y_hat[indexes].view(-1, 1), 1).to(self.device)
            if self.args.stage1 == 'base':
                b_inputs = b_inputs.repeat((1, num_models, 1))
                alpha_prior = [None] * num_models
                for idx in range(num_models):
                    alpha_prior[idx] = self.generate_alpha_from_original_proba(b_priors)
            elif self.args.stage1 == 'cr':
                alpha_prior = [None] * num_models
                for idx in range(num_models):
                    alpha_prior[idx] = self.generate_alpha_from_original_proba(b_priors[:, idx].squeeze())

            with torch.cuda.amp.autocast():
                y_recon_list = []
                alpha_infer_list = []
                for idx in range(num_models):
                    y_recon, alpha_infer = self.AEs[idx](b_inputs[:,idx,:].squeeze(), label_one_hot)
                    # print('------>', y_recon.device, alpha_infer.device)
                    y_recon_list.append(y_recon)
                    alpha_infer_list.append(alpha_infer)
                y_recon_list = torch.stack(y_recon_list, dim=0)
                alpha_infer_list = torch.stack(alpha_infer_list, dim=0)
                recon_loss, kl_loss, reg_loss = self.loss_function(label_one_hot, y_recon_list, alpha_prior, alpha_infer_list)
                loss = recon_loss + self.args.beta * kl_loss + self.lambda_t * reg_loss
            # print(recon_loss.device, kl_loss.device, reg_loss.device)
            # print(loss)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_recon += torch.mean(recon_loss).item()
            epoch_kl += torch.mean(kl_loss).item()
            epoch_reg += torch.mean(reg_loss).item()

            # print(step, loss.item())

        time_elapse = time.time() - self.time

        return epoch_loss, epoch_recon, epoch_kl, epoch_reg, time_elapse

    def save_result(self, train_loss_total, train_loss_recon, train_loss_kl):
        self.train_loss.append(train_loss_total/self.len_train)
        self.train_recon_loss.append(train_loss_recon/self.len_train)
        self.train_kl_loss.append(train_loss_kl/self.len_train)

        print('Train', train_loss_total/self.len_train, train_loss_recon/self.len_train)

        return

    def run(self, func, test_z_dataloader, best_model):
        # initialize
        self.train_loss, self.train_recon_loss, self.train_kl_loss, self.train_reg_loss = [], [], [], []

        self.accuracy_list = []
        for epoch in range(self.args.total_iter):
            if epoch < self.args.warmup_epochs * self.args.total_iter:
                self.lambda_t = 0
            else:
                self.lambda_t = self.args.lambda_t
            train_loss, train_recon, train_kl, train_reg, time_train = self.update_model()
            # print result loss
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train)
            self.save_result(train_loss, train_recon, train_kl)
            # if epoch % 5 == 0:
            accuracy_pre, accuracy = func.merge_classifier_and_autoencoder(test_z_dataloader, best_model, self.AEs, self.args.vae_batch_size)
            self.accuracy_list.append(accuracy)
            print("The test performance after NPC:", accuracy)
        # plot_(self.args.gen_dir, self.train_loss, 'train_loss')
        self.train_loss = torch.tensor(self.train_loss).numpy()
        self.train_recon_loss = torch.tensor(self.train_recon_loss).numpy()
        self.train_kl_loss = torch.tensor(self.train_kl_loss).numpy()
        self.train_reg_loss = torch.tensor(self.train_reg_loss).numpy()
        self.accuracy_list = torch.tensor(self.accuracy_list).numpy()
        np.save('./train_loss.npy', self.train_loss)
        np.save('./train_recon_loss.npy', self.train_recon_loss)
        np.save('./train_kl_loss.npy', self.train_kl_loss)
        np.save('./train_reg_loss.npy', self.train_reg_loss)
        np.save(self.args.save_path, self.accuracy_list)
        # plot_(self.args.gen_dir, self.train_recon_loss, 'train_recon_loss')
        # plot_(self.args.gen_dir, self.train_kl_loss, 'train_kl_loss')

        # torch.save(self.AE.state_dict(), self.args.gen_model_dir+'_AE.pk')
        # results = self.evaluate()
        return self.AEs

    def evaluate(self):
        logits_list = None
        test_acc = 0
        for step, batch in enumerate(self.test_loader):
            batch = tuple(t.to(self.args.device) for t in batch)
            b_inputs, b_masks, b_labels, b_z0 = batch
            label_one_hot = F.one_hot(b_labels, self.args.num_classes)
            y_recon, alpha_infer = self.AE(b_inputs, label_one_hot)
            if logits_list == None:
                logits_list = y_recon
            else:
                logits_list = torch.cat((logits_list, y_recon), 0)
        return logits_list

    def accurate_nb(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)

class text_MBRNPC:
    def __init__(self, args, grad_scaler, train_z0_dataloader, val_z0_dataloader, test_z0_dataloader, len_train, len_val, len_test, emb_dim):
        self.args = args
        self.grad_scaler = grad_scaler
        self.time = time.time()

        self.train_loader = train_z0_dataloader
        self.val_loader = val_z0_dataloader
        self.test_loader = test_z0_dataloader
        self.len_train = len_train
        self.len_val = len_val
        self.len_test = len_test

        print('\n===> AE Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.emb_dim = emb_dim
        self.AEs = nn.ModuleList()
        for i in range(args.n_model):
            AE = text_CVAE(self.args, self.emb_dim)
            AE.to(self.device)
            self.AEs.append(AE)

        self.optimizer = optim.Adam(self.AEs.parameters(), lr=self.args.lr)

    def generate_alpha_from_original_proba(self, y_prior):
        if self.args.knn_mode=='onehot':
            proba = torch.zeros(len(y_prior), self.args.num_classes).to(self.device)
            return proba.scatter(1,y_prior.view(-1, 1), self.args.prior_norm)+1.0+1/self.args.num_classes
        elif self.args.knn_mode=='proba': # proba
            if self.args.selected_class == self.args.num_classes:
                return self.args.prior_norm * y_prior + 1.0 + 1 / self.args.num_classes
            else: # topk
                values, indices = torch.topk(y_prior, k=int(self.args.selected_class), dim=1)
                y_prior_temp = Variable(torch.zeros_like(y_prior)).to(self.device)
                return self.args.prior_norm * y_prior_temp.scatter(1, indices, values) + 1.0 + 1 / self.args.num_classes

    def loss_function(self, y_tilde_data, y_tilde_recon, alpha_prior, alpha_infer):
        
        num_models = len(self.AEs)
        recon_loss_list = []
        KL_loss_list = []
        reg_loss_list = []
        alpha = alpha_infer - 1
        alpha = alpha / torch.sum(alpha, dim=-1).unsqueeze(-1)
        avg_alpha = torch.mean(alpha, 0)
        for i in range(num_models):
            recon_loss = nn.BCEWithLogitsLoss(reduction='sum')(y_tilde_recon[i].float(), y_tilde_data.float())

            KL = torch.sum(torch.lgamma(alpha_prior[i]) - torch.lgamma(alpha_infer[i]) + (alpha_infer[i] - alpha_prior[i]) *
                            torch.digamma(alpha_infer[i]), dim=1)

            recon_loss_list.append(recon_loss)
            KL_loss_list.append(torch.sum(KL))
            temp_alpha = alpha_infer[i] - 1
            temp_alpha = temp_alpha / torch.sum(temp_alpha, dim=-1).unsqueeze(-1)
            reg_loss_list.append(kl_div(avg_alpha, temp_alpha))
        recon_loss_list = torch.tensor(recon_loss_list)
        KL_loss_list = torch.tensor(KL_loss_list)
        reg_loss_list = torch.tensor(reg_loss_list)
        return recon_loss_list, KL_loss_list, reg_loss_list

    def update_model(self):
        self.AEs.train()
        num_models = len(self.AEs)
        epoch_loss, epoch_recon, epoch_kl, epoch_reg = 0, 0, 0, 0
        batch = 0

        for step, batch in enumerate(self.train_loader):
            batch = tuple(t.to(self.args.device) for t in batch)
            b_inputs, b_priors, b_labels = batch
            label_one_hot = F.one_hot(b_labels, self.args.num_classes)
            # label_one_hot = label_one_hot.scatter(1, self.y_hat[indexes].view(-1, 1), 1).to(self.device)
            # alpha_prior = self.generate_alpha_from_original_proba(b_priors)

            with torch.cuda.amp.autocast():
                y_recon_list = []
                alpha_infer_list = []
                alpha_prior_list = []
                for idx in range(num_models):
                    alpha_prior = self.generate_alpha_from_original_proba(b_priors[idx])
                    y_recon, alpha_infer = self.AEs[idx](b_inputs[:,idx,:].squeeze(), label_one_hot)
                    # print('------>', y_recon.device, alpha_infer.device)
                    y_recon_list.append(y_recon.unsqueeze(0))
                    alpha_infer_list.append(alpha_infer.unsqueeze(0))
                    alpha_prior_list.append(alpha_prior.unsqueeze(0))

                y_recon_list = torch.stack(y_recon_list, dim=0)
                alpha_infer_list = torch.stack(alpha_infer_list, dim=0)
                alpha_prior_list = torch.stack(alpha_prior_list, dim=0)
                recon_loss, kl_loss, reg_loss = self.loss_function(label_one_hot, y_recon_list, alpha_prior_list, alpha_infer_list)
                loss = torch.mean(recon_loss) + torch.mean(kl_loss) + self.lambda_t * torch.mean(reg_loss)

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_recon += torch.mean(recon_loss).item()
            epoch_kl += torch.mean(kl_loss).item()
            epoch_reg += torch.mean(reg_loss).item()

            # print(step, loss.item())

        time_elapse = time.time() - self.time

        return epoch_loss, epoch_recon, epoch_kl, epoch_reg, time_elapse

    def save_result(self, train_loss_total, train_loss_recon, train_loss_kl):
        self.train_loss.append(train_loss_total/self.len_train)
        self.train_recon_loss.append(train_loss_recon/self.len_train)
        self.train_kl_loss.append(train_loss_kl/self.len_train)

        print('Train', train_loss_total/self.len_train, train_loss_recon/self.len_train)

        return

    def run(self, func, test_z_dataloader, best_model):
        # initialize
        self.train_loss, self.train_recon_loss, self.train_kl_loss, self.train_reg_loss = [], [], [], []

        self.accuracy_list = []
        for epoch in range(self.args.total_iter):
            if epoch < 5:
                self.lambda_t = 0
            else:
                self.lambda_t = 5
            train_loss, train_recon, train_kl, train_reg, time_train = self.update_model()
            # print result loss
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train)
            self.save_result(train_loss, train_recon, train_kl)
            if epoch % 5 == 0:
                accuracy_pre, accuracy = func.merge_classifier_and_autoencoder(test_z_dataloader, best_model, self.AEs, self.args.vae_batch_size)
                self.accuracy_list.append(accuracy)
                print("The test performance after NPC:", accuracy)
        # plot_(self.args.gen_dir, self.train_loss, 'train_loss')
        self.train_loss = torch.tensor(self.train_loss).numpy()
        self.train_recon_loss = torch.tensor(self.train_recon_loss).numpy()
        self.train_kl_loss = torch.tensor(self.train_kl_loss).numpy()
        self.train_reg_loss = torch.tensor(self.train_reg_loss).numpy()
        self.accuracy_list = torch.tensor(self.accuracy_list).numpy()
        np.save('./train_loss.npy', self.train_loss)
        np.save('./train_recon_loss.npy', self.train_recon_loss)
        np.save('./train_kl_loss.npy', self.train_kl_loss)
        np.save('./train_reg_loss.npy', self.train_reg_loss)
        np.save(self.args.save_path, self.accuracy_list)
        # plot_(self.args.gen_dir, self.train_recon_loss, 'train_recon_loss')
        # plot_(self.args.gen_dir, self.train_kl_loss, 'train_kl_loss')

        # torch.save(self.AE.state_dict(), self.args.gen_model_dir+'_AE.pk')
        # results = self.evaluate()
        return self.AEs

    def evaluate(self):
        logits_list = None
        test_acc = 0
        for step, batch in enumerate(self.test_loader):
            batch = tuple(t.to(self.args.device) for t in batch)
            b_inputs, b_masks, b_labels, b_z0 = batch
            label_one_hot = F.one_hot(b_labels, self.args.num_classes)
            y_recon, alpha_infer = self.AE(b_inputs, label_one_hot)
            if logits_list == None:
                logits_list = y_recon
            else:
                logits_list = torch.cat((logits_list, y_recon), 0)
        return logits_list

    def accurate_nb(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)
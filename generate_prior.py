import time
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

from network import *
from dataloader import load_dataset

class KNN_prior_simple:
    def __init__(self, args, dataset, z0, noisy_train_labels, true_train_labels):
        self.args = args
        self.n_classes = self.args.num_classes
        self.time = time.time()

        self.dataset = dataset
        self.y_hat = noisy_train_labels
        self.y = true_train_labels
        self.z0 = z0
        self.emb_dim = z0.shape[-1]

        print('\n===> Prior Generation with KNN Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For each datapoint, get initial class name
    def get_prior(self, best_model, knn_dataloader):
        # Load Classifier
        self.net = best_model
        self.net.to(self.device)

        # k-nn
        self.net.eval()
        # knn mode on
        neigh = KNeighborsClassifier(n_neighbors=self.args.prior_k, weights='distance')
        embeddings, class_confi = [], []
        embeddings = self.z0
        class_confi = self.y_hat
        neigh.fit(embeddings, class_confi)
        print('Time : ', time.time() - self.time)

        class_preds = neigh.predict(embeddings)
        class_preds = torch.tensor(np.int64(class_preds))
        print('Prior made {} errors with train/val noisy labels'.format(torch.sum(class_preds!=self.y_hat)))
        print('Prior made {} errors with train/val clean labels'.format(torch.sum(class_preds!=self.y)))
        noisy_preds = torch.tensor([(class_preds[i] != self.y_hat[i]) and (class_preds[i] == self.y[i]) for i in range(len(class_preds))])
        print('Prior detected {} real noisy samples'.format(torch.sum(noisy_preds)))

        # # proba
        dict = {}
        model_output = neigh.predict_proba(embeddings)
        # print(model_output)
        if model_output.shape[1] < self.n_classes:
            tmp = np.zeros((model_output.shape[0], self.n_classes))
            tmp[:, neigh.classes_] = neigh.predict_proba(embeddings)
            dict['proba'] = tmp
        else:
            dict['proba'] = model_output  # data*n_class

        print('Time : ', time.time() - self.time, 'proba information saved')


        return dict['proba']

class KNN_prior_base:
    def __init__(self, args, dataset, z0, noisy_train_labels, true_train_labels):
        self.args = args
        self.n_classes = self.args.num_classes
        self.time = time.time()

        self.dataset = dataset
        self.y_hat = noisy_train_labels
        self.y = true_train_labels
        self.z0 = z0
        self.emb_dim = z0.shape[-1]

        print('\n===> Prior Generation with KNN Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For each datapoint, get initial class name
    def get_prior(self, best_model, knn_dataloader):
        # Load Classifier
        self.net = best_model

        # self.emb_dim = 768

        # self.net.load_state_dict(torch.load(self.args.model_dir +'classifier.pk'))
        self.net.to(self.device)

        # k-nn
        self.net.eval()
        # knn mode on
        neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
        embeddings, class_confi = [], []
        # embeddings = self.z0
        knn_embeds = None
        for batch in knn_dataloader:
            batch = tuple(t.to(self.args.device) for t in batch)
            b_input_ids, b_input_mask, b_z0, b_labels = batch
            tmp_dict = {}
            with torch.no_grad():
                if self.args.stage1 == 'base':
                    outputs = self.net(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                else:
                    outputs = self.net(b_input_ids, attention_mask=b_input_mask)
                    outputs = outputs[0]
            for i in range(self.n_classes):
                tmp_dict[i] = []
            for i, lbl in enumerate(b_labels.cpu().tolist()):
                tmp_dict[lbl].append(i)
            mu = outputs[-1][-1][:,0,:].squeeze()
            if knn_embeds == None:
                knn_embeds = mu
            else:
                knn_embeds = torch.cat((knn_embeds, mu), 0)
            output = outputs[0]
            output = F.softmax(output, dim=-1).cpu().detach()
            for i in range(self.n_classes):
                if len(tmp_dict[i]) == 0:
                    continue
                tmp_array = torch.tensor(tmp_dict[i])
                _, index = torch.sort(torch.gather(output[tmp_array], 1, b_labels[tmp_array].view(-1, 1).cpu().detach()).squeeze(1))
                # index = index.cpu().detach()
                embeddings.append(mu[tmp_array[index[-1]]].cpu().detach().tolist())
                class_confi.append(i)

        # class_confi = self.y_hat
        class_confi = np.array(class_confi)
        neigh.fit(embeddings, class_confi)
        print('Time : ', time.time() - self.time)

        # 2. predict class of training dataset
        knn_embeds = knn_embeds.cpu().detach().numpy()


        # onehot
        class_preds = neigh.predict(knn_embeds)
        class_preds = torch.tensor(np.int64(class_preds))
        print('Prior made {} errors with train/val noisy labels'.format(torch.sum(class_preds!=self.y_hat)))
        print('Prior made {} errors with train/val clean labels'.format(torch.sum(class_preds!=self.y)))
        noisy_preds = torch.tensor([(class_preds[i] != self.y_hat[i]) and (class_preds[i] == self.y[i]) for i in range(len(class_preds))])
        print('Prior detected {} real noisy samples'.format(torch.sum(noisy_preds)))

        # # proba
        dict = {}
        model_output = neigh.predict_proba(knn_embeds)
        # print(model_output)
        if model_output.shape[1] < self.n_classes:
            tmp = np.zeros((model_output.shape[0], self.n_classes))
            tmp[:, neigh.classes_] = neigh.predict_proba(embeddings)
            dict['proba'] = tmp
        else:
            dict['proba'] = model_output  # data*n_class

        # with open(os.path.join(self.args.cls_dir, 'proba_'+self.args.data_name), "wb") as f:
        #     pickle.dump(dict, f)s
        # f.close()

        print('Time : ', time.time() - self.time, 'proba information saved')


        return dict['proba'], class_preds.numpy()

class KNN_prior_dynamic:
    def __init__(self, args, dataset, z0, noisy_train_labels, true_train_labels, noisy_markers):
        self.args = args
        self.n_classes = self.args.num_classes
        self.time = time.time()

        self.dataset = dataset
        self.y_hat = noisy_train_labels
        self.y = true_train_labels
        self.noisy_markers = noisy_markers
        self.z0 = z0
        self.emb_dim = z0.shape[-1]

        print('\n===> Prior Generation with KNN Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For each datapoint, get initial class name
    def get_prior(self, best_model):
        # Load Classifier
        self.net = best_model
        self.net.to(self.device)

        # k-nn
        self.net.eval()
        # knn mode on
        neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
        embeddings, class_confi = [], []
        # print(self.noisy_markers.shape)
        knn_embeds = self.z0[self.noisy_markers==0, :]
        class_confi = self.y_hat[self.noisy_markers==0]

        # class_confi = self.y_hat
        knn_embeds = knn_embeds.cpu().detach().numpy()
        class_confi = class_confi.cpu().detach().numpy()
        neigh.fit(knn_embeds, class_confi)
        print('Time : ', time.time() - self.time)

        # 2. predict class of training dataset
        knn_embeds = self.z0.cpu().detach().numpy()
        class_preds = neigh.predict(knn_embeds)
        class_preds = torch.tensor(np.int64(class_preds))
        print('Prior made {} errors with train/val noisy labels'.format(torch.sum(class_preds!=self.y_hat)))
        print('Prior made {} errors with train/val clean labels'.format(torch.sum(class_preds!=self.y)))
        noisy_preds = torch.tensor([(class_preds[i] != self.y_hat[i]) and (class_preds[i] == self.y[i]) for i in range(len(class_preds))])
        print('Prior detected {} real noisy samples'.format(torch.sum(noisy_preds)))
        # onehot
        # dict = {}
        # model_output = neigh.predict(embeddings)
        # dict['class'] = np.int64(model_output)
        # with open(os.path.join(self.args.cls_dir, 'onehot_'+self.args.data_name), "wb") as f:
        #     pickle.dump(dict, f)
        # f.close()
        # print('Time : ', time.time() - self.time, 'class information saved')

        # # proba
        dict = {}
        model_output = neigh.predict_proba(knn_embeds)
        # print(model_output)
        if model_output.shape[1] < self.n_classes:
            tmp = np.zeros((model_output.shape[0], self.n_classes))
            tmp[:, neigh.classes_] = neigh.predict_proba(embeddings)
            dict['proba'] = tmp
        else:
            dict['proba'] = model_output  # data*n_class

        # with open(os.path.join(self.args.cls_dir, 'proba_'+self.args.data_name), "wb") as f:
        #     pickle.dump(dict, f)s
        # f.close()

        print('Time : ', time.time() - self.time, 'proba information saved')


        return dict['proba'], class_preds.numpy()
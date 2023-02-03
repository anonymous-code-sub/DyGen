import numpy
import torch
import time
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def mahalanobis_dist(args, train_embeds, train_labels):
    start_time = time.time()
    num = 0
    train_input_ids = []
    keys = []
    embeddings = train_embeds[train_labels!=-1].cpu()
    keys = train_labels[train_labels!=-1].cpu()

    l_max = torch.max(train_labels)

    mu = torch.zeros((l_max+1, embeddings.shape[-1]))
    for i in range(l_max+1):
        mu[i] = torch.mean(embeddings[keys==i], 0)
    print('Computed mu!')

    sigma = torch.zeros((l_max+1, embeddings.shape[-1], embeddings.shape[-1]))#.to(embeddings.device)
    for i in range(l_max+1):
        embeds = embeddings[keys==i]
        l = len(embeds)
        for b in range(10):
            bembeds = embeds[int(l/10)*b:int(l/10)*(b+1)]
            center_p = mu[i].unsqueeze(0).repeat((bembeds.shape[0], 1))
            vector = bembeds - center_p
            vector = vector.unsqueeze(-1)
            vector_T = vector.permute(0, 2, 1)
            temp = torch.bmm(vector, vector_T)
            sigma[i] += torch.sum(temp, 0)
    sigma = torch.sum(sigma, 0) / embeddings.shape[0]
    print('Computed sigma!')

    sigma_inv = torch.linalg.inv(sigma)
    dists = torch.zeros((l_max+1, embeddings.shape[0]))#.to(embeddings.device)
    for i in range(l_max+1):
        center_p = mu[i].unsqueeze(0).repeat((embeddings.shape[0], 1))
        vector = embeddings - center_p
        vector = vector.unsqueeze(1)
        vector_T = vector.permute(0,2,1)
        dists[i] = torch.matmul(torch.matmul(vector, sigma_inv), vector_T).squeeze()
    dists = dists.permute(-1,0)
    end_time = time.time()
    print('Computed Distance!')
    print('The relabeling process cost {} seconds.'.format(end_time-start_time))
    return dists

def relative_mahalanobis_dist(args, train_embeds, train_labels):
    start_time = time.time()
    num = 0
    train_input_ids = []
    keys = []
    embeddings = train_embeds[train_labels!=-1].cpu()
    keys = train_labels[train_labels!=-1].cpu()

    l_max = torch.max(train_labels)

    mu = torch.zeros((l_max+1, embeddings.shape[-1]))
    for i in range(l_max+1):
        mu[i] = torch.mean(embeddings[keys==i], 0)
    print('Computed mu!')
    total_mu = torch.mean(embeddings)

    sigma = torch.zeros((l_max+1, embeddings.shape[-1], embeddings.shape[-1]))#.to(embeddings.device)
    total_sigma = torch.zeros(embeddings.shape[-1], embeddings.shape[-1])
    for i in range(l_max+1):
        embeds = embeddings[keys==i]
        l = len(embeds)
        for b in range(10):
            bembeds = embeds[int(l/10)*b:int(l/10)*(b+1)]
            center_p = mu[i].unsqueeze(0).repeat((bembeds.shape[0], 1))
            vector = bembeds - center_p
            vector = vector.unsqueeze(-1)
            vector_T = vector.permute(0, 2, 1)
            temp = torch.bmm(vector, vector_T)
            sigma[i] += torch.sum(temp, 0)
    sigma = torch.sum(sigma, 0) / embeddings.shape[0]
    print('Computed sigma!')
    total_vector = embeds - total_mu.unsqueeze(0).repeat((embeds.shape[0], 1))
    total_vector = total_vector.unsqueeze(-1)
    total_vector_T = total_vector.permute(0,2,1)
    total_sigma = torch.sum(torch.bmm(total_vector, total_vector_T), 0) / embeddings.shape[0]

    sigma_inv = torch.linalg.inv(sigma)
    total_sigma_inv = torch.linalg.inv(total_sigma)
    dists = torch.zeros((l_max+1, embeddings.shape[0]))#.to(embeddings.device)
    for i in range(l_max+1):
        center_p = mu[i].unsqueeze(0).repeat((embeddings.shape[0], 1))
        vector = embeddings - center_p
        vector = vector.unsqueeze(1)
        vector_T = vector.permute(0,2,1)
        total_vector = embeddings - total_mu.unsqueeze(0).repeat((embeddings.shape[0], 1))
        total_vector = total_vector.unsqueeze(1)
        total_vector_T = total_vector.permute(0,2,1)
        dists[i] = torch.matmul(torch.matmul(vector, sigma_inv), vector_T).squeeze() - torch.matmul(torch.matmul(total_vector, total_sigma_inv), total_vector_T).squeeze()
    dists = dists.permute(-1,0)
    end_time = time.time()
    print('Computed Distance!')
    print('The relabeling process cost {} seconds.'.format(end_time-start_time))
    return dists

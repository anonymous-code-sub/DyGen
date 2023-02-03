import numpy
import torch
import time
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def euclidean_dist_wos(args, train_embeds, train_labels, train_labels2=None):
    print(train_labels.shape)
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    l_max = torch.max(train_labels)
    # cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    dists = torch.zeros((l_max+1, train_embeds.shape[0]))
    for i in range(l_max+1):
        cluster_centroids = torch.mean(train_embeds[train_labels==i], 0)
        cluster_centroids = cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0], 1))
        dists[i] = torch.sqrt(torch.sum((train_embeds.to(cluster_centroids.device) - cluster_centroids) ** 2, -1)).squeeze().to(train_embeds.device)
    dists = dists.permute(1, 0)
    print(dists.shape)
    torch.cuda.empty_cache()
    return dists

def euclidean_dist(args, train_embeds, train_labels, train_labels2=None):
    print(train_labels.shape)
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    l_max = torch.max(train_labels)
    cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    for i in range(l_max+1):
        cluster_centroids[i] = torch.mean(train_embeds[train_labels==i], 0)
    embeds1 = train_embeds.unsqueeze(1).repeat((1, cluster_centroids.shape[0], 1))
    embeds2 = cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0], 1, 1))
    dists = torch.sqrt(torch.sum((embeds1.to(embeds2.device) - embeds2) ** 2, -1)).to(embeds1.device)
    print(dists.shape)
    torch.cuda.empty_cache()
    return dists

def norm_euclidean_dist(args, train_embeds, train_labels, train_labels2=None):
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    l_max = torch.max(train_labels)
    cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    for i in range(l_max+1):
        cluster_centroids[i] = torch.mean(train_embeds[train_labels==i], 0)
    embeds1 = train_embeds.unsqueeze(1).repeat((1, cluster_centroids.shape[0], 1))
    embeds2 = cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0], 1, 1))
    dists = torch.sqrt(torch.sum((embeds1.to(embeds2.device) - embeds2) ** 2, -1)).to(embeds1.device)
    dists1 = torch.sqrt(torch.sum(embeds1 ** 2, -1))
    dists2 = torch.sqrt(torch.sum(embeds2 ** 2, -1)).to(embeds1.device)
    dists = dists / (0.5 * (dists + dists1 + dists2))
    return dists

def cosine_dist(args, train_embeds, train_labels):
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    # old_train_embeds = old_train_embeds[train_labels!=-1]
    l_max = torch.max(train_labels)
    cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    for i in range(l_max+1):
        cluster_centroids[i] = torch.mean(train_embeds[train_labels==i], 0)
        # old_cluster_centroids[i] = torch.mean(old_train_embeds[old_train_labels==i], 0)
    
    assigned_centroids = cluster_centroids[train_labels]
    # old_assigned_centroids = old_cluster_centroids[old_train_labels]
    # print(train_embeds.shape, assigned_centroids.shape)
    c2a = train_embeds.cpu() - assigned_centroids.cpu()
    # print(c2a.shape)
    # old_c2a = old_train_embeds - old_assigned_centroids

    centroid_distances = assigned_centroids.unsqueeze(1).repeat((1,l_max+1,1)) - cluster_centroids.unsqueeze(0).repeat((assigned_centroids.shape[0],1,1))
    # print(centroid_distances.shape)
    norm = torch.norm(centroid_distances, dim=-1)
    norm = norm + torch.ones_like(norm) * 1e-7
    # print(norm.shape)
    c2a = c2a.unsqueeze(1).repeat((1,l_max+1,1))
    dists = torch.sum((c2a * centroid_distances), -1) / norm
    return dists
    # old_centroid_distances = old_assigned_centroids.unsqueeze(1).repeat((1,))
    
def relative_dist(args, train_embeds, train_labels):
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    # old_train_embeds = old_train_embeds[train_labels!=-1]
    l_max = torch.max(train_labels)
    cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    for i in range(l_max+1):
        cluster_centroids[i] = torch.mean(train_embeds[train_labels==i], 0)
        # old_cluster_centroids[i] = torch.mean(old_train_embeds[old_train_labels==i], 0)
    
    assigned_centroids = cluster_centroids[train_labels]
    AC = train_embeds.cpu() - assigned_centroids.cpu()

    # AC = torch.sqrt(torch.sum(AC ** 2, -1))
    extended_cluster_centroids = cluster_centroids.unsqueeze(0).repeat((assigned_centroids.shape[0],1,1))
    # for i in range(len(train_labels)):
    #     extended_cluster_centroids[i, train_labels[i]] = 0
    XC = assigned_centroids.unsqueeze(1).repeat((1,l_max+1,1)) - extended_cluster_centroids
    upper = torch.sum(AC.unsqueeze(1).repeat((1,l_max+1,1)) * XC, -1)
    upper[upper==0] = 1
    lower = torch.sqrt(torch.sum(XC ** 2, -1))
    lower[lower==0] = 1
    # XC = torch.sqrt(torch.sum(XC ** 2, -1))
    # AX = train_embeds.cpu().unsqueeze(1).repeat((1,l_max+1,1)) - cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0],1,1))
    # AX = torch.sqrt(torch.sum(AX ** 2, -1))
    # AC = AC.unsqueeze(1).repeat((1, l_max+1))
    dists = upper/lower
    return dists
    # old_centroid_distances = old_assigned_centroids.unsqueeze(1).repeat((1,))

def relative_dist2(args, train_embeds, train_labels):
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    l_max = torch.max(train_labels)
    cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    for i in range(l_max+1):
        cluster_centroids[i] = torch.mean(train_embeds[train_labels==i], 0)
    
    assigned_centroids = cluster_centroids[train_labels]
    AC = train_embeds.cpu() - assigned_centroids.cpu()

    AC = torch.sqrt(torch.sum(AC ** 2, -1))
    extended_cluster_centroids = cluster_centroids.unsqueeze(0).repeat((assigned_centroids.shape[0],1,1))
    XC = assigned_centroids.unsqueeze(1).repeat((1,l_max+1,1)) - extended_cluster_centroids
    XC = torch.sqrt(torch.sum(XC ** 2, -1))
    AX = train_embeds.cpu().unsqueeze(1).repeat((1,l_max+1,1)) - cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0],1,1))
    AX = torch.sqrt(torch.sum(AX ** 2, -1))
    AC = AC.unsqueeze(1).repeat((1, l_max+1))
    dists = AC / (0.5 * (AC + XC + AX))
    return dists
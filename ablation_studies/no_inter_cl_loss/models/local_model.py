import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
import multiprocessing
from functools import partial
from torch.multiprocessing import Process, Manager

from models.lightgcn import Light_GCN, Light_GCN_1
from models.domain_disen import Domain_Disen
from models.attention import Attention
from torch.distributions import Beta


class Local_Model(nn.Module):
    def __init__(self, **params):
        super(Local_Model, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.all_items = set(range(self.item_num))

        self.tau = params['tau']
        self.beta = params['beta']
        self.local_data = params['local_data']
        self.embed_id_dim = params['embed_id_dim']
        self.potential_pos_dict = params['potential_pos_dict']
        # self.u_neg_dict = params['u_neg_dict']
        # self.user_potential_pos_items_dict = {}
        # for user_id in range(self.user_num):
        #     self.user_potential_pos_items_dict[user_id] = set([item for key, items in self.potential_pos_dict.items() if key[0] == user_id for item in items])
        # self.disen_emb_dim = params['disen_emb_dim']
        self.device = params['device']
        self.n_layers = params['n_layers']
        self.train_data = params['train_data']
        self.review_embed_dim = params['review_embed_dim']
        self.u_review_feat = params['u_review_feat']
        self.v_review_feat = params['v_review_feat']
        # self.disen_feat_agg_way = params['disen_feat_agg_way']
        self.u_emb = nn.Embedding(self.user_num, self.embed_id_dim).to(self.device)
        nn.init.xavier_uniform_(self.u_emb.weight)
        self.v_emb = nn.Embedding(self.item_num, self.embed_id_dim).to(self.device)
        nn.init.xavier_uniform_(self.v_emb.weight)
        self.u_review_feat_emb = nn.Embedding(self.user_num, self.review_embed_dim).to(self.device)
        self.u_review_feat_emb.weight.data.copy_(self.u_review_feat)
        # # self.u_review_feat_emb.weight.requires_grad = False
        self.v_review_feat_emb = nn.Embedding(self.item_num, self.review_embed_dim).to(self.device)
        self.v_review_feat_emb.weight.data.copy_(self.v_review_feat)
        # # self.v_review_feat_emb.weight.requires_grad = False

        # light_gcn_params = {
        #     'device': self.device,
        #     # 'u_emb': self.u_emb,
        #     # 'v_emb': self.v_emb,
        #     'user_num': self.user_num,
        #     'item_num': self.item_num,
        #     'train_data': self.train_data,
        #     'n_layers': self.n_layers
        # }
        # self.light_gcn = Light_GCN_1(**light_gcn_params)
        # # self.u_id_embeddings, self.v_id_embeddings = self.light_gcn.u_g_embeddings, self.light_gcn.v_g_embeddings
        # disen_params = {
        #     'orig_emb_size': self.embed_id_dim,
        #     'disen_emb_size': self.disen_emb_dim,
        #     'device': self.device
        # }
        # self.domain_disen_model = Domain_Disen(**disen_params)

        # # #the aggregation layers for id and review features
        # self.v_feat_cat_layer = nn.Linear(self.embed_id_dim*2, self.embed_id_dim).to(self.device)
        # self.v_feat_cat_norm = nn.BatchNorm1d(self.embed_id_dim).to(self.device)
        # self.u_feat_cat_layer = nn.Linear(self.embed_id_dim * 2, self.embed_id_dim).to(self.device)
        # self.u_feat_cat_norm = nn.BatchNorm1d(self.embed_id_dim).to(self.device)

        self.att = Attention(self.embed_id_dim)

        # if self.disen_feat_agg_way == 'concat':
        #     self.disen_agg_layer = nn.Linear(self.disen_emb_dim * 2, self.disen_emb_dim).to(self.device)
        #     self.disen_agg_norm = nn.BatchNorm1d(self.disen_emb_dim).to(self.device)

        # self.v_feat_layer = nn.Linear(self.embed_id_dim, self.disen_emb_dim).to(self.device)
        # self.v_feat_norm = nn.BatchNorm1d(self.disen_emb_dim).to(self.device)

        self.inter_learn_layer1 = nn.Linear(self.embed_id_dim * 2, self.embed_id_dim).to(self.device)
        nn.init.xavier_uniform_(self.inter_learn_layer1.weight)
        self.batch_norm1 = nn.BatchNorm1d(self.embed_id_dim).to(self.device)
        self.inter_learn_layer2 = nn.Linear(self.embed_id_dim, self.embed_id_dim // 2).to(self.device)
        nn.init.xavier_uniform_(self.inter_learn_layer2.weight)
        self.batch_norm2 = nn.BatchNorm1d(self.embed_id_dim // 2).to(self.device)
        self.inter_learn_layer3 = nn.Linear(self.embed_id_dim // 2, self.embed_id_dim // 4).to(self.device)
        nn.init.xavier_uniform_(self.inter_learn_layer3.weight)
        self.batch_norm3 = nn.BatchNorm1d(self.embed_id_dim // 4).to(self.device)
        self.classifier_layer = nn.Linear(self.embed_id_dim // 4, 1).to(self.device)

        # loss function
        self.criterion = nn.BCELoss()
    def get_potential_item_emb_old(self,nodes_u,nodes_v):
        pos_item_emb_list = []
        for user_id,item_id in zip(nodes_u.tolist(),nodes_v.tolist()):
            key = (user_id,item_id)
            # logger.info(key)
            if key in self.potential_pos_dict:
                if len(self.potential_pos_dict[key]) > 0:
                    pos_item_list = self.potential_pos_dict[key]
                    if len(pos_item_list) >= 4:
                        pos_item_list = random.sample(pos_item_list,4)
                else:
                    pos_item_list = [item_id]
                # pos_item = random.sample(pos_item_list,1)
                # pos_item_emb = self.v_emb(torch.tensor(pos_item).to(self.device)).squeeze()
                
                # logger.info(f'pos:{pos_item_list}')
                pos_item_emb = torch.mean(self.v_emb(torch.tensor(pos_item_list).to(self.device)), dim=0)
            else:
                user_inter_items = set(self.local_data.user_inter_dict[user_id])
                non_inter_items = random.sample(list(self.all_items - user_inter_items),4)
                # logger.info(f'neg:{non_inter_items}')
                # user_inter_items = self.local_data.user_inter_dict[user_id]
                # non_inter_items = random.sample(user_inter_items, 4)
                # if len(non_inter_items) > 0:
                #     random_non_inter_item = random.choice(non_inter_items)
                # else:
                #     random_non_inter_item = item_id
                pos_item_emb = torch.mean(self.v_emb(torch.tensor(non_inter_items).to(self.device)),dim=0)
            pos_item_emb_list.append(pos_item_emb)
        return torch.stack(pos_item_emb_list)

    def get_potential_item_emb(self, nodes_u, nodes_v):
        neg_num = 4
        user_ids = nodes_u.to(self.device)
        item_ids = nodes_v.to(self.device)

        # batch_size = len(user_ids)
        # pos_item_emb_list = torch.zeros((batch_size, self.embed_id_dim), device=self.device)
        pos_item_emb_list = self.v_emb(item_ids)

        # Precompute masks and user interacted items
        masks = torch.tensor(
            [(user_id, item_id) in self.potential_pos_dict
             for user_id, item_id in zip(user_ids.tolist(), item_ids.tolist())], dtype=torch.bool, device=self.device)

        # user_interacted_items_dict = {user_id.item(): set(self.local_data.user_inter_dict[user_id.item()]) for user_id
        #                               in user_ids.unique()}

        # Handle cases where potential_pos_dict contains the (user_id, item_id) key
        if masks.any():
            indices = masks.nonzero(as_tuple=True)[0]
            pos_item_lists = [
                random.sample(self.potential_pos_dict[(user_ids[idx].item(), item_ids[idx].item())], neg_num) if len(
                    self.potential_pos_dict[(user_ids[idx].item(), item_ids[idx].item())]) >= neg_num
                else self.potential_pos_dict[(user_ids[idx].item(), item_ids[idx].item())] if len(
                    self.potential_pos_dict[(user_ids[idx].item(), item_ids[idx].item())]) > 0
                else [item_ids[idx].item()] for idx in indices.tolist()]
            pos_item_embs = torch.stack(
                [torch.mean(self.v_emb(torch.tensor(items, device=self.device)), dim=0) for items in pos_item_lists])
            pos_item_emb_list[indices] = pos_item_embs

        # Handle cases where potential_pos_dict does not contain the (user_id, item_id) key
        if (~masks).any():
            indices = (~masks).nonzero(as_tuple=True)[0]
            non_inter_items_list = [
                random.sample(list(self.all_items - set(self.local_data.user_inter_dict[user_ids[idx].item()])),
                              neg_num)
                if len(self.all_items - set(self.local_data.user_inter_dict[user_ids[idx].item()])) >= neg_num
                else list(self.all_items - set(self.local_data.user_inter_dict[user_ids[idx].item()]))
                for idx in indices.tolist()]
            non_inter_item_embs = torch.stack(
                [torch.mean(self.v_emb(torch.tensor(items, device=self.device)), dim=0) for items in
                 non_inter_items_list])
            pos_item_emb_list[indices] = non_inter_item_embs

        return pos_item_emb_list

    def forward(self, nodes_u, nodes_v,global_protos,inter_nums):
        # self.u_id_embeddings,self.v_id_embeddings = self.light_gcn.get_user_item_id_emb(self.u_emb,self.v_emb)
        # u_id_feats = self.u_id_embeddings[nodes_u]
        # v_id_feats = self.v_id_embeddings[nodes_v]
        # self.u_rev_embeddings, self.v_rev_embeddings = self.light_gcn.get_user_item_id_emb(self.u_review_feat_emb,
        #                                                                                    self.v_review_feat_emb)
        # u_review_feats = self.u_rev_embeddings[nodes_u]
        # v_review_feats = self.v_rev_embeddings[nodes_v]
        # batch_inter_nums = [inter_nums[x] for x in nodes_u.tolist()]
        u_id_feats = self.u_emb(nodes_u)
        v_id_feats = self.v_emb(nodes_v)
        u_review_feats = self.u_review_feat_emb(nodes_u)
        v_review_feats = self.v_review_feat_emb(nodes_v)
        potential_item_id_feats = self.get_potential_item_emb(nodes_u,nodes_v)
        # score_1 = torch.sum(v_id_feats*v_id_feats,dim=1)
        # score_2 = torch.sum(v_id_feats*potential_item_id_feats,dim=1)
        # scores = torch.stack([score_1,score_2],dim=1)
        # att_weights = F.softmax(scores,dim=1)
        # v_id_embs = torch.stack([v_id_feats,potential_item_id_feats],dim=1)
        # att_weights = att_weights.unsqueeze(2)
        # v_id_feats = torch.sum(att_weights*v_id_embs,dim=1)

        # # 计算 element-wise product
        # element_wise_product = v_id_feats * potential_item_id_feats
        #
        # # 计算用户的 attention scores (s_u)
        # attention_scores_1 = torch.sum(element_wise_product, dim=1, keepdim=True)
        #
        # # # 计算物品的 attention scores (s_i)
        # # attention_scores_2 = torch.sum(element_wise_product, dim=0, keepdim=True)
        #
        # # 计算用户的 normalized scores (α_u)
        # attention_weights_1 = F.softmax(attention_scores_1, dim=0)
        #
        # # # 计算物品的 normalized scores (α_i)
        # # attention_weights_2 = F.softmax(attention_scores_2, dim=1)
        #
        # # 将 attention weights 扩展以与原始 embedding 的维度匹配
        # attention_weights_expanded_1 = attention_weights_1.expand_as(v_id_feats)
        # # attention_weights_expanded_2 = attention_weights_2.expand_as(potential_item_id_feats)
        #
        # # # 计算融合的 embedding
        # # fused_embedding_1 = torch.sum(attention_weights_expanded_1 * v_id_feats, dim=0)
        # # fused_embedding_2 = torch.sum(attention_weights_expanded_2 * potential_item_id_feats, dim=1)
        #
        # # 最终的融合 embedding
        # v_id_feats = attention_weights_expanded_1*potential_item_id_feats + (1-attention_weights_expanded_1)*v_id_feats
        # # v_id_feats = (v_id_feats+potential_item_id_feats)/2
        
        #beta distribution
        # beta_distribution = Beta(torch.tensor([self.beta]),torch.tensor([self.beta]))
        # delta = beta_distribution.sample().item()
        #uniform distribution
        # delta = torch.rand(len(nodes_u),1,device=self.device)
        #gaussian distribution
        delta = torch.normal(mean=0.5,std=0.1,size=(len(nodes_u),self.embed_id_dim),device=self.device)
        delta = torch.clamp(delta,0,1)
        v_id_feats = delta*v_id_feats+(1-delta)*potential_item_id_feats
        
        
        # proto_feats = torch.tensor([global_protos[label] for label in nodes_u.tolist()])
        # proto_feats = proto_feats.to(torch.float32).to(self.device)
        # att_w = self.att(u_id_feats,proto_feats)
        # weight_user_emb = torch.bmm(att_w.unsqueeze(2), proto_feats.unsqueeze(1))
        # u_feats = weight_user_emb.squeeze(1)
        # u_feats = u_id_feats

        # w_user_id_feats = torch.tensor([x/(x+1) for x in batch_inter_nums]).to(self.device)
        # w_proto_feats = torch.tensor([1/(x+1) for x in batch_inter_nums]).to(self.device)
        # u_feats = (u_id_feats*w_user_id_feats.unsqueeze(1)+proto_feats*w_proto_feats.unsqueeze(1))/2+weight_user_emb.squeeze(1)/2
        inter_feats = torch.cat([u_id_feats, v_id_feats], dim=1)
        inter_feats1 = self.batch_norm1(F.relu(self.inter_learn_layer1(inter_feats)))
        inter_feats2 = self.batch_norm2(F.relu(self.inter_learn_layer2(inter_feats1)))
        inter_feats3 = self.batch_norm3(F.relu(self.inter_learn_layer3(inter_feats2)))
        pred_prob = F.sigmoid(self.classifier_layer(inter_feats3))
        return pred_prob.squeeze(), u_id_feats, v_id_feats,u_review_feats,v_review_feats

    def cross_entropy_loss(self, pred_labels, true_labels):
        L_ce = self.criterion(pred_labels, true_labels.to(torch.float32))
        return L_ce

    def calc_ssl_loss_intra(self, u_id_feats, v_id_feats,u_review_feats, v_review_feats):
        norm_u_id_feats = F.normalize(u_id_feats, p=2, dim=1)
        norm_u_rev_feats = F.normalize(u_review_feats, p=2, dim=1)
        norm_all_u_rev_feats = F.normalize(self.u_review_feat_emb.weight, p=2, dim=1)
        pos_score_user = (norm_u_id_feats * norm_u_rev_feats).sum(dim=1)
        ttl_score_user = torch.matmul(norm_u_id_feats, norm_all_u_rev_feats.T)
        pos_score_user = torch.exp(pos_score_user / self.tau)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.tau), dim=1)
        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        norm_v_id_feats = F.normalize(v_id_feats, p=2, dim=1)
        norm_v_rev_feats = F.normalize(v_review_feats, p=2, dim=1)
        norm_all_v_rev_feats = F.normalize(self.v_review_feat_emb.weight, p=2, dim=1)
        pos_score_item = (norm_v_id_feats * norm_v_rev_feats).sum(dim=1)
        ttl_score_item = torch.matmul(norm_v_id_feats, norm_all_v_rev_feats.T)
        pos_score_item = torch.exp(pos_score_item / self.tau)
        ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.tau), dim=1)
        ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))

        
        return (ssl_loss_user + ssl_loss_item)

    def calc_ssl_loss_intra_item(self, v_id_feats, potential_item_id_feats):
        norm_v_id_feats = F.normalize(v_id_feats, p=2, dim=1)
        norm_potential_item_id_feats = F.normalize(potential_item_id_feats,p=2,dim=1)
        rows, cols = v_id_feats.size()
        random_indices = torch.randperm(rows)
        norm_all_potential_item_id_feats = potential_item_id_feats[random_indices]
        norm_all_potential_item_id_feats = F.normalize(norm_all_potential_item_id_feats, p=2, dim=1)
        pos_score_item = (norm_v_id_feats * norm_potential_item_id_feats).sum(dim=1)
        ttl_score_item = torch.matmul(norm_v_id_feats, norm_all_potential_item_id_feats.T)
        pos_score_item = torch.exp(pos_score_item / self.tau)
        ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.tau), dim=1)
        ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))

        return ssl_loss_item
    def calc_ssl_loss_proto(self, u_id_feats, u_feats):
        norm_u_id_feats = F.normalize(u_id_feats, p=2, dim=1)
        norm_u_proto_feats = F.normalize(u_feats, p=2, dim=1)
        rows,cols = u_id_feats.size()
        random_indices = torch.randperm(rows)
        norm_all_u_proto_feats = u_feats[random_indices]
        norm_all_u_proto_feats = F.normalize(norm_all_u_proto_feats, p=2, dim=1)
        pos_score_user = (norm_u_id_feats * norm_u_proto_feats).sum(dim=1)
        ttl_score_user = torch.matmul(norm_u_id_feats, norm_all_u_proto_feats.T)
        pos_score_user = torch.exp(pos_score_user / self.tau)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.tau), dim=1)
        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        return ssl_loss_user



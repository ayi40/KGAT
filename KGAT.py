import torch.nn as nn
import os
import torch
import numpy as np
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type in ['bi', 'kgat']:
            self.MLP_gc = nn.Linear(self.in_dim, self.out_dim)
            self.MLP_bi = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.MLP_gc.weight)
            nn.init.xavier_uniform_(self.MLP_bi.weight)
        elif self.aggregator_type in ['gcn']:
            self.MLP_gc = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.MLP_gc.weight)
        elif self.aggregator_type in ['graphsage']:
            self.MLP_graphsage = nn.Linear(self.in_dim * 2, self.out_dim)
            nn.init.xavier_uniform_(self.MLP_graphsage.weight)
        else:
            print('please check the the aggregator_type argument, which should be bi, kgat, gcn, or graphsage.')
            raise NotImplementedError

    def forward(self, ego_embed, A_in):
        neighbor_embed = torch.matmul(A_in, ego_embed)

        if self.aggregator_type in ['bi', 'kgat']:
            # add part
            add_embed = ego_embed + neighbor_embed
            add_embed = self.activation(self.MLP_gc(add_embed))
            # elementwise part
            wise_embed = ego_embed * neighbor_embed
            wise_embed = self.activation(self.MLP_bi(wise_embed))
            # total embed
            embed = add_embed + wise_embed

        elif self.aggregator_type in ['gcn']:
            # add part
            embed = ego_embed + neighbor_embed
            embed = self.activation(self.MLP_gc(embed))
        elif self.aggregator_type in ['graphsage']:
            embed = torch.cat([ego_embed, neighbor_embed], dim=1)
            embed = self.activation(self.MLP_graphsage(embed))
        else:
            print('please check the the aggregator_type argument, which should be bi, kgat, gcn, or graphsage.')
            raise NotImplementedError
        embed = self.message_dropout(embed)
        return embed


class KGAT(nn.Module):
    def __init__(self, config, args):
        super(KGAT, self).__init__()
        self._init_args(config, args)
        self._init_embedding()

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.weight_size_list[k], self.weight_size_list[k + 1],
                                                     self.mess_dropout[k], self.aggre_type))

    def _init_args(self, config, args):
        self.n_users = config['n_users']
        self.n_entities = config['n_entities']
        self.n_relations = config['n_relations']
        self.emb_dim = args.embed_size
        self.kge_dim = args.kge_size
        self.relation_dim = args.relation_size
        self.weight_size = args.layer_size
        self.weight_size_list = [self.emb_dim] + eval(self.weight_size)
        self.n_layers = len(eval(self.weight_size))
        self.aggre_type = args.aggregator_type
        self.mess_dropout = eval(args.mess_dropout)
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.A_in = config['A_in']

    def _init_embedding(self):
        self.all_embedding = nn.Embedding(self.n_users + self.n_entities, self.emb_dim)
        nn.init.xavier_uniform_(self.all_embedding.weight)
        self.r_embedding = nn.Embedding(self.n_relations, self.kge_dim)
        nn.init.xavier_uniform_(self.r_embedding.weight)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.emb_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.trans_M)

    def _cal_all_embedding(self):
        ego_embed = self.all_embedding.weight
        all_embedding = [ego_embed]
        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embedding.append(norm_embed)

        all_embedding = torch.cat(all_embedding, dim=1)
        return all_embedding

    def _cal_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        all_embed = self._cal_all_embedding()
        user_embed = all_embed[user_ids]
        item_pos_embed = all_embed[item_pos_ids + self.n_users]
        item_neg_embed = all_embed[item_neg_ids + self.n_users]

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)

        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def _cal_kg_loss(self, h, r, pos_t, neg_t):
        r_embed = self.r_embedding(r)
        W_r = self.trans_M[r]


        h_embed = self.all_embedding(h)
        pos_t_embed = self.all_embedding(pos_t)
        neg_t_embed = self.all_embedding(neg_t)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)

        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.r_embedding.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.all_embedding.weight[h_list]
        t_embed = self.all_embedding.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list

    def _update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]
            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def _calcscore(self):
        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[user_ids]
        item_embed = all_embed[item_ids + self.n_users]

        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))
        return cf_score

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self._cal_cf_loss(*input)
        if mode == 'train_kg':
            return self._cal_kg_loss(*input)
        if mode == 'update_att':
            return self._update_attention(*input)
        if mode == 'predict':
            return self._calcscore(*input)

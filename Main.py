from utility.parser import parse_args
from utility.LoadData import LoadData
import os
import torch
import random
from time import time

from utility.Dataset import RecommendDataset, KGEDataset
from torch.utils.data import DataLoader
from utils.log_helper import *
from utils.metrics import *
from KGAT import KGAT
import logging


if __name__ == '__main__':
    random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)

    args = parse_args()

    # Setting log
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load basic data from data_generator
    data_loader = LoadData(args)
    config = dict()
    config['n_users'] = data_loader.n_users
    config['n_items'] = data_loader.n_items
    config['n_relations'] = data_loader.n_relations
    config['n_entities'] = data_loader.n_entities
    # Load Laplacian adjacency matrix
    # include all relation in one matrix
    config['A_in'] = data_loader.A_in.to(device)
    # Load CKG triplets
    config['all_h_list'] = data_loader.all_h_list
    config['all_r_list'] = data_loader.all_r_list
    config['all_t_list'] = data_loader.all_t_list
    config['all_v_list'] = data_loader.all_v_list


    # Initialize Data Generator
    train_rec_db = RecommendDataset(data_loader.train_data, data_loader.train_user_dict, data_loader.n_items, 'train')
    train_rec_generator = DataLoader(dataset=train_rec_db, batch_size=args.batch_size, shuffle=True)
    test_rec_db = RecommendDataset(data_loader.test_data, data_loader.test_user_dict, data_loader.n_items, 'test')
    test_rec_generator = DataLoader(dataset=test_rec_db, batch_size=args.batch_size, shuffle=True)
    kge_db = KGEDataset(data_loader.all_h_list, data_loader.all_r_list, data_loader.all_t_list,
                              data_loader.all_kg_dict, data_loader.n_user, data_loader.n_entities)
    kge_generator = DataLoader(dataset=kge_db, batch_size=args.kge_batch_size, shuffle=True)



    # Initialize Model
    model = KGAT(config, args)
    model.to(device)

    # Initialize Training Setting
    optimizercf = torch.optim.Adam(model.parameters(), args.lr)
    optimizerkg = torch.optim.Adam(model.parameters(), args.lr)

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)
    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    # Training
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = len(train_rec_generator)
        for index, data in enumerate(train_rec_generator):
            time2 = time()
            user, pos_i, neg_i = data
            user = user.to(device)
            pos_i = pos_i.to(device)
            neg_i = neg_i.to(device)

            cf_loss = model(user, pos_i, neg_i, mode='train_cf')
            cf_loss.backward()
            optimizercf.step()
            optimizercf.zero_grad()
            cf_total_loss+=cf_loss
            if (index % 500) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, index, n_cf_batch, time() - time2, cf_loss.item(), cf_total_loss / index))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
            epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = len(kge_generator)
        for index, data in enumerate(kge_generator):
            time4 = time()
            h, r, pos_t, neg_t = data
            h = h.to(device)
            r = r.to(device)
            pos_t = pos_t.to(device)
            neg_t = neg_t.to(device)

            kg_loss = model(h, r, pos_t, neg_t, mode='train_kg')
            kg_loss.backward()
            optimizerkg.step()
            optimizerkg.zero_grad()
            kg_total_loss += kg_loss
            if (index % 300) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, index, n_kg_batch, time() - time4, kg_loss.item(), kg_total_loss / index))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                        epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

        # update attention
        time5 = time()
        h_list = torch.LongTensor(config['all_h_list']).to(device)
        r_list = torch.LongTensor(config['all_r_list']).to(device)
        t_list = torch.LongTensor(config['all_t_list']).to(device)
        relations = list(data_loader.relation_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))










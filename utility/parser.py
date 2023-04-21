import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs='?', default='./Data/yelp2018')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--kge_batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--embed_size', type=int, default=64, help='CF Embedding size.')
    parser.add_argument('--kge_size', type=int, default=64, help='KG Embedding size.')
    parser.add_argument('--relation_size', type=int, default=64, help='r Embedding size.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every propagation layer')
    parser.add_argument('--aggregator_type', nargs='?', default='kgat',
                        help='Specify the type of the aggregator method from {bi, gcn, graphsage}.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')

    args = parser.parse_args()

    save_dir = './result/yelp2018/'
    args.save_dir = save_dir

    return args
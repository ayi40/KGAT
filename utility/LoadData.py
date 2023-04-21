import collections
import numpy as np
import scipy.sparse as sp
import torch


class LoadData(object):
    def __init__(self, args):
        self.args = args

        train_file = self.args.data_path + '/train.txt'
        test_file = self.args.data_path + '/test.txt'
        kg_file = self.args.data_path + '/kg_final.txt'

        # load user-item graph data and get basic num
        self.train_data, self.train_user_dict = self._load_uigraph(train_file)
        self.test_data, self.test_user_dict = self._load_uigraph(test_file)
        self.exist_users = self.train_user_dict.keys()

        self.n_train, self.n_test = 0, 0
        self.n_user, self.n_items = 0, 0
        self._statistic_uigraph()

        # load kg data
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)

        # print basic info
        self._print_basic_info()

        # generate the adjacency
        self.adj_list, self.adj_r_list = self._get_adj_matrix()

        # generate the matrices after laplacian normalization
        self.lap_list = self._get_relation_lap_list()

        # generate the triples dic, key is 'head', value is (tail, relation)
        self.all_kg_dict = self._get_all_kg_dict()
        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()

        self.A_in = sum(self.lap_list).tocoo()
        self.A_in = self.convert_coo2tensor(self.A_in)

    def _load_uigraph(self, file_name):
        data = list()
        user_dict = dict()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            l = l.strip()
            l = [int(i) for i in l.split(' ')]
            u_id, v_id = l[0], l[1:]
            v_id = list(set(v_id))

            for v in v_id:
                data.append([u_id, v])
            if len(v_id)>0:
                user_dict[u_id] = v_id
        return np.array(data), user_dict

    def _statistic_uigraph(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    def _load_kg(self, file_name):
        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]),max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict = collections.defaultdict(list)
        relation_dict = collections.defaultdict(list)

        for head, relation, tail in kg_np:
            kg_dict[head].append((tail, relation))
            relation_dict[relation].append((head, tail))
        return kg_np, kg_dict, relation_dict

    def _print_basic_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))

    def _get_adj_matrix(self):
        adj_list = []
        adj_r_list = []

        # np2adj
        def _np2adj(npdata, row_pre, col_pre):
            n_all = self.n_users + self.n_entities

            a_rows = npdata[:, 0] + row_pre
            a_cols = npdata[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))
            return a_adj, b_adj

        R, R_inv = _np2adj(self.train_data, row_pre=0, col_pre=self.n_users)
        adj_list.append(R)
        adj_r_list.append(0)
        adj_list.append(R_inv)
        adj_r_list.append(self.n_relations + 1)

        for r_id in self.relation_dict.keys():
            K, K_inv = _np2adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            adj_list.append(K)
            adj_r_list.append(r_id+1)

            adj_list.append(K_inv)
            adj_r_list.append(r_id+2+self.n_relations)
        self.n_relations = len(adj_r_list)

        return adj_list, adj_r_list

    def _get_relation_lap_list(self):
        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            # 构建对角矩阵
            d_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_inv = sp.diags(d_inv)

            norm_adj = d_inv.dot(adj)
            return norm_adj.tocoo()

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):

            rows = lap.row
            cols = lap.col

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
        return all_kg_dict

    def _get_all_kg_data(self):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in enumerate(self.lap_list):
            all_h_list += list(lap.row)
            all_t_list += list(lap.col)
            all_v_list += list(lap.data)
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row)

        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        # resort the all_h/t/r/v_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        print('\treordering indices...')
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])
        print('\treorganize all kg data done.')

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        # try:
        #     assert sum(new_v_list) == sum(all_v_list)
        # except Exception:
        #     print(sum(new_v_list), '\n')
        #     print(sum(all_v_list), '\n')
        print('\tsort all data done.')


        return new_h_list, new_r_list, new_t_list, new_v_list

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))



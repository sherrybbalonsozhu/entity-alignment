#coding:utf-8

import numpy as np
import scipy.spatial as spatial


def get_hits(src_embedding,tgt_embedding, src_test_ents,tgt_test_ents, top_k=(1, 10, 50, 100, 200)):
        Lvec = np.array([src_embedding[e1[0]] for e1 in src_test_ents])
        Rvec = np.array([tgt_embedding[e2[0]] for e2 in tgt_test_ents])
        sim =spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
        top_lr = [0] * len(top_k)
        for i in range(Lvec.shape[0]):
                rank = sim[i, :].argsort()
                rank_index = np.where(rank == i)[0][0]
                for j in range(len(top_k)):
                        if rank_index < top_k[j]:
                                top_lr[j] += 1
        top_rl = [0] * len(top_k)
        for i in range(Rvec.shape[0]):
                rank = sim[:, i].argsort()
                rank_index = np.where(rank == i)[0][0]
                for j in range(len(top_k)):
                        if rank_index < top_k[j]:
                                top_rl[j] += 1
        print('For each left:')
        for i in range(len(top_lr)):
                print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(src_test_ents) * 100))
        print('For each right:')
        for i in range(len(top_rl)):
                print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(src_test_ents) * 100))

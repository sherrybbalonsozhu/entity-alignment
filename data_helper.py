#coding:utf-8
from data_reader import read_aligned_entities,read_name2ids,read_triples
from data_reader import    convert_names_to_ids,convert_triple_to_ids,read_entity2labels
import random
import numpy as np
import time
import os

class DataHelper(object):
    def __init__(self,params):
        #数据路径
        self.src_triple_path=params.src_triple_path #source triple
        self.tgt_triple_path=params.tgt_triple_path #target triple
        self.src_attr_path=params.src_attr_path #source attribute
        self.tgt_attr_path=params.tgt_attr_path #target attribute
        self.src_entity2id_path=params.src_entity2id_path
        self.src_relation2id_path=params.src_relation2id_path
        self.tgt_entity2id_path=params.tgt_entity2id_path
        self.tgt_relation2id_path=params.tgt_relation2id_path
        self.aligned_ent_path = params.aligned_ent_path  # pre-aligned entities
        self.aligned_train_ent_path = params.aligned_train_ent_path  # pre-aligned entities
        self.aligned_dev_ent_path = params.aligned_dev_ent_path  # pre-aligned entities
        self.aligned_test_ent_path = params.aligned_test_ent_path  # pre-aligned entities
        self.aligned_rel_path=params.aligned_rel_path # pre-aligned relations
        self.new_aligned_path=params.new_aligned_path #新加入的对齐实体
        #labels
        self.src_label_path=params.src_label_path
        self.tgt_label_path=params.tgt_label_path
        self.src_ent2labels=read_entity2labels(self.src_label_path)
        self.tgt_ent2labels=read_entity2labels(self.tgt_label_path)
        #params
        self.sep=params.sep or "\t"
        self.iter=params.iter or False
        #读取数据
        self.load_and_process_data()

        #超参数
        #负样本个数
        self.neg_num=params.neg_num or 20


    def load_and_process_data(self):
        #entity2ids
        self.src_entity2ids,self.src_id2entities=read_name2ids(self.src_entity2id_path)
        self.tgt_entity2ids,self.tgt_id2entities=read_name2ids(self.tgt_entity2id_path)
        #relation2ids
        self.src_relation2ids,self.src_id2relations=read_name2ids(self.src_relation2id_path)
        self.tgt_relation2ids, self.tgt_id2relations=read_name2ids(self.tgt_relation2id_path)
        #triples
        src_triples_names=read_triples(self.src_triple_path,sep=self.sep)
        self.src_triples=convert_triple_to_ids(src_triples_names,self.src_entity2ids,self.src_relation2ids)
        tgt_triples_names=read_triples(self.tgt_triple_path,sep=self.sep)
        self.tgt_triples=convert_triple_to_ids(tgt_triples_names,self.tgt_entity2ids,self.tgt_relation2ids)
        print("source triple num: %s, target triple num: %s"%(len(src_triples_names),len(tgt_triples_names)))
        print("src ent num:%s, tgt ent num:%s"%(len(self.src_id2entities),len(self.tgt_id2entities)))
        print("src relation num:%s, tgt relation num:%s"%(len(self.src_id2relations),len(self.tgt_id2relations)))
        #attributes

        #aligned entities
        # self.src_entities_names,self.tgt_entities_names=read_aligned_entities(self.aligned_ent_path)
        # self.src_train_entities_names,self.tgt_train_entities_names=read_aligned_entities(self.aligned_ent_path)
        # src_entities=np.array(convert_names_to_ids(self.src_entities_names,self.src_entity2ids))
        # tgt_entities=np.array(convert_names_to_ids(self.tgt_entities_names,self.tgt_entity2ids))
        # self.data_num=len(src_entities)
        # src_entities = np.reshape(src_entities, newshape=[self.data_num,1])
        # tgt_entities=np.reshape(tgt_entities,newshape=[self.data_num,1])

        self.src_train_entities_names,self.tgt_train_entities_names=read_aligned_entities(self.aligned_train_ent_path,sep=self.sep)
        self.src_dev_entities_names,self.tgt_dev_entities_names=read_aligned_entities(self.aligned_dev_ent_path,sep=self.sep)
        self.src_test_entities_names,self.tgt_test_entities_names=read_aligned_entities(self.aligned_test_ent_path,sep=self.sep)
        src_train_entities=np.array(convert_names_to_ids(self.src_train_entities_names,self.src_entity2ids))
        src_dev_entities=np.array(convert_names_to_ids(self.src_dev_entities_names,self.src_entity2ids))
        tgt_train_entities=np.array(convert_names_to_ids(self.tgt_train_entities_names,self.tgt_entity2ids))
        tgt_dev_entities=np.array(convert_names_to_ids(self.tgt_dev_entities_names,self.tgt_entity2ids))
        self.train_src_entities=np.reshape(src_train_entities,[len(src_train_entities),1])
        self.train_tgt_entities=np.reshape(tgt_train_entities,[len(tgt_train_entities),1])
        self.dev_src_entities=np.reshape(src_dev_entities,[len(src_dev_entities),1])
        self.dev_tgt_entities=np.reshape(tgt_dev_entities,[len(tgt_dev_entities),1])
        self.src_entities=np.concatenate([self.train_src_entities,self.dev_src_entities],axis=0)
        self.tgt_entities=np.concatenate([self.train_tgt_entities,self.dev_tgt_entities],axis=0)
        # new aligned
        if os.path.exists(self.new_aligned_path) and self.iter:
            self.src_new_entities_names, self.tgt_new_entities_names = read_aligned_entities(self.new_aligned_path,sep="\t")
            src_new_entities = np.array(convert_names_to_ids(self.src_new_entities_names, self.src_entity2ids))
            tgt_new_entities = np.array(convert_names_to_ids(self.tgt_new_entities_names, self.tgt_entity2ids))
            src_new_entities =np.reshape(src_new_entities,newshape=[len(src_new_entities),1])
            tgt_new_entities =np.reshape(tgt_new_entities,newshape=[len(tgt_new_entities),1])
            self.src_entities=np.concatenate([self.src_entities,src_new_entities],axis=0)
            self.tgt_entities=np.concatenate([self.tgt_entities,tgt_new_entities],axis=0)

        #test data
        src_test_entities=np.array(convert_names_to_ids(self.src_test_entities_names,self.src_entity2ids))
        tgt_test_entities=np.array(convert_names_to_ids(self.tgt_test_entities_names,self.tgt_entity2ids))
        self.test_src_entities=np.reshape(src_test_entities,[len(src_test_entities),1])
        self.test_tgt_entities=np.reshape(tgt_test_entities,[len(tgt_test_entities),1])

        print("train data：%s，test data:%s"%(len(self.src_entities),len(self.test_tgt_entities)))

        #entity vocab size
        self.src_entity_num=len(self.src_entity2ids)
        self.tgt_entity_num=len(self.tgt_entity2ids)
        self.src_rel_num=len(self.src_relation2ids)
        self.tgt_rel_num=len(self.tgt_relation2ids)

        #aligned relations
        # src_rel_names,tgt_rel_names=read_aligned_entities(self.aligned_rel_path)
        # self.src_rel=convert_names_to_ids(src_rel_names,self.src_relation2ids)
        # self.tgt_rel=convert_names_to_ids(tgt_rel_names,self.tgt_relation2ids)


    def train_batch_generator(self,batch_size=128,shuffle=False,cur_epoch=0,is_training=True):
        batch_datas={}
        batch_datas["src_triples"]=self.src_triples
        batch_datas["tgt_triples"]=self.tgt_triples
        # batch_datas["src_rel"]=self.src_rel
        # batch_datas["tgt_rel"]=self.tgt_rel
        # neg_src_rel=np.random.choice(self.src_rel_num,size=[len(self.src_rel),self.neg_num])
        # neg_tgt_rel=np.random.choice(self.tgt_rel_num,size=[len(self.tgt_rel),self.neg_num])
        # batch_datas["neg_src_rel"]=neg_src_rel
        # batch_datas["neg_tgt_rel"]=neg_tgt_rel

        data_num=len(self.src_entities)
        batch_num=(data_num+batch_size-1)//batch_size
        src_entities = self.src_entities
        tgt_entities = self.tgt_entities
        if shuffle:
            ids=random.sample(list(range(data_num)),data_num)
            src_entities=self.src_entities[ids]
            tgt_entities=self.tgt_entities[ids]
        neg_src=np.random.choice(self.src_entity_num,size=[data_num,self.neg_num])
        neg_tgt=np.random.choice(self.tgt_entity_num,size=[data_num,self.neg_num])

        batch_datas["all_src_ent"]=src_entities
        batch_datas["all_tgt_ent"]=tgt_entities

        for i in range(batch_num):
            s=i*batch_size
            e=(i+1)*batch_size
            batch_src_ent=src_entities[s:e]
            batch_tgt_ent=tgt_entities[s:e]
            batch_datas["src_ent"]=batch_src_ent
            batch_datas["tgt_ent"]=batch_tgt_ent
            start_time=time.time()
            #负样本
            size=len(batch_src_ent)
            if cur_epoch%10==0:
                neg_src=np.random.choice(self.src_entity_num,size=[data_num,self.neg_num])
                neg_tgt=np.random.choice(self.tgt_entity_num,size=[data_num,self.neg_num])
            batch_datas["neg_src_ent"]=neg_src[s:e]
            batch_datas["neg_tgt_ent"]=neg_tgt[s:e]
            if is_training:
                # ent_mask=np.random.choice([1,0] *data_num, size=[data_num,1])
                ent_mask=np.ones([data_num,1])
                ent_mask[s:e,:]=0
                batch_datas["ent_mask"]=ent_mask
            else:
                batch_datas["ent_mask"]=np.ones([data_num,1])
            end_time=time.time()
            # print("batch time:%s"%(end_time-start_time))
            yield batch_datas


if __name__=="__main__":
    from params import Params
    param=Params(data_name="ja_en")
    print(param)

    helper=DataHelper(param)

    #print(helper.tgt_entity2ids)
    #print(helper.tgt_triples)

    gen=helper.train_batch_generator(batch_size=128)
    for batch_data in gen:
        print(batch_data)

    print("src ent num: %s, tgt ent num: %s"%(helper.src_entity_num,helper.tgt_entity_num))
    print("src rel num: %s, tgt rel num: %s"%(helper.src_rel_num,helper.tgt_rel_num))






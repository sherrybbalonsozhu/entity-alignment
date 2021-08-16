#coding:utf-8
from data_helper import DataHelper
from models.model_base import BaseModel,GCNModel
from params import Params
from trainer import BaseTrainer
import tensorflow as tf
from data_reader import read_embedding

params={
    #data
    "src_triple_num":105998,
    "tgt_triple num": 115722}

zh_en={
    #data
    "src_triple_num":70414,
    "tgt_triple_num": 95142,
    "src_ent_num": 19388,
    "tgt_ent_num": 19572,
    "src_rel_num": 1701,
    "tgt_rel_num": 1323,
    #model
    "num_layer":2,
    "emb_dim":300,
    "rel_dim":20,
    "emb_norm": True,
    "output_norm": True,  # True
    #train
    "optimizer":tf.train.AdamOptimizer,#tf.train.AdamOptimizer,#tf.train.GradientDescentOptimizer,
    "lr":0.002
   }


fr_en={
    #data
    "src_triple_num":105998,
    "tgt_triple num": 115722,
    "src_ent_num": 19661,
    "tgt_ent_num": 19993,
    "src_rel_num": 903,
    "tgt_rel_num": 1208,
    #model
    "num_layer":2,
    "emb_dim":300,
    "rel_dim":20,
    "emb_norm":True,
    "output_norm":True, #True
    #train
    "optimizer":tf.train.AdamOptimizer,#tf.train.AdamOptimizer,#tf.train.GradientDescentOptimizer,
    "lr":0.002
   }

ja_en={
    #data
    "src_triple_num":77214,
    "tgt_triple num": 93484,
    "src_ent_num": 19814,
    "tgt_ent_num": 19780,
    "src_rel_num": 1299,
    "tgt_rel_num": 1153,
    #model
    "num_layer":2,
    "emb_dim":300,
    "rel_dim":20,
    "emb_norm":False,
    "output_norm":True, #True
    #train
    "optimizer":tf.train.AdamOptimizer,#tf.train.AdamOptimizer,#tf.train.GradientDescentOptimizer,
    "lr":0.002
   }



if __name__=="__main__":
    #超参数
    # params=Params(data_name="zh_en")
    # params.update(zh_en)

    params=Params(data_name="fr_en")
    params.update(fr_en)

    #embedding
    src_ent_embeding=read_embedding(params.src_ent_emb_path,vocab_size=params.src_ent_num,emb_dim=300)
    tgt_ent_embeding=read_embedding(params.tgt_ent_emb_path,vocab_size=params.tgt_ent_num,emb_dim=300)


    #data
    data_helper=DataHelper(params)

    #model
    # model=BaseModel(params)
    model=GCNModel(params,src_ent_embedding=src_ent_embeding,tgt_ent_embedding=tgt_ent_embeding)

    #trainer
    trainer=BaseTrainer(model,params)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    #train
    # trainer.restore_last_session(sess)
    trainer.train(sess,data_helper,iter_num=5000)

    #test
    trainer.evaluate(sess,data_helper)

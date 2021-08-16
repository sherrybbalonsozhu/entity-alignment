#coding:utf-8
from data_helper import DataHelper
from models.model_base import GATModel
from params import Params
from trainer import BaseTrainer
import tensorflow as tf
from data_reader import read_embedding

config={
    #data
    "src_triple_num":105998,
    "tgt_triple num": 115722,
    #model
    "num_layer":2,
    #train
    "lr":50,
   }
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
    #train
    "optimizer":tf.train.AdamOptimizer,#tf.train.AdamOptimizer,#tf.train.GradientDescentOptimizer,
    "lr":0.002
   }


if __name__=="__main__":
    #超参数
    params=Params(data_name="zh_en")
    params.update(zh_en)
    #data
    data_helper=DataHelper(params)
    #embedding
    src_ent_embeding=read_embedding(params.src_ent_emb_path,vocab_size=params.src_ent_num,emb_dim=300)
    tgt_ent_embeding=read_embedding(params.tgt_ent_emb_path,vocab_size=params.tgt_ent_num,emb_dim=300)

    #model
    # model=GATModel(params)
    model=GATModel(params,src_ent_embedding=src_ent_embeding,tgt_ent_embedding=tgt_ent_embeding)

    #trainer
    trainer=BaseTrainer(model,params)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    #train
    # trainer.restore_last_session(sess)
    trainer.train(sess,data_helper,iter_num=5000)

    #test
    trainer.evaluate(sess,data_helper)

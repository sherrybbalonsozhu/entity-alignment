#coding:utf-8
from data_helper import DataHelper
from models.models_crossKG import CrossGAT
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
    "emb_dim":200,
    "rel_dim":50,
    #train
    "optimizer":tf.train.GradientDescentOptimizer, #tf.train.AdamOptimizer,
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
    "emb_norm": False, #False
    "output_norm": False, #True
    #train
    "optimizer":tf.train.AdamOptimizer,#tf.train.AdamOptimizer,#tf.train.GradientDescentOptimizer,
    "lr":0.002,
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
    "emb_norm":False, #False
    "output_norm":False, #True
    #train
    "optimizer":tf.train.AdamOptimizer,#tf.train.AdamOptimizer,#tf.train.GradientDescentOptimizer,
    "lr":0.002,
   }

ja_en={
    #data
    "src_triple_num":77214,
    "tgt_triple_num": 93484,
    "src_ent_num": 19814,
    "tgt_ent_num": 19780,
    "src_rel_num": 1299,
    "tgt_rel_num": 1153,
    #model
    "num_layer":2,
    "emb_dim":300,
    "rel_dim":20,
    "emb_norm":False, #False
    "output_norm":False, #True
    #train
    "optimizer":tf.train.AdamOptimizer,#tf.train.AdamOptimizer,#tf.train.GradientDescentOptimizer,
    "lr":0.002,
   }
def train(data_name="0.1/fr_en",data_param=fr_en):
    #超参数
    params=Params(data_name=data_name)
    params.update(data_param)

   #data
    data_helper=DataHelper(params)
    #embedding
    src_ent_embeding=read_embedding(params.src_ent_emb_path,vocab_size=params.src_ent_num,emb_dim=300)
    tgt_ent_embeding=read_embedding(params.tgt_ent_emb_path,vocab_size=params.tgt_ent_num,emb_dim=300)

    #model
    # model=CrossGAT(params)
    model=CrossGAT(params,src_ent_embedding=src_ent_embeding,tgt_ent_embedding=tgt_ent_embeding)

    #trainer
    trainer=BaseTrainer(model,params)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    #train
    trainer.restore_last_session(sess)
    # trainer.train(sess,data_helper,iter_num=5000)

    #test
    trainer.evaluate(sess,data_helper)
    sess.close()

if __name__=="__main__":
    data_params={"zh_en":zh_en,"ja_en":ja_en,"fr_en":fr_en}
    train(data_name="zh_en",data_param=data_params["zh_en"])
    # train(data_name="ja_en",data_param=data_params["ja_en"])
    # train(data_name="fr_en",data_param=data_params["fr_en"])





#coding:utf-8
import tensorflow as tf
from models.model_base import BaseModel

class CrossGAT(BaseModel):
    def __init__(self,params,src_ent_embedding=None,src_rel_embedding=None,
                 tgt_ent_embedding=None,tgt_rel_embedding=None):
        super(CrossGAT,self).__init__(params,src_ent_embedding,src_rel_embedding,
                                      tgt_ent_embedding,tgt_rel_embedding)
    def build_model(self,features):
        src_triples=features["src_triples"]
        tgt_triples=features["tgt_triples"]
        src_ent=features["src_ent"]
        tgt_ent=features["tgt_ent"]
        neg_src_ent=features["neg_src_ent"]
        neg_tgt_ent=features["neg_tgt_ent"]
        ent_mask=features["ent_mask"]

        src_heads,src_relatioins,src_tails=src_triples[:,0],src_triples[:,1],src_triples[:,2]
        tgt_heads,tgt_relations,tgt_tails=tgt_triples[:,0],tgt_triples[:,1],tgt_triples[:,2]
        #gat
        src_ent_embedding,src_rel_embedding=self.get_init_embeddings(
            self.init_src_entity_embedding,self.init_src_relation_embedding,self.src_ent_num,self.src_rel_num,name="src")
        tgt_ent_embedding,tgt_rel_embedding=self.get_init_embeddings(
            self.init_tgt_entity_embedding,self.init_tgt_relation_embedding,self.tgt_ent_num,self.tgt_rel_num,name="tgt")
        with tf.variable_scope("gat"):
            src_entity_embedding,tgt_entity_embedding=self.cross_gat(
                src_heads,src_tails,src_ent_embedding,
                tgt_heads,tgt_tails,tgt_ent_embedding,
                src_aligned=src_ent,tgt_aligned=tgt_ent,ent_mask=ent_mask)

        self.src_entity_embedding=src_entity_embedding
        self.tgt_entity_embedding=tgt_entity_embedding

        # 计算对齐实体之间的距离
        self.loss=self.get_loss(src_entity_embedding,tgt_entity_embedding,
                                src_ent,tgt_ent,neg_src_ent,neg_tgt_ent,ent_mask)

        self.train_op=self.get_train_op(self.loss,learning_rate=self.lr)
        # self.build_summary_op()

    def get_loss(self,src_entity_embedding,tgt_entity_embedding,
                 src_ent,tgt_ent,neg_src_ent,neg_tgt_ent,ent_mask):
        src_ent_vecs=tf.nn.embedding_lookup(src_entity_embedding,src_ent)
        tgt_ent_vecs=tf.nn.embedding_lookup(tgt_entity_embedding,tgt_ent)
        neg_src_ent_vecs=tf.nn.embedding_lookup(src_entity_embedding,neg_src_ent)
        neg_tgt_ent_vecs=tf.nn.embedding_lookup(tgt_entity_embedding,neg_tgt_ent)

        pos_dist=tf.reduce_sum(tf.abs(src_ent_vecs-tgt_ent_vecs),axis=2,keepdims=False)
        neg_dist_tgt=tf.reduce_sum(tf.abs(src_ent_vecs-neg_tgt_ent_vecs),axis=2)
        neg_dist_src=tf.reduce_sum(tf.abs(tgt_ent_vecs-neg_src_ent_vecs),axis=2)
        #计算损失
        loss1=tf.maximum(0.,pos_dist-neg_dist_src+self.margin)*(1-ent_mask)
        loss2=tf.maximum(0.,pos_dist-neg_dist_tgt+self.margin)*(1-ent_mask)
        loss=(tf.reduce_sum(loss1)+tf.reduce_sum(loss2))/(2*tf.reduce_sum(1-ent_mask)*self.neg_num)
        return loss

    def cross_gat(self,src_heads,src_tails,src_ent_embedding,
                    tgt_heads,tgt_tails,tgt_ent_embedding,
                    src_aligned,tgt_aligned,ent_mask):
        src_ent_emb=src_ent_embedding
        tgt_ent_emb=tgt_ent_embedding
        for i in range(self.num_layer):
            with tf.variable_scope("layer_%s"%i):
                #cross GAT
                src_ent_emb,tgt_ent_emb=self.sparseCrossGATLayer(src_aligned,tgt_aligned,src_ent_emb,tgt_ent_emb,
                                             ent_mask,in_dim=self.emb_dim)

                #GAT
                with tf.variable_scope("gat"):
                    src_ent_emb=self.sparseGATLayer(src_heads,src_tails,in_dim=self.emb_dim,
                                           entity_embeddings=src_ent_emb,triple_num=self.src_triple_num)
                with tf.variable_scope("gat",reuse=True):
                    tgt_ent_emb=self.sparseGATLayer(tgt_heads,tgt_tails,in_dim=self.emb_dim,
                                            entity_embeddings=tgt_ent_emb,triple_num=self.tgt_triple_num)
        # src_ent_emb=tf.nn.l2_normalize(src_ent_emb,axis=1) #不加
        # tgt_ent_emb=tf.nn.l2_normalize(tgt_ent_emb,axis=1)
        # entity_emb=tf.layers.batch_normalization(entity_emb,axis=-1,training=self.is_training)
        src_ent_emb=tf.nn.dropout(src_ent_emb,keep_prob=self.keep_prob)
        tgt_ent_emb=tf.nn.dropout(tgt_ent_emb,keep_prob=self.keep_prob)
        return src_ent_emb,tgt_ent_emb

    def sparseCrossGATLayer_bak(self,src_ent,tgt_ent,src_ent_embedding,tgt_ent_embedding,ent_mask,in_dim):
        # src = entity_embeddings.get_shape()[0]
        src_ent_emb=tf.squeeze(tf.nn.embedding_lookup(src_ent_embedding,src_ent),axis=1)
        tgt_ent_emb=tf.squeeze(tf.nn.embedding_lookup(tgt_ent_embedding,tgt_ent),axis=1)
        indices_src2tgt = tf.concat([src_ent, tgt_ent], axis=-1)
        indices_tgt2src = tf.concat([tgt_ent, src_ent], axis=-1)
        with tf.variable_scope("src2tgt"):
            #attention score
            #score Sparse Matrix
            indices=tf.cast(indices_src2tgt,tf.int64)
            #scores=tf.nn.leaky_relu(tf.reduce_sum(head_emb*tail_emb,axis=1)) #dot score
            #FFN score
            W=tf.get_variable(name="weight",shape=[in_dim*2,1],initializer=tf.initializers.he_uniform())
            b=tf.get_variable(name="bias",shape=[1])
            scores=tf.nn.leaky_relu(tf.matmul(tf.concat([src_ent_emb,tgt_ent_emb],axis=1),W)+b)
            scores=scores-(1-ent_mask)*100000
            scores=tf.squeeze(scores,axis=1)

            scores=tf.exp(scores)
            scores_matrix=tf.SparseTensor(indices=indices,values=scores,dense_shape=[self.src_ent_num,self.tgt_ent_num])
            scores_sum=tf.sparse_reduce_sum(scores_matrix,axis=1,keepdims=True)
            #triple sparse Matrix
            outputs_src=tf.sparse_tensor_dense_matmul(scores_matrix,tgt_ent_embedding)/(scores_sum+1e-8)
            outputs_src=outputs_src+src_ent_embedding
            outputs_src=tf.nn.relu(outputs_src)

        with tf.variable_scope("tgt2src"):
            #attention score
            #score Sparse Matrix
            indices=tf.cast(indices_tgt2src,tf.int64)
            #scores=tf.nn.leaky_relu(tf.reduce_sum(head_emb*tail_emb,axis=1)) #dot score
            #FFN score
            W=tf.get_variable(name="weight",shape=[in_dim*2,1],initializer=tf.initializers.he_uniform())
            b=tf.get_variable(name="bias",shape=[1])
            scores=tf.nn.leaky_relu(tf.matmul(tf.concat([tgt_ent_emb,src_ent_emb],axis=1),W)+b)
            scores=scores-(1-ent_mask)*100000
            scores=tf.squeeze(scores,axis=1)

            scores=tf.exp(scores)
            scores_matrix=tf.SparseTensor(indices=indices,values=scores,dense_shape=[self.tgt_ent_num,self.src_ent_num])
            scores_sum=tf.sparse_reduce_sum(scores_matrix,axis=1,keepdims=True)
            #triple sparse Matrix
            outputs_tgt=tf.sparse_tensor_dense_matmul(scores_matrix,src_ent_embedding)/(scores_sum+1e-8)
            outputs_tgt=outputs_tgt+tgt_ent_embedding
            outputs_tgt=tf.nn.relu(outputs_tgt)

            return outputs_src,outputs_tgt

    def sparseCrossGATLayer_add(self,src_ent,tgt_ent,src_ent_embedding,tgt_ent_embedding,ent_mask,in_dim):
        # src = entity_embeddings.get_shape()[0]
        src_ent_emb=tf.squeeze(tf.nn.embedding_lookup(src_ent_embedding,src_ent),axis=1)
        tgt_ent_emb=tf.squeeze(tf.nn.embedding_lookup(tgt_ent_embedding,tgt_ent),axis=1)
        #mask
        src_ent_emb=src_ent_emb*ent_mask
        tgt_ent_emb=tgt_ent_emb*ent_mask
        #crossKG information
        outputs_src=tf.scatter_nd(indices=src_ent,updates=tgt_ent_emb,shape=[self.src_ent_num,in_dim])
        outputs_tgt=tf.scatter_nd(indices=tgt_ent,updates=src_ent_emb,shape=[self.tgt_ent_num,in_dim])
        #combine crossKG information
        outputs_src=tf.nn.relu(outputs_src+src_ent_embedding)
        outputs_tgt=tf.nn.relu(outputs_tgt+tgt_ent_embedding)
        return outputs_src,outputs_tgt

    def sparseCrossGATLayer(self,src_ent,tgt_ent,src_ent_embedding,tgt_ent_embedding,ent_mask,in_dim):
        # src = entity_embeddings.get_shape()[0]
        src_ent_emb=tf.squeeze(tf.nn.embedding_lookup(src_ent_embedding,src_ent),axis=1)
        tgt_ent_emb=tf.squeeze(tf.nn.embedding_lookup(tgt_ent_embedding,tgt_ent),axis=1)
        #mask
        src_ent_emb=src_ent_emb*ent_mask
        tgt_ent_emb=tgt_ent_emb*ent_mask
        #crossKG information
        outputs_src=tf.scatter_nd(indices=src_ent,updates=tgt_ent_emb,shape=[self.src_ent_num,in_dim])
        outputs_tgt=tf.scatter_nd(indices=tgt_ent,updates=src_ent_emb,shape=[self.tgt_ent_num,in_dim])
        #combine crossKG information
        with tf.variable_scope("src_cross_inf"):
            W=tf.get_variable(name="weight",shape=[in_dim*2,in_dim],initializer=tf.initializers.he_uniform())
            b=tf.get_variable(name="bias",shape=[in_dim])
            gate=tf.nn.sigmoid(tf.matmul(tf.concat([src_ent_embedding,outputs_src],axis=1),W)+b)
            outputs_src=src_ent_embedding*gate+outputs_src*(1-gate)
            outputs_src=tf.nn.relu(outputs_src)
        with tf.variable_scope("tgt_cross_inf"):
            W = tf.get_variable(name="weight", shape=[in_dim * 2, in_dim], initializer=tf.initializers.he_uniform())
            b = tf.get_variable(name="bias", shape=[in_dim])
            gate = tf.nn.sigmoid(tf.matmul(tf.concat([tgt_ent_embedding, outputs_tgt], axis=1), W) + b)
            outputs_tgt=tgt_ent_embedding*gate+outputs_tgt*(1-gate)
            outputs_tgt=tf.nn.relu(outputs_tgt)
        outputs_src = tf.nn.l2_normalize(outputs_src, axis=1)
        outputs_tgt = tf.nn.l2_normalize(outputs_tgt, axis=1)
        return outputs_src,outputs_tgt

    def sparseGATLayer(self,heads,tails,in_dim, entity_embeddings,triple_num):
        n_e=entity_embeddings.get_shape()[0]
        #embedding
        head_emb=tf.nn.embedding_lookup(entity_embeddings,heads)
        tail_emb=tf.nn.embedding_lookup(entity_embeddings,tails)
        # triple_num=tf.shape(heads,out_type=tf.int32)[0]
        #每个节点的邻居三元组位置[node_id,triple_id]
        indices_in = tf.concat([tf.expand_dims(tails, 1), tf.expand_dims(tf.range(0, triple_num), 1)], axis=-1)
        indices_out = tf.concat([tf.expand_dims(heads, 1), tf.expand_dims(tf.range(0, triple_num), 1)], axis=-1)
        with tf.variable_scope("gat_dense"):
            W = tf.get_variable(name="weights", shape=[1, in_dim], initializer=tf.initializers.ones)
            neighbors_in = tf.multiply(head_emb, W)
            neighbors_out = tf.multiply(tail_emb, W)
        with tf.variable_scope("in_direction_gat"):
            #attention score
            #score Sparse Matrix
            indices=tf.cast(indices_in,tf.int64)
            #scores=tf.nn.leaky_relu(tf.reduce_sum(head_emb*tail_emb,axis=1)) #dot score
            #FFN score
            W=tf.get_variable(name="weight",shape=[in_dim*2,1],initializer=tf.initializers.he_uniform())
            b=tf.get_variable(name="bias",shape=[1])
            scores=tf.nn.leaky_relu(tf.matmul(tf.concat([tail_emb,head_emb],axis=1),W)+b)
            scores=tf.squeeze(scores,axis=1)

            scores=tf.exp(scores)
            scores_matrix=tf.SparseTensor(indices=indices,values=scores,dense_shape=[n_e,triple_num])
            scores_sum=tf.sparse_reduce_sum(scores_matrix,axis=1,keepdims=True)
            #triple sparse Matrix
            outputs_in=tf.sparse_tensor_dense_matmul(scores_matrix,neighbors_in)#/(scores_sum+1e-8)
            outputs_in=(outputs_in+entity_embeddings)/(scores_sum+1.0)
        with tf.variable_scope("out_direction_gcn"):
            #score Sparse Matrix
            indices=tf.cast(indices_out,tf.int64)
            # scores=tf.exp(tf.nn.leaky_relu(tf.reduce_sum(head_emb*tail_emb,axis=1)))#dot scores
            #FFN scores
            W=tf.get_variable(name="weight",shape=[in_dim*2,1],initializer=tf.initializers.he_uniform())
            b=tf.get_variable(name="bias",shape=[1])
            scores=tf.nn.leaky_relu(tf.matmul(tf.concat([head_emb,tail_emb],axis=1),W)+b)
            scores=tf.squeeze(scores,axis=1)

            scores=tf.exp(scores)
            scores_matrix=tf.SparseTensor(indices=indices,values=scores,dense_shape=[n_e,triple_num])
            scores_sum=tf.sparse_reduce_sum(scores_matrix,axis=1,keepdims=True)
            #triple sparse Matrix
            outputs_out=tf.sparse_tensor_dense_matmul(scores_matrix,neighbors_out)#/(scores_sum+1e-9)
            outputs_out=(outputs_out+entity_embeddings)/(scores_sum+1.0)
        outputs=(outputs_in+outputs_out)/2.0

        outputs=tf.nn.relu(outputs)
        # outputs=tf.nn.dropout(outputs,keep_prob=self.keep_prob)
        # outputs=tf.nn.l2_normalize(outputs)
        # outputs=tf.layers.batch_normalization(outputs,axis=-1,training=self.is_training)
        return outputs


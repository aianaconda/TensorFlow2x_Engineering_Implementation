# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""
import tensorflow as tf


def cost_matrix(x,y):
    x_col = tf.expand_dims(x,-2)
    y_lin = tf.expand_dims(y,-3)
    c = tf.reduce_sum((tf.abs(x_col-y_lin))**2,axis=-1)
    return c


def sinkhorn_loss(x,y,epsilon,niter,reduction = 'mean'):
    '''
    Parameters
    ----------
    x : 输入A
    y : 输入B
    epsilon :缩放参数
    n : A/B中元素个数
    niter : 迭代次数
    Return:
        返回结果和sinkhorn距离
    '''
    def M(C,u,v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + tf.expand_dims(u,-1) + tf.expand_dims(v,-2)) / epsilon
    def lse(A):
        return tf.reduce_logsumexp(A,axis=1,keepdims=True)


    x_points = tf.shape(x)[-2]
    y_points = tf.shape(y)[-2]

    
    dim = len( tf.shape(x) )
    if dim == 2:
        batch = 1
    else:
        batch = tf.shape(x)[0]
        
    mu = tf.ones([batch,x_points] , dtype = tf.float32)*(1.0/tf.cast(x_points,tf.float32))
    nu = tf.ones([batch,y_points] , dtype = tf.float32)*(1.0/tf.cast(y_points,tf.float32))
    mu = tf.squeeze(mu)
    nu = tf.squeeze(nu)
    
    
    u, v = 0. * mu, 0. * nu
    C = cost_matrix(x, y)  # Wasserstein cost function    
    for i in range(niter):
        
        u1 = u  #保存上一步U值
        
        u = epsilon * (tf.math.log(mu+1e-8) - tf.squeeze(lse(M(C,u, v)) )  ) + u
        v = epsilon * (tf.math.log(nu+1e-8) - tf.squeeze( lse(tf.transpose(M(C,u, v))) ) ) + v
        
        err = tf.reduce_mean( tf.reduce_sum( tf.abs(u - u1),-1) )
        # print("err",err)
        if err.numpy() < 1e-1: #如果U值没有在更新，则结束
            break

    u_final,v_final = u,v
    pi = tf.exp(M(u_final,v_final))
    if reduction == 'mean':
        cost = tf.reduce_mean(pi*C) 
    elif reduction == 'sum':
        cost = tf.reduce_sum(pi*C)

    return cost,pi, C


if __name__ == '__main__':
    A = [[0.,0.],[1.,0.],[2.,0.],[3.,0.]]
    B = [[0.,1.],[1.,1.],[2.,1.],[3.,1.]]
    
    A = tf.convert_to_tensor(A)
    B = tf.convert_to_tensor(B)
    
    costmatrix = cost_matrix(A,B)
    print(costmatrix)
    '''
    array([[ 1,  2,  5, 10],
           [ 2,  1,  2,  5],
           [ 5,  2,  1,  2],
           [10,  5,  2,  1]])>
    '''
    # result, cost ,C= sinkhorn_loss(A,B,epsilon = 1,n = 4, niter = 5) 
    cost , result, C= sinkhorn_loss(A,B,epsilon = 1, niter = 5) 
    print(result)
    print(cost)
    '''
    tf.Tensor(
    [[1.8792072e-01 5.9113003e-02 2.9430646e-03 2.3191265e-05]
     [5.9113003e-02 1.3739800e-01 5.0545905e-02 2.9430634e-03]
     [2.9430634e-03 5.0545905e-02 1.3739800e-01 5.9113003e-02]
     [2.3191265e-05 2.9430646e-03 5.9113003e-02 1.8792072e-01]], shape=(4, 4), dtype=float32)
    '''
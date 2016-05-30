import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_class=W.shape[1]
  num_feature=X.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    #W*Xi C*1
    x=np.exp(np.dot(W.T,X[i,:]))
    denominator=np.sum(x)
    numerator=x[y[i]]
    loss-=np.log(numerator/denominator)
    #numerator and denominator
    #for j in range(num_class):
    normalize_score=x/denominator
    nm=np.reshape(normalize_score, (num_class, 1))
    
    #CxD
    dscore=nm.dot(np.reshape(X[i,:],(1,num_feature)))
    #print(dscore.shape)

    dscore[y[i],:]-=X[i,:]
    dW+=dscore.T

  loss/=num_train
  dW = dW/num_train + reg*W
  #
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_class=W.shape[1]
  num_feature=X.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #W.T x X =>NxC
  X_scores=np.exp(np.dot(X,W))
  #score sum N
  sum_scores=np.sum(X_scores,axis=1)

  #CxN
  propotion=(X_scores.T/sum_scores)

  loss=np.sum(np.log(propotion[y,range(num_train)]))*-1

  propotion[y,range(num_train)]-=1
  #C*D
  dW+=propotion.dot(X).T

  loss/=num_train
  dW = dW/num_train + reg*W
  #
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


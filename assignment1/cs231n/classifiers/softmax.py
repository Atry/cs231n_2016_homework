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

  #############################################################################
  # DONE: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_trains = X.shape[0]
  num_pixels = X.shape[1]
  num_classes = W.shape[1]
  for i in range(num_trains):
    correct_label = y[i]
    scores = X[i].dot(W)
    count = 0
    for j in range(num_classes):
      count += np.exp(scores[j])
    loss += -np.log(np.exp(scores[correct_label]) / count)
    dW[:, correct_label] += -X[i]
    for j in range(num_classes):
      dW[:, j] += X[i] / count * np.exp(scores[j])
        
  loss /= num_trains
  dW /= num_trains
  loss += 0.5 * reg * np.sum(W * W)      
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  #############################################################################
  # DONE: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_trains = X.shape[0]
  num_pixels = X.shape[1]
  num_classes = W.shape[1]
  scores = X.dot(W)
  total_exp = np.sum(np.exp(scores), 1)
  loss = -np.sum(np.log(np.exp(scores[range(num_trains),y]) / total_exp)) / num_trains + 0.5 * reg * np.sum(W * W)
  output_indices = (y.reshape(num_trains, 1) + (np.arange(num_pixels) * num_classes).reshape(1, num_pixels)).reshape(num_trains * num_pixels)
  numerator_dW = np.bincount(output_indices, -X.reshape(num_trains * num_pixels)).reshape(num_pixels, num_classes)
  denominator_dW = X.T.dot(np.exp(scores) / total_exp.reshape(num_trains, 1))
  dW = (numerator_dW + denominator_dW) / num_trains + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


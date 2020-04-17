import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from numpy import ndarray
import numpy as np

def autoencoder_1hid(n_input, n_hidden):
    '''A simple 3 layer AutoEncoder setup'''
    # The model
    n_output = n_input

    # Sequential setup
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(input_shape=(n_input,), units=n_hidden, activation=None, name='encoder'),
      tf.keras.layers.Dense(units=n_output, activation='softmax', name='decoder')
    ])
    
    return model


def train(model, X, Y, n_epochs=1000, learning_rate=2.0):
    '''Train with gradient descent and plot the losses.'''
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    losses = []
    for i in range(n_epochs):
        # train all batch together
        with tf.GradientTape() as tape:
            Y_pred = model(X.toarray())        # y_pred is a tensor
            loss = tf.reduce_mean(tf.square(Y_pred - Y.toarray()))
        grads = tape.gradient(loss, model.variables)    # get the gradients
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        losses.append(loss)

        if (i+1)%100 == 0:
            print("Epoch:{}, loss = {:.5f}".format(i+1, loss))

    plt.plot(losses)
    print('\nPlotting the loss value:')
    

def plot_enc_2d(data:'2d array', label:'list', offset=(0,0), size=[]):
    '''Scatter plot'''
    # check input shape
    if data.shape[1] == 2: pass
    elif data.shape[0] == 2: data = data.transpose()
    else: 
      print('Data dimension error:', data.shape)
      return -1
    
    z1 = data[:,0]   # coordinate on x axis
    z2 = data[:,1]   # coordinate on y axis
    if size:
      plt.scatter(z1, z2, s=size, edgecolors='black', alpha=0.5)
    else:
      plt.scatter(z1, z2)

    # plot annotation
    for i, txt in enumerate(label):
      if size:
        scale = max(size) * 20

        plt.annotate(txt, (z1[i]+size[i]/scale+offset[0], z2[i]+offset[1]), fontsize=14)
      else:
        plt.annotate(txt, (z1[i]+offset[0], z2[i]+offset[1]), fontsize=14)

    plt.grid()


class nHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder multiple labels into a n-hot vector.
    Need a pre-fit OneHotEncoder as init input.
    
    The number of labels m should be the same. If not, pad "none" prior to
    calling the transform method.
    
    transform e.g.
    ---- word group ----         ------- word list -------
    [['apple', 'orange'],        [['apple'],     [['orange'], 
     ['apple', 'eat'],      ==>   ['apple'], and  ['eat'],    
     ['orange',  'juice']         ['milk']]       ['juice']]
     
                                 ---- one-hot vector -----
                                 [[1,0,0,0],     [[0,1,0,0],
                     then   ==>   [1,0,0,0], and  [0,0,1,0]
                                  [0,1,0,0]]      [0,0,0,1]]
                                 
                                 ------ n-hot vector ------
                                 [[1,1,0,0],
                    finally ==>   [1,0,1,0],
                                  [0,1,0,1]]   
    """
    def __init__(self, ohot):
        '''
        So far only 1d OneHotEncoder is tested.
        '''
        self.ohot = ohot        
        
    def fit(self, X, y=None):
        # no fit needed
        return self
    
    def transform(self, X:'list') -> 'sparse matrix': 
        '''
        Convert an n-hot vector back to word.
        '''
        assert type(X) is list
        
        # break down X into m lists of words
        word_list = [[] for _ in X[0]]
        for group in X:
            for list_, word in zip(word_list, group):
                list_.append([word])
        
        # convert each word list into one-hot
        list_1hot = [self.ohot.transform(list_) for list_ in word_list]
        
        # add them up
        nhot_vec = csr_matrix(list_1hot[0].shape, dtype='float32')
        for vec in list_1hot:
            nhot_vec += vec
            
        return nhot_vec
    
    def inverse_transform(self, X:'array', n_value:'int'=1) -> 'list' : 
        '''
        Convert an n-hot vector back to word.
        n_value: number of words to convert back, only applies when input X
                 is a np.array rather than sparse matrix.
        '''
        topN_indices = []
        if type(X) is csr_matrix:
            # the 1's in each vector
            [topN_indices.append(vec.indices) for vec in X]
        elif type(X) is ndarray:
            topN_indices = find_topN_idx_2d(X, n_value)
        else:
            raise TypeError('Input type not recognized.', type(X))
            
        words_list = []
        # row by row
        for vec, indices in zip(X, topN_indices):
            words = []
            for idx in indices:  
                words.append(self.ohot.categories_[0][idx])
            words_list.append(words)
        
        return words_list

    
def find_topN_idx_2d(arr:'2d-array', N:'int', return_value=False) -> '2d-list':
    '''
    find the indices of top N values in a 2d array (list).
    scipy sparse matrix is not supported.
    '''
    assert len(arr[0]) > 0
    return [find_topN_idx(a, N, return_value) for a in arr]


def find_topN_idx(arr:'list/array', N:'int', return_value=False) -> 'list':
    '''
    find the indices of top N values in a 1d array (list).
    scipy sparse matrix is not supported.
    '''
    N = int(N)
    assert N > 0 
    
    # stack
    sk = [(-1, -float('inf')) for _ in range(N)]
    
    for idx, val in enumerate(arr):
        # push new value
        if val > sk[-1][1]:
            sk.pop()
            sk.append((idx, val))
            
            # re-order
            n = N-1
            while n>0:
                if sk[n][1] > sk[n-1][1]:
                    sk[n-1], sk[n] = sk[n], sk[n-1]  # swap
                else:
                    break
                n = n-1
        # value too small
        else:
            pass
    
    if return_value: return sk
    else: return [tpl[0] for tpl in sk]
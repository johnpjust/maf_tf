import train
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mafs
import struct
import numpy as np
import glob
import scipy.io

# ## read data FMNIST.....
fnames_data = [r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/train-images-idx3-ubyte', r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/t10k-images-idx3-ubyte',r'C:\Users\justjo\Downloads\public_datasets/MNIST/train-images.idx3-ubyte', r'C:\Users\justjo\Downloads\public_datasets/MNIST/t10k-images.idx3-ubyte']

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

data = []
for f in fnames_data:
    data.append(read_idx(f))
val = data[1].reshape((data[1].shape[0], -1))/128. - 1.
test = np.concatenate(data[2:])
test = test.reshape((test.shape[0],-1))/128. - 1.
data = data[0].reshape((data[0].shape[0],-1))/128. - 1.

## Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][2]).
# # n = 2000
# x2 = np.random.randn(2000).astype(dtype=np.float32) * 2.
# x1 = np.random.randn(2000).astype(dtype=np.float32) + (x2 * x2 / 4.)
# data = np.stack([x1, x2], axis=-1)

## iris dataset
# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data

# ## cifar10
# fnames_cifar = glob.glob(r'C:\Users\justjo\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\train\*')
# data=[np.load(f, allow_pickle=True, encoding='latin1') for f in fnames_cifar]
# data = np.concatenate([a['data'] for a in data])/128. - 1.
# f = r'C:\Users\justjo\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\test\test_batch'
# val = np.load(f, allow_pickle=True, encoding='latin1')['data']/128. - 1.
#
# ## svhn
# svhn = scipy.io.loadmat(r'C:\Users\justjo\Downloads\public_datasets\SVHN.mat')
# test = np.moveaxis(svhn['X'],3,0)
# test = np.reshape(test, (test.shape[0],-1))/128. - 1.

# ## svd
# _, _, vh = scipy.linalg.svd(data, full_matrices=False)
# data = np.matmul(data, vh.T)
# val = np.matmul(val, vh.T)
# test = np.matmul(test, vh.T)

## build model
model = mafs.MaskedAutoregressiveFlow(data.shape[1], [100], tf.nn.tanh, 5)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

## train
t = train.Trainer(model)
init=tf.global_variables_initializer()
sess.run(init)
# t.train(sess, data[:60000,:], data[60000:,:], early_stopping=10, check_every_N=5, show_log=True, batch_size=100)
# t.train(sess, data[:1000,:], data[1000:,:], early_stopping=5, check_every_N=5, show_log=True, batch_size=1000)
t.train(sess, data.astype(np.float32), val.astype(np.float32), early_stopping=20, check_every_N=5, show_log=True, batch_size=100, max_iterations=20000, test_data=test.astype(np.float32))

import matplotlib.pyplot as plt
import scipy.stats
out = model.eval(data, sess)
out2 = model.eval(val, sess)
out3 = model.eval(test, sess)
dist = scipy.stats.johnsonsu.fit(out)
out = (np.arcsinh((out - dist[-2]) / dist[-1]) * dist[1] + dist[0])
out2 = (np.arcsinh((out2 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
out3 = (np.arcsinh((out3 - dist[-2]) / dist[-1]) * dist[1] + dist[0])

plt.figure()
plt.hist(out, 50, density=True, alpha=0.5, label='cifar_train');
plt.hist(out2, 50, density=True, alpha=0.5, label='cifar_val')
plt.hist(out3, 50, density=True, alpha=0.5, label='svhn')

saver = tf.train.Saver()
saver.save(sess, r'C:\Users\justjo\PycharmProjects\maf_tf\Models\maf_cifarVSsvhn_h100_f5_svdall_tanh_dim3072\model')

s = model.gen(sess, 1000)
plt.scatter(data[:,0], data[:,1])
plt.scatter(s[:,0], s[:,1])
import train
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mafs
import struct
import numpy as np
import glob
import scipy.io
from skimage import color
import matplotlib
# ## read data FMNIST.....
fnames_data = [r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/train-images-idx3-ubyte', r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/t10k-images-idx3-ubyte',r'C:\Users\justjo\Downloads\public_datasets/MNIST/train-images.idx3-ubyte', r'C:\Users\justjo\Downloads\public_datasets/MNIST/t10k-images.idx3-ubyte']

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

data_ = []
for f in fnames_data:
    data_.append(read_idx(f))
val = data_[1].reshape((data_[1].shape[0], -1))/128. - 1.
test = np.concatenate(data_[2:])
test = test.reshape((test.shape[0],-1))/128. - 1.
data = data_[0].reshape((data_[0].shape[0],-1))/128. - 1.

## Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][2]).
# n = 5000
# x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
# x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
# data_ = np.stack([x1, x2], axis=-1)
# val = data_[:1000, :]
# test = data_[:50,:]
# data = data_[1000:,:]

### iris dataset
# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data

# ## cifar10
# fnames_cifar = glob.glob(r'C:\Users\justjo\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\train\*')
# data=[np.load(f, allow_pickle=True, encoding='latin1') for f in fnames_cifar]
# data = np.concatenate([a['data'] for a in data])/128. - 1.
# # data = np.concatenate([a['data'].reshape((10000,3,32,32)) for a in data])
# # data = np.transpose(data, (0, 2, 3, 1))
# # data = data/255.
# # data = matplotlib.colors.rgb_to_hsv(data)
# # data = data.reshape((-1,3072)) - 0.5
#
# f = r'C:\Users\justjo\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\test\test_batch'
# val = np.load(f, allow_pickle=True, encoding='latin1')['data']/128. - 1.
# # val = np.load(f, allow_pickle=True, encoding='latin1')['data']/255.
# # val = val.reshape((10000,3,32,32))
# # val = np.transpose(val, (0, 2, 3, 1))
# # val = matplotlib.colors.rgb_to_hsv(val)
# # val = val.reshape((-1,3072)) - 0.5
#
# #
# ## svhn
# svhn = scipy.io.loadmat(r'C:\Users\justjo\Downloads\public_datasets\SVHN.mat')
# test = np.moveaxis(svhn['X'],3,0)
# test = np.reshape(test, (test.shape[0],-1))/128. - 1.
# # test = np.moveaxis(svhn['X'],3,0)/255.
# # test = matplotlib.colors.rgb_to_hsv(test)
# # test = np.reshape(test, (test.shape[0],-1))*2. - 1.


#### svd
# _, _, vh = scipy.linalg.svd(data, full_matrices=False)
# data = np.matmul(data, vh.T)
# val = np.matmul(val, vh.T)
# test = np.matmul(test, vh.T)

## build model
num_layers = 10
num_hidden=[1024]*2
act = tf.nn.tanh
model = mafs.MaskedAutoregressiveFlow(data.shape[1], num_hidden, act, num_layers, batch_norm=True)
# model_contrastive = mafs.MaskedAutoregressiveFlow(data.shape[1], num_hidden, act, num_layers, batch_norm=False, SCE=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

SCE=False
## train typical
if not SCE:
    t = train.Trainer(model) ## only pass model but don't re-initialize for SCE
# else:
#     t = train.Trainer(model, SCE=True, model_contrastive=model_contrastive) ## only pass model but don't re-initialize for SCE

## optimizer has some kind of parameters that require initialization
init=tf.global_variables_initializer()
sess.run(init)

train_idx = np.arange(data.shape[0])
# N=1000
# np.random.shuffle(train_idx)

# t.train(sess, data[:60000,:], data[60000:,:], early_stopping=10, check_every_N=5, show_log=True, batch_size=100)
# t.train(sess, data[:1000,:], data[1000:,:], early_stopping=5, check_every_N=5, show_log=True, batch_size=1000)

## MLE training
t.train(sess, data.astype(np.float32), val_data=val.astype(np.float32), early_stopping=100, check_every_N=5, show_log=True, batch_size=100, max_iterations=20000, test_data=test.astype(np.float32), saver_name='temp/tmp_model')
# t.train(sess, data.astype(np.float32), val_data=val.astype(np.float32), early_stopping=100, check_every_N=5, show_log=True, batch_size=100, max_iterations=20000, saver_name='temp/tmp_model')

# ## update contrastive parameters; SCE --> deep copy of model params
# for n in range(50):
#     if SCE:
#         for m, n in zip(model_contrastive.mades, model.mades):
#             m.input_order = n.input_order
#             m.Mmp = n.Mmp
#             m.Ms = n.Ms
#         model_parms = sess.run(model.parms)
#         for m, n in zip(model_contrastive.parms, model_parms):
#             sess.run(tf.assign(m, n))
#
#     ## SCE training --> after training wth MLE
#     if SCE:
#         N=5000
#         np.random.shuffle(train_idx)
#         s = model.gen(sess, N)
#         # t.train_SCE(sess, data[train_idx[:N]].astype(np.float32), contrastive_data=s, val_data=val.astype(np.float32), early_stopping=100, check_every_N=5, show_log=True, batch_size=100, max_iterations=20000, test_data=test.astype(np.float32), saver_name='temp/tmp_model')
#         t.train_SCE(sess, data.astype(np.float32), contrastive_data=s, val_data=val.astype(np.float32), early_stopping=100, check_every_N=5, show_log=True, batch_size=100, max_iterations=20000, test_data=test.astype(np.float32), saver_name='temp/tmp_model')
#
# ###

import matplotlib.pyplot as plt
import scipy.stats
s = model.gen(sess, 5000)
out = model.eval(data, sess)
out2 = model.eval(val, sess)
out3 = model.eval(test, sess)
sout_ = model.eval(s,sess)
dist = scipy.stats.johnsonsu.fit(out)
out = (np.arcsinh((out - dist[-2]) / dist[-1]) * dist[1] + dist[0])
out2 = (np.arcsinh((out2 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
out3 = (np.arcsinh((out3 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
sout = (np.arcsinh((sout_ - dist[-2]) / dist[-1]) * dist[1] + dist[0])

plt.figure()
plt.hist(out, 50, density=True, alpha=0.3, label='cifar_train')
plt.hist(out2, 50, density=True, alpha=0.3, label='cifar_val')
plt.hist(out3, 50, density=True, alpha=0.3, label='svhn')
plt.hist(sout, 50, density=True, alpha=0.3, label='samples')
plt.xlabel('MAF Density')
plt.legend()
plt.xlim([-6,3])
plt.savefig(r'C:\Users\justjo\Desktop\maf_cifarVSsvhn_density_samples_'
            r'.png', bbox_inches='tight')

saver = tf.train.Saver()
saver.save(sess, r'C:\Users\justjo\PycharmProjects\maf_tf\Models\maf_digitsVSfashion_h100_f5_tanh_dim784\model')

# s = model.gen(sess, 5000)
plt.figure();plt.scatter(data[:,0], data[:,1], alpha=1, label='data')
plt.scatter(test[:,0], test[:,1], alpha=0.2)
plt.scatter(s[:,0], s[:,1], alpha=0.3, label='sampled')
st = np.matmul(s, vh)
plt.scatter(st[:,0], st[:,1], alpha=0.2, label='MAF_SVD')
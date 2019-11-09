import train
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mafs
import struct
import numpy as np
import glob
import scipy.io
import gzip
# from skimage import color
# import matplotlib
# ## read data FMNIST.....

# def read_idx(filename):
#     with gzip.open(filename, 'rb') as f:
#         # for line in fin:
#         #     print('got line', line)
#     # with open(filename, 'rb') as f:
#         zero, data_type, dims = struct.unpack('>HBB', f.read(4))
#         shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
#         return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
#
#
# train_data = read_idx(r'D:\publicDatasets\FMNIST\train-images-idx3-ubyte.gz')
# train_data = train_data.reshape((train_data.shape[0], -1)) / 128. - 1.
# train_idx = np.arange(train_data.shape[0])
# np.random.shuffle(train_idx)
# val = train_data[-int(0.2 * train_data.shape[0]):]
# train_data = train_data[:-int(0.2 * train_data.shape[0])]
# test = read_idx(r'D:\publicDatasets\FMNIST\t10k-images-idx3-ubyte.gz')
# test = test.reshape((test.shape[0], -1)) / 128. - 1.
#
# fnames_data = [r'D:\publicDatasets\MNIST\train-images-idx3-ubyte.gz',
#                r'D:\publicDatasets\MNIST\t10k-images-idx3-ubyte.gz']
# cont_data = []
# for f in fnames_data:
#     cont_data.append(read_idx(f))
# cont_data = np.concatenate(cont_data)
# cont_data = cont_data.reshape((cont_data.shape[0], -1)) / 128. - 1.

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
fnames_cifar = glob.glob(r'D:\publicDatasets\CIFAR10\data*')
train_data=[np.load(f, allow_pickle=True, encoding='latin1') for f in fnames_cifar]
train_data = np.concatenate([a['data'] for a in train_data])/128. - 1.
train_idx = np.arange(train_data.shape[0])
np.random.shuffle(train_idx)
val = train_data[-int(0.2 * train_data.shape[0]):]
train_data = train_data[:-int(0.2 * train_data.shape[0])]
# data = np.concatenate([a['data'].reshape((10000,3,32,32)) for a in data])
# data = np.transpose(data, (0, 2, 3, 1))
# data = data/255.
# data = matplotlib.colors.rgb_to_hsv(data)
# data = data.reshape((-1,3072)) - 0.5
#
f = r'D:\publicDatasets\CIFAR10\test_batch'
test = np.load(f, allow_pickle=True, encoding='latin1')['data']/128. - 1.
# val = np.load(f, allow_pickle=True, encoding='latin1')['data']/255.
# val = val.reshape((10000,3,32,32))
# val = np.transpose(val, (0, 2, 3, 1))
# val = matplotlib.colors.rgb_to_hsv(val)
# val = val.reshape((-1,3072)) - 0.5
#
# #
# ## svhn
cont_data = scipy.io.loadmat(r'D:\publicDatasets\SVHN\test_32x32.mat')
cont_data = np.moveaxis(cont_data['X'],3,0)
cont_data = np.reshape(cont_data, (cont_data.shape[0],-1))/128. - 1.
# test = np.moveaxis(svhn['X'],3,0)/255.
# test = matplotlib.colors.rgb_to_hsv(test)
# test = np.reshape(test, (test.shape[0],-1))*2. - 1.


#### svd
# n = 3072
# _, _, vh = scipy.linalg.svd(train_data, full_matrices=False)
# train_data = np.matmul(train_data, vh.T)[:,:n]
# val = np.matmul(val, vh.T)[:,:n]
# test = np.matmul(test, vh.T)[:,:n]
# cont_data = np.matmul(cont_data, vh.T)[:,:n]

## build model
num_layers = 5
num_hidden=[100]
act = tf.nn.relu
model = mafs.MaskedAutoregressiveFlow(train_data.shape[1], num_hidden, act, num_layers, batch_norm=True)
# model_contrastive = mafs.MaskedAutoregressiveFlow(data.shape[1], num_hidden, act, num_layers, batch_norm=False, SCE=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# SCE=False
# ## train typical
# if not SCE:
t = train.Trainer(model) ## only pass model but don't re-initialize for SCE
# else:
#     t = train.Trainer(model, SCE=True, model_contrastive=model_contrastive) ## only pass model but don't re-initialize for SCE

## optimizer has some kind of parameters that require initialization
init=tf.global_variables_initializer()
sess.run(init)

# train_idx = np.arange(train_data.shape[0])
# N=1000
# np.random.shuffle(train_idx)

# t.train(sess, data[:60000,:], data[60000:,:], early_stopping=10, check_every_N=5, show_log=True, batch_size=100)
# t.train(sess, data[:1000,:], data[1000:,:], early_stopping=5, check_every_N=5, show_log=True, batch_size=1000)

## MLE training
t.train(sess, train_data.astype(np.float32), val_data=val.astype(np.float32), early_stopping=100, check_every_N=5, show_log=True, batch_size=100, max_iterations=20000, test_data=cont_data.astype(np.float32), saver_name='temp/tmp_model')
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

# import matplotlib.pyplot as plt
# import scipy.stats
s = model.gen(sess, 5000)
# out = model.eval(train_data, sess)
out2 = model.eval(test, sess)
out3 = model.eval(cont_data, sess)
sout_ = model.eval(s,sess)
# dist = scipy.stats.johnsonsu.fit(out)
# out = (np.arcsinh((out - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# out2 = (np.arcsinh((out2 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# out3 = (np.arcsinh((out3 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# sout = (np.arcsinh((sout_ - dist[-2]) / dist[-1]) * dist[1] + dist[0])

# plt.figure()
# plt.hist(out, 50, density=True, alpha=0.3, label='cifar_train')
# plt.hist(out2, 50, density=True, alpha=0.3, label='cifar_val')
# plt.hist(out3, 50, density=True, alpha=0.3, label='svhn')
# plt.hist(sout, 50, density=True, alpha=0.3, label='samples')
# plt.xlabel('MAF Density')
# plt.legend()
# plt.xlim([-6,3])
# plt.savefig(r'C:\Users\justjo\Desktop\maf_cifarVSsvhn_density_samples_'
#             r'.png', bbox_inches='tight')
#
# saver = tf.train.Saver()
# saver.save(sess, r'C:\Users\justjo\PycharmProjects\maf_tf\Models\maf_digitsVSfashion_h100_f5_tanh_dim784\model')
#
# # s = model.gen(sess, 5000)
# plt.figure();plt.scatter(data[:,0], data[:,1], alpha=1, label='data')
# plt.scatter(test[:,0], test[:,1], alpha=0.2)
# plt.scatter(s[:,0], s[:,1], alpha=0.3, label='sampled')
# st = np.matmul(s, vh)
# plt.scatter(st[:,0], st[:,1], alpha=0.2, label='MAF_SVD')
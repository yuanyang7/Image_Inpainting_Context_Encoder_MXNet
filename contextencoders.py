# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import subprocess
import mxnet as mx
import numpy as np
import lmdb
import matplotlib.pyplot as plt
import logging
import cv2
from datetime import datetime
import random
import math
import logging, logging.handlers
import logging.config
import time

def make_dcgan_sym(withDropOut=False, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm

    #build_reconstruction
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(name = 'conv1',
    data = data, kernel=(4, 4), stride = (2, 2), num_filter = 64,pad = (1,1),no_bias = no_bias)
    bn1 = mx.symbol.LeakyReLU(conv1, name = 'bn1', slope = 0.2)
    conv2 = mx.symbol.Convolution(name = 'conv2',
                                data = bn1, kernel = (4, 4), stride = (2, 2), num_filter = 64,pad = (1,1),no_bias = no_bias)
    bn2 = mx.symbol.LeakyReLU(data = BatchNorm(conv2,fix_gamma = fix_gamma, eps = eps), name = 'bn2', slope = 0.2)
    conv3 = mx.symbol.Convolution(name = 'conv3',
                                data = bn2, kernel = (4, 4), stride = (2, 2), num_filter = 128,pad = (1,1),no_bias = no_bias)
    bn3 = mx.symbol.LeakyReLU(data = BatchNorm(conv3,fix_gamma = fix_gamma, eps = eps), name = 'bn3', slope = 0.2)
    conv4 = mx.symbol.Convolution(name = 'conv4',
                                data = bn3, kernel = (4, 4), stride = (2, 2), num_filter = 256,pad = (1,1),no_bias = no_bias)
    if withDropOut == True:
        conv4 = mx.symbol.Dropout(data = conv4,p = 0.5)
    bn4 = mx.symbol.LeakyReLU(data = BatchNorm(conv4,fix_gamma = fix_gamma, eps = eps), name = 'bn4', slope = 0.2)
    conv5 = mx.symbol.Convolution(name = 'conv5',
                                data = bn4, kernel = (4, 4), stride = (2, 2), num_filter = 512,pad = (1,1),no_bias = no_bias)
    if withDropOut == True:
        conv5 = mx.symbol.Dropout(data = conv5,p = 0.5)
    bn5 = mx.symbol.LeakyReLU(data = BatchNorm(conv5,fix_gamma = fix_gamma, eps = eps), name = 'bn5', slope = 0.2)
    conv6 = mx.symbol.Convolution(name = 'conv6', data = bn5, kernel = (4, 4),
                                  num_filter = 4000,
                                  no_bias = no_bias)
    bn6 = mx.symbol.LeakyReLU(data = BatchNorm(conv6,fix_gamma = fix_gamma, eps = eps), name = 'bn60', slope = 0.2)

    deconv4 = mx.symbol.Deconvolution(bn6, kernel = (4,4),  num_filter = 512, name = 'deconv4',
                                      no_bias = no_bias)
    debn4 = mx.symbol.Activation(data = BatchNorm(deconv4,fix_gamma = fix_gamma, eps = eps), name = 'debn4', act_type = 'relu')
    deconv3 = mx.symbol.Deconvolution(debn4, kernel = (4,4), stride = (2, 2), num_filter = 256, name = 'deconv3',pad = (1,1),no_bias = no_bias)
    debn3 = mx.symbol.Activation(data = BatchNorm(deconv3,fix_gamma = fix_gamma, eps = eps), name = 'debn3', act_type = 'relu')
    deconv2 = mx.symbol.Deconvolution(debn3, kernel = (4,4), stride = (2, 2), num_filter = 128, name = 'deconv2',pad = (1,1),no_bias = no_bias)
    debn2 = mx.symbol.Activation(data = BatchNorm(deconv2,fix_gamma = fix_gamma, eps = eps), name = 'debn2', act_type = 'relu')
    deconv1 = mx.symbol.Deconvolution(debn2, kernel = (4,4), stride = (2, 2), num_filter = 64, name = 'deconv1',pad = (1,1),no_bias = no_bias)
    debn1 = mx.symbol.Activation(data = BatchNorm(deconv1,fix_gamma = fix_gamma, eps = eps), name = 'debn1', act_type = 'relu')
    recon0 = mx.symbol.Deconvolution(debn1, kernel = (4,4), stride = (2, 2), num_filter = 3, name = 'recon0',pad = (1,1),no_bias = no_bias)
    recon = mx.symbol.Activation(recon0,name = 'recon', act_type = 'tanh')

    recon = mx.symbol.Group([recon, mx.sym.BlockGrad(bn6)])

    crop_result = mx.symbol.Variable('crop_result')
    crop = mx.symbol.Variable('crop')
    gloss = mx.symbol.MakeLoss(mx.symbol.mean(mx.symbol.sum(mx.symbol.square((crop - crop_result),axis=(1,2,3) ))))

    data2 = mx.symbol.Variable('data2')
    label = mx.symbol.Variable('label')

    if gan=='ali':
        latent=mx.symbol.Variable('latent')

    conv11 = mx.symbol.Convolution(name = 'conv11',#
                                 data = data2, kernel = (4, 4), stride = (2, 2), num_filter = 64,pad = (1,1),no_bias = no_bias)
   bn11 = mx.symbol.LeakyReLU(data = conv11, name = 'bn11', slope = 0.2)
    conv22 = mx.symbol.Convolution(name = 'conv22',
                                 data = bn11, kernel = (4, 4), stride = (2, 2), num_filter = 128,pad = (1,1),no_bias = no_bias)
    bn22 = mx.symbol.LeakyReLU(data = BatchNorm(conv22,fix_gamma = fix_gamma, eps = eps), name = 'bn22', slope = 0.2)
    conv33 = mx.symbol.Convolution(name = 'conv33',
                                 data = bn22, kernel = (4, 4), stride = (2, 2), num_filter = 256,pad = (1,1),no_bias = no_bias)
    if withDropOut == True:
        conv33 = mx.symbol.Dropout(data = conv33,p = 0.5)
    bn33 = mx.symbol.LeakyReLU(data = BatchNorm(conv33,fix_gamma = fix_gamma, eps = eps), name = 'bn33', slope = 0.2)
    conv44= mx.symbol.Convolution(name = 'conv44',
                                data = bn33, kernel = (4, 4), stride = (2, 2), num_filter = 512,pad = (1,1),no_bias = no_bias)
    if withDropOut == True:
        conv44 = mx.symbol.Dropout(data = conv44,p = 0.5)
    bn44 = mx.symbol.LeakyReLU(data=BatchNorm(conv44,fix_gamma = fix_gamma, eps = eps), name = 'bn4', slope = 0.2)

    if gan == 'ali':
        conv49 = mx.symbol.Convolution(name = 'conv49',
                                     data = bn44, kernel = (4, 4), num_filter = 512, no_bias = no_bias)
        conv50 = mx.sym.concat(conv49, mx.sym.BlockGrad(latent))
        conv55 = mx.symbol.FullyConnected(conv50, num_hidden = 1)
    else:
        conv50 = mx.symbol.Convolution(name = 'conv50',
                               data = bn44, kernel = (4, 4), num_filter = 1, no_bias = no_bias)
        conv55 = mx.sym.Flatten(conv50)
    if gan != "wgan" :
        dloss = mx.symbol.LogisticRegressionOutput(data = conv55, label = label, name = 'dloss')
        if gan == 'ali_err':
            dloss = mx.symbol.Group([data2, bn11, bn22, bn33, bn44, conv50, conv55, dloss])

    if gan=="wgan" :
        return gloss,conv55,recon
    else:
        return gloss,dloss,recon

def get_mnist():
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234)
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (64,64)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 64, 64))
    X = np.tile(X, (1, 3, 1, 1))
    X_train = X[:60000]
    X_test = X[60000:]

    return X_train, X_test


class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]


class ImagenetIter(mx.io.DataIter):
    def __init__(self, path, batch_size, data_shape):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec = path,
            data_shape  = data_shape,
            batch_size  = batch_size,
            rand_crop   = False,
            rand_mirror = False,
            max_crop_size = 128,
            min_crop_size = 128)
        self.provide_data = [('data', (batch_size,) + data_shape)]
        self.provide_label = []

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        data = self.internal.getdata()
        data = data * (2.0/255.0)
        data -= 1
        return [data]
        
def get_lsun(nTrain, nTest, dumpDir=None, nOffset=0):
    lsun=[]
    env = lmdb.open(db_path, map_size=1099511627776, max_readers=100, readonly=True)
    n=0
    nSkiped=0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            if nSkiped<nOffset:
                nSkiped=nSkiped+1
                continue
            img = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
            img, rect  = mx.image.random_crop(mx.nd.array(img), (128, 128))
            lsun.append(img.asnumpy())
            if dumpDir!=None:
                cv2.imwrite(os.path.join(dumpDir, '%05d.png'%(n)), img.asnumpy())
            n=n+1
            if n==nTrain+nTest:
                break

    print('nTrain=', nTrain, 'nTest=', nTest)
    lsun =  np.asarray([x for x in lsun])
    lsun = lsun.transpose((0, 3, 1, 2))
    lsun = lsun.astype(np.float32)/(255.0/2) - 1.0
    lsun_train = lsun[:nTrain]
    lsun_test = lsun[nTrain:]

    return lsun_train, lsun_test

class LsunIter(mx.io.DataIter):
    def __init__(self, path, batch_size, data_shape):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec = path,
            data_shape  = data_shape,
            batch_size  = batch_size,
            rand_crop   = True,
            rand_mirror = True,
            max_crop_size = 256,
            min_crop_size = 192)

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def visual(title, X, epoch=-1,t=-1):
    print('visual {')
    print( X.shape)
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    print(X.shape)
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = int(np.ceil(np.sqrt(X.shape[0])))
    print(n)
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    cv2.imshow(title, buff)
    cv2.waitKey(1)

def fill_buf2(buf, i, img, shape, xOFF):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]
    m=m/2

    sx = (i%m + xOFF*m)*shape[0]
    sy = (i/m)*shape[1]

    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def visual2(title, X, epoch,t, valX):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = int(np.ceil(np.sqrt(X.shape[0])))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        if i<X.shape[0]/2:
            fill_buf2(buff, i, img, X.shape[1:3], 0)
    valX = valX.transpose((0, 2, 3, 1))
    valX = np.clip((valX+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    for i, img in enumerate(valX):
        if i<valX.shape[0]/2:
            fill_buf2(buff, i, img, valX.shape[1:3], 1)
    cv2.imshow(title, buff)
    cv2.waitKey(1)

def visual_yy(title, X,epoch,t):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0])).astype(np.int)
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    cv2.imwrite(os.path.join(result_path, 'img'+title+str(int(epoch))+'.'+str(int(t))+'.jpg'),buff)
    

def calcPSNR1(f,f2):
    mse = mx.nd.mean( mx.nd.square(mx.nd.subtract(f,f2)))
    if mse.asnumpy() == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse.asnumpy())


def calcSSIM(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    img1 = img1.asnumpy()
    img2 = img2.asnumpy()
    mu1 = cv2.GaussianBlur(img1,(11,11),sd)  
    mu2 = cv2.GaussianBlur(img2,(11,11),sd) 
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1,(11,11),sd)  - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2,(11,11),sd) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2,(11,11),sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_num / ssim_den
	
    return mx.nd.mean(mx.nd.array(ssim_map)).asnumpy()[0]    
    

def psnr1(X, Y):
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)
    n = int(np.ceil(np.sqrt(X.shape[0])))
    buff = np.zeros((n * X.shape[1], n * X.shape[2], X.shape[3]), dtype = np.uint8)

    Y = Y.transpose((0, 2, 3, 1))
    Y = np.clip((Y + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)

    s = 0
    i = 0
    for img, imgOut in zip(X, Y):
        fill_buf(buff, i, img, X.shape[1:3])
        fX = os.path.join(opt['dirTmp'], str(i)+'.png')
        cv2.imwrite(fX,img)
        i = i + 1
        fill_buf(buff, i, imgOut, X.shape[1:3])
        fY = os.path.join(opt['dirTmp'], str(i)+'.png')
        cv2.imwrite(fY,imgOut)
        i = i + 1
        psnr = calcPSNR1(mx.nd.array(fX), mx.nd.array(fY))
	ssim = calcSSIM(mx.nd.array(fX), mx.nd.array(fY))
        s += psnr
	s_ssim += ssim
        
        if i == X.shape[0]:
            break
    print("averPSNR=", s / (i / 2))
    print("averSSIM=", s_ssim / (i / 2))
    if opt['dspImage'] == 1:
        cv2.imshow("psnr", buff)
        cv2.waitKey(1)        
        
def dumpLMDB():
    nTrain=50000
    nTest=0
    random.seed(2017)
    lsun_train, lsun_test = get_lsun(0, 512, 'data/lsun/dining_room_test', 50000)
    print("dumpLMDB done")

def test(db_path, batch_size):
    print('test db_path=', db_path, 'gan=', opt['gan'], 'batch_size=', batch_size, 'random=',
          opt['random'], 'dropout=', opt['dropout'])
    nTrain=0
    nTest=64 
    random.seed(2017)
    val_iter = ImagenetIter(db_path, batch_size, (3, 128, 128))
    val_iter.reset()
    for t2, batch2 in enumerate(val_iter):
        images_ori = mx.nd.array(batch2.data[0].asnumpy())
        break
    setMaskParam(opt['random']==1)
    images_out=inpainting(images_ori, True)
    psnr1(images_ori.asnumpy(), images_out.asnumpy())

def inpainting(images_ori, is_train=True):
    p1=mx.ndarray.slice(images_ori,begin=(0,0,0,0),end=(batch_size,3,l1,ori_size))
    p2=mx.ndarray.slice(images_ori,begin=(0,0,l1,0),end=(batch_size,3,l4,l2))
    p3=mx.ndarray.slice(images_ori,begin=(0,0,l1,l3),end=(batch_size,3,l4,ori_size))
    p4=mx.ndarray.slice(images_ori,begin=(0,0,l4,0),end=(batch_size,3,ori_size,ori_size))
    p1_0=mx.ndarray.slice(images_ori,begin=(0,0,0,0),end=(batch_size,3,l1-overlap_size,ori_size))
    p2_0=mx.ndarray.slice(images_ori,begin=(0,0,l1-overlap_size,0),end=(batch_size,3,l4+overlap_size,l2-overlap_size))
    p3_0=mx.ndarray.slice(images_ori,begin=(0,0,l1-overlap_size,l3+overlap_size),end=(batch_size,3,l4+overlap_size,ori_size))
    p4_0=mx.ndarray.slice(images_ori,begin=(0,0,l4+overlap_size,0),end=(batch_size,3,ori_size,ori_size))
    images_ctx_val=mx.nd.concat(p2,crop_para,p3,dim=3)
    images_ctx_val=mx.nd.concat(p1,images_ctx_val,p4,dim=2)
    modA.forward(mx.io.DataBatch([images_ctx_val],[]), is_train=is_train)
    outA_val=modA.get_outputs()[0].as_in_context(mx.cpu())
    images_ctx_val=mx.nd.concat(p2_0,outA_val,p3_0,dim=3)
    images_ctx_val=mx.nd.concat(p1_0,images_ctx_val,p4_0,dim=2)
    return images_ctx_val

def setMaskParam(bRand):
    global crop_x, crop_y, l1, l2, l3, l4
    crop_x = int((128 - hiding_size) / 2)   
    crop_y = int((128 - hiding_size) / 2)   
    if bRand:
        crop_x = int(random.uniform(margin,ori_size-hiding_size-margin))
        crop_y = int(random.uniform(margin,ori_size-hiding_size-margin))

    l1 = crop_x + overlap_size
    l2 = crop_y + overlap_size
    l3 = crop_y + hiding_size - overlap_size
    l4 = crop_x + hiding_size - overlap_size

class UTCFormatter(logging.Formatter):
    converter = time.gmtime

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'utc': {
            '()': UTCFormatter,
            'format': '%(asctime)s %(message)s',
        },
        'local': {
            'format': '%(asctime)s %(message)s',
    }
    },
    'handlers': {
        'console1': {
            'class': 'logging.StreamHandler',
            'formatter': 'utc',
        },
        'console2': {
            'class': 'logging.StreamHandler',
            'formatter': 'local',
        },
    },
    'root': {
        'handlers': ['console1', 'console2'],
        }
}

if __name__ == '__main__':
    opt = {                            #dict
        'mode'        : 'train',        # ‘psnr’ 'dumpLMDB' 'test'
        'gpu'            : 1,                # 0 for CPU
        'netA'        : None,
        'netD'        : None,
        'dataset'        : 'imagenet',        # 'imagenet': .rec  'lsun': lmdb
        'check_point'  : 1,            # 1: True
        'dspImage'     : 1,
        'batch_size'    : 64,
        'tVisual'        : 40,
        'train_path'   : 'data/lsun/dining_room_train50k.rec',
        'val_path'     : 'data/lsun/dining_room_val.rec',
        'test_path'     : 'data/lsun/dining_room_test.rec',
        'dirTmp'      : '/Volumes/RamDisk',
        'loggerServer'  : '192.168.2.5',
        'loggerPort'    : logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        'gan'            : "dcgan",     #'dcgan', 'wgan', 'ali'
        'dropout'       : 0,
        'random'        : 0,
        'overlap_size'   : 4,
        'lrD'          :0.0002,
    }
    print(opt)
    for k in opt :
        if os.getenv(k) != None:
            opt[k]=os.getenv(k)
            print(k, opt[k])
            if k=='lrD':
                opt[k]=float(os.getenv(k))
            else:
                if k!='mode' and k!='dataset' and k!='train_path' and k!='val_path' and k!='netA' and k!='netD' and k!='gan' and k!='loggerServer':
                    opt[k]=int(os.getenv(k))
    print(opt)

    rootLogger = logging.getLogger('')
    rootLogger.setLevel(logging.DEBUG)
    socketHandler = logging.handlers.SocketHandler(opt['loggerServer'], opt['loggerPort'])
    # don't bother with a formatter, since a socket handler sends the event as
    # an unformatted pickle
    rootLogger.addHandler(socketHandler)
    print(opt['loggerServer'], ":", opt['loggerPort'])

   # Now, we can log to the root logger, or any other logger. First the root...
    rootLogger.info('The local time is %s', time.asctime())


    if opt['mode']=='psnr':
        f=mx.nd.array(cv2.imread("test.jpeg"))
        f2=mx.nd.array(cv2.imread("test-1.jpeg"))
        psnr=calcPSNR1(f,f2)
        ssim=calcSSIM(f,f2)
        print('PSNR=', psnr)
        print('SSIM=', ssim)
        sys.exit()   

    # =============setting============
    dataset = opt['dataset']
    check_point = opt['check_point']==1
    print('check_point=', check_point)
    batch_size = opt['batch_size']
    print('batch_size=', batch_size)

    db_path = '../data/lsun/classroom_val_lmdb/'
    db_path = './data/lsun/dining_room_val_lmdb/'
    db_path = './data/lsun/dining_room_train_lmdb/'

    imgnet_path = 'oxbuild_val.rec'
    imgnet_path = 'data/oxbuild.rec'
    imgnet_path = 'fn_oxbuild.rec'
    imgnet_val_path = 'data/oxbuild_val.rec'
    imgnet_val_path = 'oxbuild_val-yy.rec'
    imgnet_val_path = 'fn_oxbuild-val.rec'
    imgnet_path = 'data/lsun/dining_room_train-300.rec'
    imgnet_path = 'data/lsun/dining_room_train.rec'
    imgnet_val_path = 'data/lsun/dining_room_val.rec'
    imgnet_path = opt['train_path']
    imgnet_val_path = opt['val_path']

    if opt['mode']=='dumpLMDB':
        dumpLMDB()
        sys.exit()

    result_path = './result/'
    ndf = 64
    ngf = 64 
    nc = 3
    lr = 0.001 
    lrD=lr
    lrD = opt['lrD']
    lr =  lrD*10
    beta1 = 0.5 
    if opt['gpu']==1:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    #====
    n_epochs = 10000*10
    lambda_recon = 0.999
    lambda_adv = 1-lambda_recon
    print(lambda_recon)
    print(lambda_adv)

    ori_size=128
    hiding_size = 64
    overlap_size = opt['overlap_size']       
    margin=24 
    crop_x=crop_y=l1=l2=l3=l4=0
    print("overlap_size=", overlap_size)
    
    gan= opt['gan']
    print("gan=["+gan+"]")
    wclip = 0.01
    one = mx.nd.ones((batch_size,1), ctx) / batch_size
    mone = -mx.nd.ones((batch_size,1), ctx) / batch_size
    

    if opt['dropout']==1:
        withDropOut=True
    else:
        withDropOut=False
    print("withDropOut=", withDropOut)
    symG, symD, symA = make_dcgan_sym(withDropOut)    

    label = mx.nd.zeros((batch_size,), ctx)
    images_ctx = mx.nd.zeros((batch_size,nc,128,128,), ctx)
    images_crops = mx.nd.zeros((batch_size, nc, hiding_size,hiding_size,), ctx)
    
    z=mx.nd.zeros((batch_size,nc,64-overlap_size*2,64-overlap_size*2,))
    crop_para_ori=mx.nd.array([[[2*117. / 255. - 1.]],[[2*104. / 255. - 1.]],[[2*123. / 255. - 1.]]])
    crop_para=mx.nd.broadcast_add(z,crop_para_ori)

    # =============module G=============
    modG = mx.mod.Module(symbol = symG, data_names = ('crop_result',),label_names = ('crop',), context = ctx)
    modG.bind(data_shapes = [('crop_result',(batch_size,nc,hiding_size,hiding_size,))],
              label_shapes = [('crop', (batch_size,nc,hiding_size,hiding_size,))],inputs_need_grad = True)
    modG.init_params(initializer = mx.init.Normal(0.02))
    if gan == "wgan":
        modG.init_optimizer(
            optimizer = 'rmsprop',
            optimizer_params = {
                'learning_rate': lr,
            })
    else:
        modG.init_optimizer(
            optimizer = 'adam',
            optimizer_params = {
                'learning_rate': lr,
                'wd': 0.,
                'beta1': beta1,
            })
    mods = [modG]

    # =============module D=============
    if gan == "wgan":
        modD = mx.mod.Module(symbol = symD, data_names = ('data2',),label_names = None, context = ctx)
        modD.bind(data_shapes = [('data2', (batch_size,nc,hiding_size,hiding_size,))],
                  inputs_need_grad = True)
        modD.init_params(initializer = mx.init.Normal(0.02))
        modD.init_optimizer(
            optimizer = 'rmsprop',
            optimizer_params = {
                'learning_rate': lrD,
            })
    else:
        if gan == 'ali':
            modD = mx.mod.Module(symbol = symD, data_names = ('data2','latent'), label_names = ('label',), context = ctx)
            modD.bind(data_shapes = [('data2', (batch_size,nc,hiding_size,hiding_size,)),
                                   ('latent', (batch_size,4000,1,1,)),
                                   ],
                      label_shapes = [('label', (batch_size,))],
                      inputs_need_grad = True)
            print(modD.data_shapes)
        else:
            modD = mx.mod.Module(symbol = symD, data_names = ('data2',), label_names = ('label',), context = ctx)
            modD.bind(data_shapes = [('data2', (batch_size,nc,hiding_size,hiding_size,))],
                      label_shapes = [('label', (batch_size,))],
                      inputs_need_grad = True)
        modD.init_params(initializer = mx.init.Normal(0.02))
        modD.init_optimizer(
            optimizer = 'adam',
            optimizer_params = {
                'learning_rate': lrD,
                'wd': 0.,
                'beta1': beta1,
            })
    mods.append(modD)

    # =============module A=============
    if gan == "wgan":
        modA = mx.mod.Module(symbol = symA, data_names = ('data',),label_names = None,context = ctx)
        modA.bind(data_shapes = [('data',(batch_size,nc,128,128,))],)
        modA.init_params(initializer = mx.init.Normal(0.02))
        modA.init_optimizer(
            optimizer = 'rmsprop',
            optimizer_params = {
                'learning_rate': lr,
            })
    else:
        modA = mx.mod.Module(symbol = symA, data_names = ('data',),label_names = None,context = ctx)
        modA.bind(data_shapes = [('data',(batch_size,nc,128,128,))],)
        modA.init_params(initializer = mx.init.Normal(0.02))
        modA.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': lr,
                'wd': 0.,
                'beta1': beta1,
            })
    mods.append(modA)

    if (opt['netA'] != None):
        print('load netA=', opt['netA'])
        modA.load_params(opt['netA'])
    if (opt['netD'] != None):
        print('load netD=', opt['netD'])
        modD.load_params(opt['netD'])
    if opt['mode'] == 'test':
        test(opt['train_path'], batch_size)
        test(opt['test_path'], batch_size)
        sys.exit()

# ==============data==============
    if dataset == 'imagenet':
        train_iter = ImagenetIter(imgnet_path, batch_size, (3, 128, 128))
        val_iter = ImagenetIter(imgnet_val_path, batch_size, (3, 128, 128))
    elif dataset == 'lsun':
        assert 0
        nTrain=256*2
        nTrain=300*10*4
        nTest=256*2
        lsun_train, lsun_test = get_lsun(nTrain, nTest)
        print('get_lsun() done')
        train_iter = mx.io.NDArrayIter(lsun_train, batch_size=batch_size)
        val_iter = mx.io.NDArrayIter(lsun_test, batch_size=batch_size)
        print('lsun iter done')

    # ============printing==============
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = mx.mon.Monitor(10, norm_stat, pattern = ".*output|d1_backward_data", sort = True)
    mon = None
    if mon is not None:
        for mod in mods:
            pass

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label * np.log(pred + 1e-12) + (1. - label) * np.log(1. - pred + 1e-12)).mean()

    mG = mx.metric.CustomMetric(fentropy)
    mD = mx.metric.CustomMetric(fentropy)
    mACC = mx.metric.CustomMetric(facc)

    print('Training...')
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    for epoch in range(n_epochs):
        train_iter.reset()
        val_iter.reset()
        for t, batch in enumerate(train_iter):
            images_ori = mx.nd.array(batch.data[0].asnumpy())
            images_ctx = images_ori.copy()
            
            setMaskParam(opt['random'] == 1)
            
            images_crops=mx.ndarray.slice(images_ori,begin=(0,0,crop_x,crop_y),end=(batch_size,3,crop_x+hiding_size,crop_y+hiding_size))#

            p1=mx.ndarray.slice(images_ori,begin=(0,0,0,0),end=(batch_size,3,l1,ori_size))
            p2=mx.ndarray.slice(images_ori,begin=(0,0,l1,0),end=(batch_size,3,l4,l2))
            p3=mx.ndarray.slice(images_ori,begin=(0,0,l1,l3),end=(batch_size,3,l4,ori_size))
            p4=mx.ndarray.slice(images_ori,begin=(0,0,l4,0),end=(batch_size,3,ori_size,ori_size))
            
            images_ctx=mx.nd.concat(p2,crop_para,p3,dim=3)
            images_ctx=mx.nd.concat(p1,images_ctx,p4,dim=2)

            if mon is not None:
                mon.tic()
            modA.forward(mx.io.DataBatch([images_ctx],[]), is_train=True)
            outA = modA.get_outputs()[0]

            if gan == 'ali_dsp':
                outputs = modA.get_outputs()
                for output in outputs:
                    print("outputA=", output)

            if gan == "wgan" :
                #clip
                for params in modD._exec_group.param_arrays:
                    for param in params:
                        mx.nd.clip(param, -wclip, wclip, out = param)
        
            label[:] = 0
            if gan == "wgan":
                modD.forward(mx.io.DataBatch([outA], []), is_train = True)
                modD.backward([mone])
            else:
                if gan == 'ali':
                    latent = modA.get_outputs()[1]
                    modD.forward(mx.io.DataBatch([outA, latent], [label]), is_train = True)
                    if gan == 'ali_dsp':
                        print("modD.forward done");
                        outputs = modD.get_outputs()
                        for output in outputs:
                            print("outputD=", output)
                        sys.exit()
                else:
                    modD.forward(mx.io.DataBatch([outA], [label]), is_train = True)
                modD.backward()

            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

            if gan!="wgan":
                modD.update_metric(mD, [label])    
                modD.update_metric(mACC, [label])

            # update discriminator on real
            label[:] = 1
            if gan=="wgan":
                modD.forward(mx.io.DataBatch([images_crops],[]), is_train=True)
                modD.backward([one])
            else:
                if gan=='ali':
                    modD.forward(mx.io.DataBatch([images_crops, latent], [label]), is_train=True)
                else:
                    modD.forward(mx.io.DataBatch([images_crops], [label]), is_train=True)
                modD.backward()
            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            modD.update() 
            
            if  gan!="wgan":
                modD.update_metric(mD, [label])
                modD.update_metric(mACC, [label])

            # update generator
            label[:] = 1
            if gan == 'ali':
                modD.forward(mx.io.DataBatch([outA, latent], [label]), is_train=True)
            else:
                modD.forward(mx.io.DataBatch([outA], [label]), is_train=True)
            if gan == "wgan":
                modD.backward([mone])                
            else:
                modD.backward()
            diffD = modD.get_input_grads() 

            #--l2 loss--
            modG.forward(mx.io.DataBatch([outA], [images_crops]), is_train=True)
            modG.backward()
            diffG = modG.get_input_grads()
            modG.update()

            #--sum loss--
            if overlap_size > 0:
                #overlap
                z_overlap = mx.nd.ones((batch_size,nc,hiding_size,hiding_size,))
                overlap_params = mx.nd.array([[[10]],[[10]],[[10]]])
                overlap_params = mx.nd.broadcast_mul(z_overlap,overlap_params)
                z_notoverlap = mx.nd.ones((batch_size,nc,hiding_size - overlap_size * 2,hiding_size - overlap_size * 2,))
                p1_overlap = mx.ndarray.slice(overlap_params,begin = (0,0,0,0),end = (batch_size,3,overlap_size,hiding_size))
                p2_overlap = mx.ndarray.slice(overlap_params,begin = (0,0,overlap_size,0),end = (batch_size,3,hiding_size - overlap_size,overlap_size))
                p3_overlap = mx.ndarray.slice(overlap_params,begin = (0,0,overlap_size,hiding_size - overlap_size),end = (batch_size,3,hiding_size - overlap_size,hiding_size))
                p4_overlap = mx.ndarray.slice(overlap_params,begin = (0,0,hiding_size - overlap_size,0),end = (batch_size,3,hiding_size,hiding_size))
                overlap_mask = mx.nd.concat(p2_overlap,z_notoverlap,p3_overlap,dim = 3)
                overlap_mask = mx.nd.concat(p1_overlap,overlap_mask,p4_overlap,dim = 2)
                if opt['gpu'] == 1:
                    overlap_mask = overlap_mask.as_in_context(mx.gpu())
                diffG[0] = mx.nd.multiply(diffG[0], overlap_mask)
            diffA = lambda_adv * diffD[0] + lambda_recon * diffG[0]
            modA.backward([diffA])
            modA.update()

            mG.update([label], modD.get_outputs())

            if mon is not None:
                mon.toc_print()

            t += 1
            if t % 4== 0:
                print('epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get())
                rootLogger.info('%s epoch:%d iter:%d metric:%f %f %f ', time.asctime(), epoch, t, mACC.get()[1], mG.get()[1], mD.get()[1])
                mACC.reset()
                mG.reset()
                mD.reset()
                

            if t%opt['tVisual'] == 0 : 
                outA = outA.as_in_context(mx.cpu())
                p1_0 = mx.ndarray.slice(images_ori,begin = (0,0,0,0),end = (batch_size,3,l1 - overlap_size,ori_size))
                p2_0 = mx.ndarray.slice(images_ori,begin = (0,0,l1 - overlap_size,0),end = (batch_size,3,l4 + overlap_size,l2 - overlap_size))
                p3_0 = mx.ndarray.slice(images_ori,begin = (0,0,l1 - overlap_size,l3 + overlap_size),end = (batch_size,3,l4 + overlap_size,ori_size))
                p4_0 = mx.ndarray.slice(images_ori,begin = (0,0,l4 + overlap_size,0),end = (batch_size,3,ori_size,ori_size))
                
                images_ctx = mx.nd.concat(p2_0,outA,p3_0,dim = 3)
                images_ctx = mx.nd.concat(p1_0,images_ctx,p4_0,dim = 2)
                
                print("val_iter.getdata")、
                for t2, batch2 in enumerate(val_iter):
                    images_ori = mx.nd.array(batch2.data[0].asnumpy())
                    break
                print("val_iter.getdata done")
                
                images_ctx_val = inpainting(images_ori)
                
                visual2('result_img', images_ctx.asnumpy(),epoch,t, images_ctx_val.asnumpy())

            if t%4==0:
                val_iter.reset()

        if check_point==True:
            #outA=outA.asnumpy()
            #outA=mx.nd.array(outA)
            #outA=outA.as_in_context(mx.cpu())
            #images_ctx=mx.nd.concat(p2,outA,p3,dim=3)
            #images_ctx=mx.nd.concat(p1,images_ctx,p4,dim=2)
            #visual('result_img', images_ctx.asnumpy(),epoch,t)  
            pass

        if epoch % 5 == 0 :    
                print('Saving...')
                modA.save_params('%s%s_A_%s-%04d.params' % (result_path,dataset, stamp, epoch))
                modG.save_params('%s%s_G_%s-%04d.params' % (result_path,dataset, stamp, epoch))
                modD.save_params('%s%s_D_%s-%04d.params' % (result_path,dataset, stamp, epoch))
                t = opt['dspImage']
                opt['dspImage'] = 0
                test(opt['train_path'], 256)
                test(opt['val_path'], 256)
                opt['dspImage'] = 0



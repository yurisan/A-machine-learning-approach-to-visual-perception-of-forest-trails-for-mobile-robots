import argparse
import random
from PIL import Image, ImageOps
import glob
import cv2
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import datasets
from chainer import iterators
from chainer import initializers
from chainer import serializers
from chainer.training import updaters 
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer.datasets import LabeledImageDataset


class ColumnNet(chainer.Chain):
    def __init__(self, train=True):
        super(ColumnNet, self).__init__(
            conv1=L.Convolution2D(3, 32, 4, initialW=initializers.Uniform(0.05)),
            conv2=L.Convolution2D(32, 32, 4, initialW=initializers.Uniform(0.05)),
            conv3=L.Convolution2D(32, 32, 4, initialW=initializers.Uniform(0.05)),
            conv4=L.Convolution2D(32, 32, 3, initialW=initializers.Uniform(0.05)),
            l1=L.Linear(512, 200, initialW=initializers.Uniform(0.05)),
            l2=L.Linear(200, 3, initialW=initializers.Uniform(0.05)),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2)
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        return self.l2(self.l1(h))

def random_flip(image, label):
    if random.randint(0, 1) == 0:
        pass
    else:
        image = image.astype(np.uint8)
        image = Image.fromarray(image.transpose(1,2,0))
        image = ImageOps.mirror(image)
        image = np.asarray(image).transpose(2,0,1)
        image = image.astype(np.float32)

        if label == 0:
            label = np.array(2)
        elif label == 2:
            label = np.array(0)

    return image, label


def random_rotate(image):
    angle = random.uniform(-15, 15)
    image = image.astype(np.uint8)
    image = Image.fromarray(image.transpose(1,2,0))
    image = image.rotate(angle)
    image = np.asarray(image).transpose(2,0,1)
    image = image.astype(np.float32)

    return image


def random_scale(image):
    im_height = 101
    im_width = 101
    scaled_width = random.randint(91, 111)

    if scaled_width < 101:
        image = image.astype(np.uint8)
        image = Image.fromarray(image.transpose(1,2,0))
        image = image.resize((scaled_width, scaled_width), Image.BILINEAR) 
        bg    = Image.new("RGB", (im_width, im_height), (0, 0, 0))
        center = (im_width - scaled_width) // 2, (im_height - scaled_width) // 2
        bg.paste(image, center)
        image = np.asarray(bg).transpose(2,0,1)
        image = image.astype(np.float32)
    elif scaled_width > 101:
        image = image.astype(np.uint8)
        image = Image.fromarray(image.transpose(1,2,0))
        image = image.resize((scaled_width, scaled_width), Image.BILINEAR) 
        top = (scaled_width - im_width) // 2
        left = (scaled_width - im_height) // 2
        bottom = top + im_height
        right = left + im_width
        image = image.crop((top, left, bottom, right))
        image = np.asarray(image).transpose(2,0,1)
        image = image.astype(np.float32)
    
    return image


def random_translate(image):
    im_height = 101
    im_width = 101

    translate_width = random.randint(-10, 10)
    translate_height = random.randint(-10, 10)

    if translate_height < 0:
        top = -1 * translate_height
        bottom = im_height + translate_height
    else:
        top = 0
        bottom = im_height - translate_height

    if translate_width < 0:
        left = -1 * translate_width
        right = im_height + translate_width
    else:
        left = 0
        right = im_height - translate_height
        
    image = image[:, top:bottom, left:right]
    image = image.astype(np.uint8)
    image = Image.fromarray(image.transpose(1,2,0))
    bg    = Image.new("RGB", (im_width, im_height), (0, 0, 0))
    bg.paste(image, (top,left))
    image = np.asarray(bg).transpose(2,0,1)
    image = image.astype(np.float32)

    return image


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        # shared process 
        image = image.astype(np.uint8)
        image = Image.fromarray(image.transpose(1,2,0))
        image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        image = np.asarray(image).transpose(2,0,1)
        image = image.astype(np.float32)
        image -= self.mean[:, None, None]

        # random flip
        image, label = random_flip(image, label)

        # random affine distortion
        # random rotate
        image = random_rotate(image)
        # random translate
        image = random_translate(image)
        # random scale
        image = random_scale(image)

        image *= (1.0 / 255.0)  # Scale to [0, 1]

        return image, label


class PreprocessedTestDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        # shared process 
        image = image.astype(np.uint8)
        image = Image.fromarray(image.transpose(1,2,0))
        image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        image = np.asarray(image).transpose(2,0,1)
        image = image.astype(np.float32)
        image -= self.mean[:, None, None]
        image *= (1.0 / 255.0)  # Scale to [0, 1]

        return image, label


def main():
    parser = argparse.ArgumentParser(description='ColumnNet')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true') 
    args = parser.parse_args()
 

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    Model = ColumnNet()
    model = L.Classifier(Model)
   
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.SGD(lr=0.005)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.05))

    mean = np.load(args.mean)
    train = PreprocessedDataset(args.train, args.root, mean, 101)
    val   = PreprocessedTestDataset(args.val, args.root, mean, 101)

    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, shuffle=False)


    if args.test:
        val_interval = 5, 'epoch'
        log_interval = 1, 'epoch'
    else:
        val_interval = 5, 'epoch'
        log_interval = 1, 'epoch'


    # Set up an optimizer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')
 
    # Set up a trainer
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),trigger=val_interval)
    trainer.extend(extensions.ExponentialShift('lr', 0.95), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'validation/main/map', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/map'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/map'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    # Run the training
    trainer.run()
    chainer.serializers.save_npz(args.out + '/columnnet.model', Model)


if __name__ == "__main__":
    main()

import argparse
from PIL import Image
import glob
import cv2
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import datasets
from chainer import iterators
from chainer import serializers
from chainer.training import updaters 
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer.datasets import LabeledImageDataset


class ColumnNet(chainer.Chain):
    def __init__(self, train=True):
        super(ColumnNet, self).__init__(
            conv1=L.Convolution2D(3, 32, 4),
            conv2=L.Convolution2D(32, 32, 4),
            conv3=L.Convolution2D(32, 32, 4),
            conv4=L.Convolution2D(32, 32, 3),
            l1=L.Linear(512, 200),
            l2=L.Linear(200, 3),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2)
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        return self.l2(self.l1(h))


class TransformDataset(object):
    def __init__(self, dataset, transform):

        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index):
        in_data = self._dataset[index]
        if isinstance(index, slice):
            return [self._transform(in_data_elem) for in_data_elem in in_data]
        else:
            return self._transform(in_data)

    def __len__(self):
        return len(self._dataset)


def resize(img):
    img = Image.fromarray(img.transpose(1,2,0))
    img = img.resize((101, 101), Image.BILINEAR)
    return np.asarray(img).transpose(2,0,1)


def transform(inputs):
    img, label = inputs
    img = resize(img.astype(np.uint8))
    img = img.astype(np.float32)
    return img, label


def load_dataset(lines):
    pathsAndLabels = []
    for line in lines:
        words = line.replace("\n", "").split(",")
        pathsAndLabels.append(np.asarray(["/data/" + words[0] + "/", words[1]]))

    # Make data for chainer
    fnames = []
    labels = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + "*")
        for imgName in imagelist:
            try:
                file_check = Image.open(imgName)
                file_check = np.array(file_check, dtype=np.uint8)
                fnames.append(imgName)
                labels.append(label)
            except Exception:
                pass

    dataset = LabeledImageDataset(list(zip(fnames, labels)))
    return TransformDataset(dataset, transform)

def main():
    parser = argparse.ArgumentParser(description='ColumnNet')
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
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    
    # Load the ColumnNet dataset
    f = open('train_list.txt')
    train_lines = f.readlines()
    f.close()

    f = open('val_list.txt')
    val_lines = f.readlines()
    f.close()

    #dataset = LabeledImageDataset(list(zip(fnames, labels)))
    #transform_dataset = TransformDataset(dataset, transform)

    #train, val = datasets.split_dataset_random(transform_dataset, int(len(dataset) * 0.8), seed=0)

    train = load_dataset(train_lines)
    val = load_dataset(val_lines)

    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, shuffle=False)


    if args.test:
        val_interval = 5, 'epoch'
        log_interval = 1, 'epoch'
    else:
        val_interval = 100000, 'iteration'
        log_interval = 1000, 'iteration'


    # Set up an optimizer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')
 
    # Set up a trainer
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),trigger=val_interval)
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
    chainer.serializers.save_npz('result/columnnet.model', Model)


if __name__ == "__main__":
    main()

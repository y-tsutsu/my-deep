import sys
import os
import random
import matplotlib.pyplot as plt
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


class MLP(chainer.Chain):
    ''' Network definition '''

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),     # n_in -> n_units
            l2=L.Linear(None, n_units),     # n_units -> n_units
            l3=L.Linear(None, n_out),       # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def setup_optimizer(model):
    ''' Setup optimizer '''
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    return optimizer


def load_mnist_dataset(batchsize):
    ''' MNISTの手書き数字データのダウンロード '''
    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)
    return (len(train), train_iter, test_iter)


def draw_digit(train):
    ''' 手書き数字データを描画する関数 '''
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size), range(size))
    Z = train.reshape(size, size)   # convert from vector to 28x28 matrix
    Z = Z[::-1, :]                  # flip vertical
    plt.xlim(0, 27)
    plt.ylim(0, 27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

    plt.show()


def check_draw_digit(train_iter):
    ''' 手書き数字データを試しに3点描画する関数 '''
    for train in train_iter:
        draw_digit(train[0][0])
        draw_digit(train[42][0])
        draw_digit(train[99][0])


def do_training(train_iter, test_iter, model, optimizer, model_file_name):
    ''' 学習 '''
    epoch = 2
    out = 'result'

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()

    # Save model
    chainer.serializers.save_npz(model_file_name, model)


def judge(train_iter, N, batchsize, model):
    ''' 判定 '''
    plt.style.use('fivethirtyeight')

    def inner_draw_digit(data, n, ans, recog):
        size = 28
        plt.subplot(10, 10, n)
        Z = data.reshape(size, size)    # convert from vector to 28x28 matrix
        Z = Z[::-1, :]                  # flip vertical
        plt.xlim(0, 27)
        plt.ylim(0, 27)
        plt.pcolor(Z)
        plt.title('ans={}, recog={}'.format(ans, recog), size=8)
        plt.gray()
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')

        if ans != recog:
            print('Mismatch!:', n)

    plt.figure(figsize=(15, 15))

    ridx = random.randint(0, N // batchsize - 1)
    for idx, train in enumerate(train_iter):
        if idx == ridx:
            cnt = 0
            for x_train, y_train in train:
                x = chainer.Variable(x_train.astype(
                    np.float32).reshape(1, 784))
                y = model.predictor(x)
                cnt += 1
                inner_draw_digit(x_train, cnt, y_train, np.argmax(y.data))
            break

    plt.show()


def main():
    unit = 1000
    batchsize = 100

    # model
    model = L.Classifier(MLP(unit, 10))

    # Load the MNIST dataset
    N, train_iter, test_iter = load_mnist_dataset(batchsize)

    # Draw digit image
    # check_draw_digit(train_iter)

    model_file_name = 'mymodel.npz'
    if os.path.isfile(model_file_name):
        # Reload
        chainer.serializers.load_npz(model_file_name, model)
    else:
        # Training
        optimizer = setup_optimizer(model)
        do_training(train_iter, test_iter, model, optimizer, model_file_name)

    # Judge
    judge(train_iter, N, batchsize, model)


if __name__ == '__main__':
    main()

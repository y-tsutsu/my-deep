import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers, serializers
import chainer.functions as F
import sys
import os


def download_mnist_data():
    '''
    MNISTの手書き数字データのダウンロード
    HOME/scikit_learn_data/mldata/mnist-original.mat にキャッシュされる
    '''
    print('fetch MNIST dataset')
    mnist = fetch_mldata('MNIST original')
    # mnist.data : 70,000件の784次元ベクトルデータ
    mnist.data = mnist.data.astype(np.float32)
    mnist.data /= 255     # 0-1のデータに変換

    # mnist.target : 正解データ（教師データ）
    mnist.target = mnist.target.astype(np.int32)

    return mnist


def draw_digit(data):
    ''' 手書き数字データを描画する関数 '''
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size), range(size))
    Z = data.reshape(size, size)    # convert from vector to 28x28 matrix
    Z = Z[::-1, :]                  # flip vertical
    plt.xlim(0, 27)
    plt.ylim(0, 27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

    plt.show()


def check_draw(mnist):
    ''' 手書き数字データを試しに3点描画する関数 '''
    draw_digit(mnist.data[5])
    draw_digit(mnist.data[12345])
    draw_digit(mnist.data[33456])


def create_model(n_units):
    '''
    Prepare multi-layer perceptron model
    多層パーセプトロンモデルの設定
    入力 784次元、出力 10次元
    '''
    model = FunctionSet(l1=F.Linear(784, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, 10))
    return model


def forward(model, x_data, y_data, train=True):
    '''
    Neural net architecture
    ニューラルネットの構造
    '''
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    # 多クラス分類なので誤差関数としてソフトマックス関数の
    # 交差エントロピー関数を用いて、誤差を導出
    return (F.softmax_cross_entropy(y, t), F.accuracy(y, t))


def check_relu():
    ''' F.reluテスト '''
    x_data = np.linspace(-10, 10, 100, dtype=np.float32)
    x = Variable(x_data)
    y = F.relu(x)

    plt.figure(figsize=(7, 5))
    plt.ylim(-2, 10)
    plt.plot(x.data, y.data)
    plt.show()


def check_drop():
    '''
    dropout(x, ratio=0.5, train=True) テスト
    x: 入力値
    ratio: 0を出力する確率
    train: Falseの場合はxをそのまま返却する
    return: ratioの確率で0を、1−ratioの確率で,x*(1/(1-ratio))の値を返す
    '''
    n = 1000
    v_sum = 0
    for i in range(n):
        x_data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        x = Variable(x_data)
        dr = F.dropout(x, ratio=0.6, train=True)

        for j in range(6):
            sys.stdout.write(str(dr.data[j]) + ', ')
        print('')
        v_sum += dr.data

    # outputの平均がx_dataとだいたい一致する
    sys.stdout.write(str((v_sum / float(n))))


def setup_optimizer(model):
    ''' Setup optimizer '''
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    return optimizer


def do_training(batchsize, n_epoch, N, x_train, y_train, x_test, y_test, N_test, model, optimizer, model_file_name):
    ''' 学習 '''
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # Learning loop
    for epoch in range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        # N個の順番をランダムに並び替える
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        # 0〜Nまでのデータをバッチサイズごとに使って学習
        for i in range(0, N, batchsize):
            x_batch = x_train[perm[i:i + batchsize]]
            y_batch = y_train[perm[i:i + batchsize]]

            # 勾配を初期化
            optimizer.zero_grads()
            # 順伝播させて誤差と精度を算出
            loss, acc = forward(model, x_batch, y_batch)
            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

            train_loss.append(loss.data)
            train_acc.append(acc.data)
            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        # 訓練データの誤差と、正解精度を表示
        print('train mean loss={}, accuracy={}'.format(
            sum_loss / N, sum_accuracy / N))

        # evaluation
        # テストデータで誤差と、正解精度を算出し汎化性能を確認
        sum_accuracy = 0
        sum_loss = 0
        for i in range(0, N_test, batchsize):
            x_batch = x_test[i:i + batchsize]
            y_batch = y_test[i:i + batchsize]

            # 順伝播させて誤差と精度を算出
            loss, acc = forward(model, x_batch, y_batch, train=False)

            test_loss.append(loss.data)
            test_acc.append(acc.data)
            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        # テストデータでの誤差と、正解精度を表示
        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))

    # 学習したパラメーターを保存
    serializers.save_npz(model_file_name, model)

    # 精度と誤差をグラフ描画
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_acc)), train_acc)
    plt.plot(range(len(test_acc)), test_acc)
    plt.legend(['train_acc', 'test_acc'], loc=4)
    plt.title('Accuracy of digit recognition.')
    plt.plot()
    plt.show()


def judge(N, x_train, y_train, model):
    ''' 判定 '''
    plt.style.use('fivethirtyeight')

    def draw_digit3(data, n, ans, recog):
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

    plt.figure(figsize=(15, 15))

    cnt = 0
    for idx in np.random.permutation(N)[:100]:
        # Forwarding for prediction
        xxx = x_train[idx].astype(np.float32)
        h1 = F.dropout(
            F.relu(model.l1(Variable(xxx.reshape(1, 784)))), train=False)
        h2 = F.dropout(F.relu(model.l2(h1)), train=False)
        y = model.l3(h2)
        cnt += 1
        draw_digit3(x_train[idx], cnt, y_train[idx], np.argmax(y.data))

    plt.show()


def main():
    plt.style.use('ggplot')
    # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
    batchsize = 100
    # 学習の繰り返し回数
    n_epoch = 20
    # 中間層の数
    n_units = 1000

    mnist = download_mnist_data()
    # check_draw(mnist)

    # 学習用データを N個、検証用データを残りの個数と設定
    N = 60000
    x_train, x_test = np.split(mnist.data, [N])
    y_train, y_test = np.split(mnist.target, [N])
    N_test = y_test.size

    model = create_model(n_units)

    # check_relu()
    # check_drop()

    model_file_name = 'mymodel.npz'
    if os.path.isfile(model_file_name):
        # 読み込み
        serializers.load_npz(model_file_name, model)
    else:
        # 学習
        optimizer = setup_optimizer(model)
        do_training(batchsize, n_epoch, N, x_train, y_train, x_test,
                    y_test, N_test, model, optimizer, model_file_name)

    # 判定
    judge(N, x_train, y_train, model)


if __name__ == '__main__':
    main()

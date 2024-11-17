from starter import NN
import autograd.numpy as anp
import numpy as np

layers = [3, 2, 3]
learning_rate = 0.01
weights = [[[0.0, 0.0],
            [0.4963, 0.7682],
            [0.0885, 0.1320],
            [0.3074, 0.6341]],
           [[0.0, 0.0, 0.0],
            [0.4901, 0.8964, 0.4556],
            [0.6323, 0.3489, 0.4017]]]
params = [weights[:-1], weights[-1]]
nn = NN(layers, learning_rate, weights=weights)


def test_init():
    assert len(nn.h[0]) == 2
    assert len(nn.q[0]) == 2
    assert len(nn.x) == 3
    assert len(nn.z) == 3


def test_sigmoid():
    assert np.round(nn.sigmoid(0.3451), 4) == 0.5854


def test_softmax():
    z = [0.6969, 0.7511, 0.5272]
    assert np.round(nn.softmax(z)[0], 4) == 0.3449


def test_ft():
    X = anp.array([0.35, 0.20, 0.50])
    out = nn.ft(X, nn.params[0])

    assert np.round(out[0], 4) == 0.5854
    assert np.round(out[1], 4) == 0.6485
    assert np.round(nn.h[0][1], 4) == 0.6485
    assert out.shape == (2,)


def test_model():
    X = anp.array([0.35, 0.20, 0.50])
    p = nn.model(X, params)

    assert np.round(p[1], 4) == 0.3641
    assert np.round(nn.p[1], 4) == 0.3641
    assert np.round(nn.z[0], 4) == 0.6969


def test_loss():
    X = anp.array([0.35, 0.20, 0.50])
    y = anp.array([1.0, 0.0, 0.0])
    p = nn.model(X, params)
    loss = nn.loss(p, y)

    assert np.round(loss, 4) == 1.8611


def test_back_prop():
    X = anp.array([0.35, 0.20, 0.50])
    y = anp.array([1.0, 0.0, 0.0])
    nn.model(X, params)
    gradients = nn.back_propagation(X, y)

    assert np.round(gradients[0][1][0], 4) == 0.0117
    assert np.round(gradients[1][1][0], 4) == -0.3835

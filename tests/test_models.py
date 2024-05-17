from milligrad.tensor import Tensor
from tests.utils import create_randn_pair, create_xavier_pair, create_one_hot_pair

import pytest

import torch
import torch.nn.functional as F

import numpy as np


@pytest.mark.parametrize("w1_shape, w2_shape", [
    ((100, 50), (50, 10)),
    ((10, 5), (5, 1)),
    ((768, 512), (512, 2)),
])
def test_perceptron(w1_shape, w2_shape):
    x_milli, x_torch = create_randn_pair((16, w1_shape[0]))
    y_milli, y_torch = create_randn_pair((16, w2_shape[1]))
    
    w1_milli, w1_torch = create_xavier_pair(w1_shape)
    b1_milli, b1_torch = create_randn_pair((w1_shape[1],))
    
    w2_milli, w2_torch = create_xavier_pair(w2_shape)
    b2_milli, b2_torch = create_randn_pair((w2_shape[1],))
    
    # Forward pass
    z1_milli = x_milli @ w1_milli + b1_milli
    z1_torch = x_torch @ w1_torch + b1_torch
    
    a1_milli = z1_milli.relu()
    a1_torch = torch.relu(z1_torch)
    
    z2_milli = a1_milli @ w2_milli + b2_milli
    z2_torch = a1_torch @ w2_torch + b2_torch
    
    loss_milli = ((z2_milli - y_milli)**2).mean()
    loss_torch = ((z2_torch - y_torch)**2).mean()
    
    np.testing.assert_allclose(
        loss_milli.data, loss_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-5
    )
    
    # Backward pass
    loss_milli.backward()
    loss_torch.backward()
    
    np.testing.assert_allclose(
        x_milli.grad, x_torch.grad.numpy(),
        err_msg="Backward pass input data gradient mismatch", atol=1e-5
    )
    
    np.testing.assert_allclose(
        w1_milli.grad, w1_torch.grad.numpy(),
        err_msg="Backward pass w1 gradient mismatch", atol=1e-5
    )
    
    np.testing.assert_allclose(
        b1_milli.grad, b1_torch.grad.numpy(),
        err_msg="Backward pass b1 gradient mismatch", atol=1e-5
    )
    
    np.testing.assert_allclose(
        w2_milli.grad, w2_torch.grad.numpy(),
        err_msg="Backward pass w2 gradient mismatch", atol=1e-5
    )
    
    np.testing.assert_allclose(
        b2_milli.grad, b2_torch.grad.numpy(),
        err_msg="Backward pass b2 gradient mismatch", atol=1e-5
    )


@pytest.mark.parametrize("batch_size, seq_len, input_dim, hidden_dim", [
    (3, 3, 10, 20),
    (4, 4, 15, 25),
    (5, 100, 8, 12),
])
def test_rnn(batch_size, seq_len, input_dim, hidden_dim):
    wi_milli, wi_torch = create_xavier_pair((input_dim, hidden_dim))
    wh_milli, wh_torch = create_xavier_pair((hidden_dim, hidden_dim))
    bh_milli, bh_torch = create_randn_pair((hidden_dim,))
    wo_milli, wo_torch = create_xavier_pair((hidden_dim, input_dim))
    bo_milli, bo_torch = create_randn_pair((input_dim,))

    x_milli, x_torch = create_one_hot_pair((batch_size, seq_len), input_dim)
    y_milli, y_torch = create_one_hot_pair((batch_size, seq_len), input_dim)
    h0_milli, h0_torch = create_randn_pair((batch_size, hidden_dim))

    # Forward pass
    h_milli_list = []
    h_torch_list = []

    h_milli = h0_milli
    h_torch = h0_torch
    for t in range(seq_len):
        xt_milli = x_milli[:, t, :]
        xt_torch = x_torch[:, t, :]
        
        h_milli = (xt_milli @ wi_milli + h_milli @ wh_milli + bh_milli).tanh()
        h_torch = torch.tanh(xt_torch @ wi_torch + h_torch @ wh_torch + bh_torch)
        
        h_milli_list.append(h_milli)
        h_torch_list.append(h_torch)

    # Stack hidden states and pass through a linear layer
    h_milli_stacked = Tensor.stack(h_milli_list, axis=1)
    h_torch_stacked = torch.stack(h_torch_list, dim=1)
    
    y_hat_milli = h_milli_stacked @ wo_milli + bo_milli
    y_hat_torch = h_torch_stacked @ wo_torch + bo_torch
    
    np.testing.assert_allclose(
        y_hat_milli.data, y_hat_torch.detach().numpy(),
        err_msg="Forward pass mismatch for stacked hidden states through linear layer", atol=1e-5
    )
    
    loss_milli = -(y_milli * y_hat_milli.log_softmax()).sum(-1).mean()
    loss_torch = -(y_torch * F.log_softmax(y_hat_torch, dim=-1)).sum(-1).mean()
    
    np.testing.assert_allclose(
        loss_milli.data, loss_torch.detach().numpy(),
        err_msg="Forward pass mismatch in RNN", atol=1e-5
    )

    # Backward pass
    loss_milli.backward()
    loss_torch.backward()

    
    np.testing.assert_allclose(
        h0_milli.grad, h0_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch for h0", atol=1e-5
    )
    
    np.testing.assert_allclose(
        wi_milli.grad, wi_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch for wi", atol=1e-5
    )
    
    np.testing.assert_allclose(
        wh_milli.grad, wh_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch for wh", atol=1e-5
    )
    
    np.testing.assert_allclose(
        bh_milli.grad, bh_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch for bh", atol=1e-5
    )
    
    np.testing.assert_allclose(
        wo_milli.grad, wo_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch for wo", atol=1e-5
    )
    
    np.testing.assert_allclose(
        bo_milli.grad, bo_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch for bo", atol=1e-5
    )

if __name__ == "__main__":
    test_rnn(3, 3, 10, 20)
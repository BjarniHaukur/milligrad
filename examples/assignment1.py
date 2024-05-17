import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from milligrad import Tensor, topological_sort


########## 1. Read CIFAR-10

# read in pickled cifar-10 data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# read in the data and convert the pixel values to float32
train = unpickle('cifar10/data_batch_1')
val = unpickle('cifar10/data_batch_2')
test = unpickle('cifar10/test_batch')
x_train, y_train = train[b"data"].astype(np.float32), np.array(train[b"labels"])
x_val, y_val = val[b"data"].astype(np.float32), np.array(val[b"labels"])
x_test, y_test = test[b"data"].astype(np.float32), np.array(test[b"labels"])



# drop 50% of val for ease of use
ind = int(len(x_val)*0.5)
ind = np.random.permutation(len(x_val))[:ind] # there may otherwise be some structure in the data
x_val, y_val = x_val[ind], y_val[ind]


#################### 2. Preprocess the data

# normalize the input data with the mean and std of the training data
mean, std = x_train.mean(), x_train.std()
x_train, x_val, x_test = (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std

# one-hot encode the labels
y_train, y_val, y_test = np.eye(10)[y_train], np.eye(10)[y_val], np.eye(10)[y_test]


########## 3 - 4. Initialize the model and write the forward function (i.e. \_\_call\_\_)


class Model:
    def __init__(self):
        self.w = Tensor.randn(3072, 10) * 0.01
        self.b = Tensor.randn(10) * 0.01
        
    def __call__(self, x):
        return x @ self.w + self.b
    
    def parameters(self):
        return [self.w, self.b]

# # 5. Computing the cost is simply:
# $$\text{cost} = -(y * \log{\operatorname{softmax}(\text{model prediction})).\operatorname{sum}().\operatorname{mean}()}$$
# All these operations are defined on the Tensor class. By performing these operations, we construct a computational graph which we can then backpropagate through to compute the gradients.

# # 6. Accuracy

def accuracy(y, y_hat):
    return np.mean(y.argmax(axis=1) == y_hat.argmax(axis=1))

# # 7. Computing the gradients
# After we compute the cost, we can call `cost.backward()` to backpropagate through the computational graph and compute the gradients. All `Tensor` objects will have their `.grad` attribute updated with the gradient of the cost with respect to that tensor. So the gradient of the weights/biases can simply be accessed by `model.weight.grad` and `model.bias.grad`.


w = Tensor.randn(3072, 10) * 0.01
b = Tensor.randn(10) * 0.01

x = Tensor.randn(100, 3072)
y = Tensor.randn(100, 10)

y_hat = x @ w + b
cost = -(y * y_hat.log_softmax()).sum().mean() + 1e-3 * (w ** 2).sum().sum()

print(f"The cost is {cost.data}")
# The cost is 1.2976623704333754


print("The computational graph:")
print([x._grad_fn for x in reversed(topological_sort(cost)) if x._grad_fn])
# The computational graph:
# ['+', 'mean', 'sum', '-', '*', 'log_softmax', '+', '@', '*', 'sum', 'sum', '**2']
# note that the regularization is at the end of the graph as was to be expected

cost.backward() # backpropagate from the cost tensor
print(f"The gradient of w has mean: {w.grad.mean()} and std: {w.grad.std()}")
# The gradient of w has mean: 4.444123036534561e-08 and std: 0.0970924337613376

# # 8. Validating that the gradients are correct
# 
# I have a repo with a more in-depth implementation of this Tensor class [here](https://github.com/BjarniHaukur/milligrad). There I validate that the forward and backward passes are correct by comparison to manually computed gradients and by comparing to PyTorch's autograd. Therefore I am confident that it will also work here.


# not a normal way of using the Tensors library but it is just to fit in with the provided code
def ComputeCost(X, Y, W, b, lamda):
	""" Computes the forward pass of the corresponding model and returns the cost """
	X, Y, W, b = Tensor(X), Tensor(Y), Tensor(W), Tensor(b[:,0])
	
	y_hat = X @ W + b
	cost = -(Y * y_hat.log_softmax()).sum().mean() + lamda * ((W**2).sum().sum())
	return cost.data

def ComputeGradsNum(X, Y, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[1]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	c = ComputeCost(X, Y, W, b, lamda)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[1]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

LAMBDA = 1e-3

# for computational reasons we only use the first 100 samples and the first 1000 features
x_subset = x_train[:100, :1000]
y_subset = y_train[:100]

# we use these exact weights for ComputeGradsNum, ComputeGradsNumSlow and the Tensor autograd implementation
w = np.random.rand(1000, 10) * 0.01
b = np.random.rand(10, 1) * 0.01

grad_w_num, grad_b_num = ComputeGradsNum(x_subset, y_subset, w, b, lamda=LAMBDA, h=1e-5)
grad_w_slow, grad_b_slow = ComputeGradsNumSlow(x_subset, y_subset, w, b, lamda=LAMBDA, h=1e-5)


w_tensor, b_tensor = Tensor(w), Tensor(b.squeeze()) # no need for the extra dimension for the bias

y_hat = Tensor(x_subset) @ w_tensor + b_tensor
cost = -(Tensor(y_subset) * y_hat.log_softmax()).sum().mean() + LAMBDA * (w_tensor ** 2).sum().sum()
cost.backward()

grad_w_tensor, grad_b_tensor = w_tensor.grad, b_tensor.grad

# I chose to use np.testing to validate, it simply compares each element of the two arrays and raises an error if they are not within some tolerance level.


atol = 1e-6
np.testing.assert_allclose(grad_w_num, grad_w_tensor, atol=atol)
np.testing.assert_allclose(grad_b_num.squeeze(), grad_b_tensor, atol=atol)
np.testing.assert_allclose(grad_w_slow, grad_w_tensor, atol=atol)
np.testing.assert_allclose(grad_b_slow.squeeze(), grad_b_tensor, atol=atol)

# and it passses!

# # 9. Training the model with mini-batch stochastic gradient descent


class SGD: # the 'stochastic' nature depends on the usage, not the implementation, but this is clear
    def __init__(self, params:list[Tensor], lr:float=0.01):
        self.params = params
        self.lr = lr

    def step(self): # perform the update
        for p in self.params:
            p.data -= self.lr * p.grad
            
    # most tensors are created at runtime and therefore we need not zero the gradients
    # however, weights and biases persist and we need to zero the gradients between each gradient descent step
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)



np.random.seed(400)

EPOCHS = 40
BATCH_SIZE = 100
LR = 0.001
LAMBDA = 1.0

model = Model()
optim = SGD(model.parameters(), lr=LR)

def train(model, optim, x_train, y_train, x_val, y_val):
    train_losses_epoch = []
    val_losses_epoch = []
    for _ in tqdm(range(EPOCHS)):
        train_losses = []
        permutation = np.random.permutation(len(x_train))
        for i in range(len(x_train) // BATCH_SIZE):
            idxs = permutation[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            x = Tensor(x_train[idxs])
            y = Tensor(y_train[idxs])
            y_hat = model(x)
            # cross-entropy loss with L2 regularization
            # sum over classes, then mean over batch for cross-entropy
            loss = -(y * y_hat.log_softmax()).sum().mean() + LAMBDA * (model.w ** 2).sum().sum()
            # y acts as a selector so GD minimizes -log(y_hat) i.e. maximizing the likelihood
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            train_losses.append(loss.data)
        
        val_losses = []
        permutation = np.random.permutation(len(x_val))
        for i in range(len(x_val) // BATCH_SIZE):
            idxs = permutation[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            x = Tensor(x_val[idxs])
            y = Tensor(y_val[idxs])
            y_hat_val = model(x)
            val_loss = -(y * y_hat_val.log_softmax()).sum().mean() + LAMBDA * (model.w ** 2).sum().sum()
            val_losses.append(val_loss.data)

        
        train_losses_epoch.append(np.mean(train_losses))
        val_losses_epoch.append(np.mean(val_losses))
        
    return train_losses_epoch, val_losses_epoch

train_losses, val_losses = train(model, optim, x_train, y_train, x_val, y_val)


plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training with LAMBDA={LAMBDA} and LR={LR}")
plt.legend()
plt.show()


print(f"Training accuracy: {accuracy(y_train, model(Tensor(x_train)).data):%}")
# Training accuracy: 40.940000%
print(f"Validation accuracy: {accuracy(y_val, model(Tensor(x_val)).data):%}")
# Validation accuracy: 33.728571%
print(f"Test accuracy: {accuracy(y_test, model(Tensor(x_test)).data):%}")
# Test accuracy: 37.680000%


# display the weights for each class
def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()
 
montage(model.w.data.T)
# this shows only slight differences in the weights, which is expected as the model is not very accurate






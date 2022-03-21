# Numpy手撕两层神经网络

## 热身

一个Relu神经网络，一个隐藏层，没有bias。预测从x到y，使用L2 loss。

- $$
  h = W_1*X + b_1
  $$

  

- $$
  a = max(0, h)
  $$

  

- $$
  y_hat = W_2*a + b_2
  $$

  

完全使用Numpy来计算前向神经网络，loss，反向传播。 



## 代码

为了手撕方便，公式先不用b1,b2

```python
N, D_in, H, D_out = 64, 1000, 100, 10

#随机创建一些训练数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(N, D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for it in range(500):
    #forward pass
    h = x.dot(w1) #拿到一个N x H的向量
    h_relu = np.maximum(h, 0) #N x H
    y_pred = h_relu.dot(w2) #N x D_out
    
    #compute loss
    loss = np.square(y_pred - y).sum()
    print(it, loss)
    
    #Backward pass
    #compute the gradient
    '''
    y = ax + b
    dy / dx = a
    dy / da = x
     '''
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    #update weight of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    
    
    
    
    
```





# 利用Tensor修改

```python
N, D_in, H, D_out = 64, 1000, 100, 10

#随机创建一些训练数据
x = torch.randn(N, D_in, requires_grad=True)
y = torch.randn(N, D_out, requires_grad=True)

w1 = torch.randn(N, D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for it in range(500):
    #forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    #compute loss
    loss = (y_pred - y).pow(2).sum()
    print(it, loss)
    
	#Backward pass
	loss.backward()
    
    
	#update weight of w1 and w2
    with torch.no_grad():
		w1 -= learning_rate * grad_w1
		w2 -= learning_rate * grad_w2
   		w1.grad.zero_()
    	w2.grad.zero_()
    
```

## 简单的autograd

```python 
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w*x + b # y = 2 * 1 + 3

y.backward()

print(w.grad) # 1
print(x.grad) # 2
print(b.grad) # 1
```

## torch.nn

```python
import torch.nn as nn
N, D_in, H, D_out = 64, 1000, 100, 10

#随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H), #w_1 * x + b_1 线性层
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),

)
# model = model.cuda()


loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-6
for it in range(500):
    #forward pass
    y_pred = model(x)
    #compute loss
    loss = loss_fn(y_pred, y)
    print(it, loss)
    
    model.zero_grad()
	#Backward pass
	loss.backward()
    
    
    
	#update weight of w1 and w2
    with torch.no_grad():
		for param in model.parameters():
            param -= learning_rate * param.grad

```



## optim

```python
import torch.nn as nn
N, D_in, H, D_out = 64, 1000, 100, 10

#随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H), #w_1 * x + b_1 线性层
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),

)
'''
torch.nn.init.normal_(model[0].weight)
torch.nn.init.normal_(model[2].weight)
'''

# model = model.cuda()


loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for it in range(500):
    #forward pass
    y_pred = model(x)
    #compute loss
    loss = loss_fn(y_pred, y)
    print(it, loss)
    
    optimizer.zero_grad()
   
	#Backward pass
	loss.backward()
    
    #update model parameters
    optimizer.step()
    
    
	
```



## nn.Module

```python
import torch.nn as nn
N, D_in, H, D_out = 64, 1000, 100, 10

#随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=False)
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)
     
    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred
        
model = TwoLayerNet(D_in, H, D_out)
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for it in range(500):
    #forward pass
    y_pred = model(x)
    #compute loss
    loss = loss_fn(y_pred, y)
    print(it, loss)
    
    optimizer.zero_grad()
   
	#Backward pass
	loss.backward()
    
    #update model parameters
    optimizer.step()
    
```


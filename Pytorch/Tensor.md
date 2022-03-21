# Tensor

## 基础语法

```python
#构造一个未初始化的5x3矩阵
x = torch.empty(5,3)

#构造一个随机初始化的矩阵
x = torch.rand(5,3)

#构造一个全部为0，类型为long的矩阵
torch.zeros(5,3, dtype=long)
x.dtype
#或者
x = torch.zeros(5,3).long()
x.dtype

#用数据直接构造tensor,以下构造出的是1行2列
x = torch.tensor([5.5, 3])

#从一个已有的tensor构建一个tensor。这些方法会重用原来的tensor特征，e.g:数据类型，除非提供新的数据
x = x.new_ones(5,3 dtype=torch.double)

#得到tensor的形状
x.shape()
x.size()


```

## 基础运算

```python
#以上得到了x矩阵，现构造一个新的y矩阵
y = torch.rand(5,3)
#加法
x + y
torch.add(x, y)

#in-place加法，以下加完后的y跟原来的y的地址仍然一样，若不用add_(),那么就会给新的y分配一个新的地址
y.add_(x)

#关于indexing
x[:, 1:] #x为五行三列，该操作可以保留所有行，从第二列向后取
x[1:, 1:] #从第二行第二列向后取

#resizeing
x = torch.randn(4,4) #随机生成一个4x4矩阵
y = x.view(16)  #利用.view将原来矩阵重新变成1x16矩阵
z = x.view(2, 8) #利用.view将原来矩阵变成2x8矩阵
q = x.view(2,-1) #-1是自动判断列数，2x8=4x4，所以是2x8


```

## Tensor和Numpy转换

```python
#torch tensor转为numpy array
a = torch.ones(5)
b = a.numpy()

#改变Numpy array里的值，tensor也跟着变了
b[1] = 2 #此时的a也跟着变化了

#numpy array转为torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)   #同样跟着变换

```

## Cuda Tensor

```python
#tensor搬到GPU再搬回CPU
if torch.cuda.is_available():
	device = torch.device('cuda')
	y = torch.ones_like(x, device = device)
	x = x.to(device)
    z = x + y
    print(z.to('cpu', torch.double))
    
#转为numpy时要搬到cpu
y.to('cpu').data.numpy()
y.cpu().data.numpy()

    
```

## 测试GPU代码

```python
import torch
import time

print(torch.__version__)        # 返回pytorch的版本
print(torch.cuda.is_available())        # 当CUDA可用时返回True

a = torch.randn(10000, 1000)    # 返回10000行1000列的张量矩阵
b = torch.randn(1000, 2000)     # 返回1000行2000列的张量矩阵

t0 = time.time()        # 记录时间
c = torch.matmul(a, b)      # 矩阵乘法运算
t1 = time.time()        # 记录时间
print(a.device, t1 - t0, c.norm(2))     # c.norm(2)表示矩阵c的二范数

device = torch.device('cuda')       # 用GPU来运行
a = a.to(device)
b = b.to(device)

# 初次调用GPU，需要数据传送，因此比较慢
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

# 这才是GPU处理数据的真实运行时间，当数据量越大，GPU的优势越明显
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

```


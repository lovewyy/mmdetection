import torch

print(torch.__version__)

'''
a = torch.tensor([[2., 4.]], requires_grad=True)
b = torch.zeros(1, 2)
b[0, 0] = a[0, 0] ** 2 + a[0, 1]
b[0, 1] = a[0, 1] ** 3 + a[0, 0] * 2
out = 2 * b
out.backward(torch.tensor([[1., 0.]]), retain_graph=True)
result = a.grad
print(result)
'''


a = torch.tensor([[1., 2.]], requires_grad=True)
w = torch.tensor([[0.1], [0.01]],requires_grad=True)
loss = torch.mm(a, w)
print(loss)
# loss.backward(torch.tensor([[1.], [2.], [3.]]))
print(loss.size())
loss.backward()
print(w.grad)

b = torch.tensor([[[1., 2., 3], [3., 4., 5], [5., 6., 7.]],[[1., 2., 3], [3., 4., 5], [5., 6., 7.]]])
a = torch.squeeze(b)
print(a)
print(b)
print(a.size())
print(b.size())








import torch;
x = torch.rand(2, 5, 3)
print(x)
# print(torch.__version__)

scalar1 = torch.tensor(2)
print(scalar1)
print(scalar1.ndim)
print(scalar1.shape)

vector = torch.tensor([10, 10, 10])
print(vector)
print(vector.ndim)
print(vector.shape)
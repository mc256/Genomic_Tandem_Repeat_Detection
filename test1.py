import torch

print(torch.cuda.is_available())
print(torch.cuda.set_device(0))


x = torch.rand(13, 13).cuda()
y = torch.rand(13,13).cuda()
print(x+y)


x = torch.rand(5, 3)
y = torch.rand(5, 3)
x = x.cuda()
y = y.cuda()

print(x+ y)
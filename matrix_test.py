import torch

m1 = torch.tensor([1, 2, 3])
m2 = torch.tensor([4, 5, 6])
outer_result = torch.outer(m1, m2)
print(outer_result)
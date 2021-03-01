import torch
torch.cuda.is_available()


big_zero = torch.tensor([[1,2,3],
                        [4,5,6]]).repeat((5,1))




big_zero.size()
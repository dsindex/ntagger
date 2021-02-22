import sys
import torch

# from https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor

# source 3D : [batch_size, seq_size, dim] = [2, 5, 3]
source = torch.FloatTensor([
    [[ 0.2413, -0.6667,  0.2621],
     [-0.4216,  0.3722, -1.2258],
     [-0.2436, -1.5746, -0.1270],
     [ 1.6962, -1.3637,  0.8820],
     [ 0.3490, -0.0198,  0.7928]],

    [[-0.0973,  2.3106, -1.8358],
     [-1.9674,  0.5381,  0.2406],
     [ 3.0731,  0.3826, -0.7279],
     [-0.6262,  0.3478, -0.5112],
     [-0.4147, -1.8988, -0.0092]]
     ])

# index 2D : [batch_size, * <= seq_size] = [2, 4]
index = torch.LongTensor([[0, 1, 2, 3], 
                          [1, 2, 3, 4]])

# compute offset
offset = torch.arange(0, source.size(0) * source.size(1), source.size(1))
print(offset)
'''
tensor([0, 5])
'''

# add offset to index
index = index + offset.unsqueeze(1)
print(index)
'''
tensor([[0, 1, 2, 3],
        [6, 7, 8, 9]])
'''

# reshape source to 2D
reshaped_source = source.reshape(-1, source.shape[-1])
print(reshaped_source)
'''
tensor([[ 0.2413, -0.6667,  0.2621],
        [-0.4216,  0.3722, -1.2258],
        [-0.2436, -1.5746, -0.1270],
        [ 1.6962, -1.3637,  0.8820],
        [ 0.3490, -0.0198,  0.7928],
        [-0.0973,  2.3106, -1.8358],
        [-1.9674,  0.5381,  0.2406],
        [ 3.0731,  0.3826, -0.7279],
        [-0.6262,  0.3478, -0.5112],
        [-0.4147, -1.8988, -0.0092]])
'''

# index slicing, 2D -> 3D
source = reshaped_source[index]
print(source)
'''
tensor([[[ 0.2413, -0.6667,  0.2621],
         [-0.4216,  0.3722, -1.2258],
         [-0.2436, -1.5746, -0.1270],
         [ 1.6962, -1.3637,  0.8820]],

        [[-1.9674,  0.5381,  0.2406],
         [ 3.0731,  0.3826, -0.7279],
         [-0.6262,  0.3478, -0.5112],
         [-0.4147, -1.8988, -0.0092]]])
'''

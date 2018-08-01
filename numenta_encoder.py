import torch

class Encoder:
    def __init__(w, scale, output_size):
        self.w = w
        self.osize = output_size
        self.scale = scale
        self.grid = torch.sparse.LongTensor((360 * scale, 180 * scale))

    def encode(lat, long, velocity):
        

if __name__ == '__main__':
    scale = int(1e6)

    indices = torch.LongTensor([[0, 0, 1], [0, 1, 1]])
    values = torch.FloatTensor([2, 3, 4])
    sizes = [360 * scale, 180 * scale]
    grid = torch.sparse_coo_tensor(indices, values, sizes)

    print(grid)
    print(vars(grid))

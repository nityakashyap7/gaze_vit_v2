import torch
class GazeToMask():
    def __init__(self, N=84, sigmas=[10,10,10,10], coeficients = [1,1,1,1]):
        self.N = N
        assert len(sigmas) == len(coeficients)
        self.sigmas = sigmas
        self.coeficients = coeficients
        self.masks = self.initialize_mask()

    def generate_single_gaussian_tensor(self, map_size, mean_x, mean_y, sigma):
        x = torch.arange(map_size, dtype=torch.float32).unsqueeze(1).expand(map_size, map_size)
        y = torch.arange(map_size, dtype=torch.float32).unsqueeze(0).expand(map_size, map_size)
        # Calculate the Gaussian distribution for each element
        gaussian_tensor = (1 / (2 * torch.pi * sigma ** 2)) * torch.exp(
            -((x - mean_x) ** 2 + (y - mean_y) ** 2) / (2 * sigma ** 2))

        return gaussian_tensor

    def initialize_mask(self):
        temp_map = []
        N = self.N
        for i in range(len(self.sigmas)):
            temp = self.generate_single_gaussian_tensor(2 * N, N - 1, N - 1, self.sigmas[i])
            temp = temp/ temp.max()
            temp_map.append(self.coeficients[i]*temp)

        temp_map = torch.stack(temp_map, 0)


        return temp_map

    def find_suitable_map(self, Nx2=168, index=0, mean_x=0.5, mean_y=0.5):
        # returns a map such that the center of the gaussian is located at (mean_x, mean_y) of the map
        start_x, start_y = int((1 - mean_x) * Nx2 / 2), int((1 - mean_y) * Nx2 / 2)
        desired_map = self.masks[index][start_y:start_y + Nx2 // 2, start_x:start_x + Nx2 // 2]
        return desired_map

    def find_bunch_of_maps(self, means=[[0.5, 0.5]], offset_start=0):
        current_maps = torch.zeros([self.N, self.N])
        bunch_size = len(means)
        assert bunch_size + offset_start <= len(self.sigmas), f'The bunch is too long! It\'s length is {bunch_size}'
        Nx2 = self.N * 2
        for i in range(bunch_size):
            mean_x, mean_y = means[i][0], means[i][1]
            # mean_x, mean_y = 0, 0
            temp = self.find_suitable_map(Nx2, i+offset_start, mean_x, mean_y)
            current_maps = current_maps + temp

        return current_maps / torch.max(current_maps)

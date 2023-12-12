import numpy as np
import matplotlib.pyplot as plt

class SelfOrganizingMap:
    def __init__(self, dimensions=(10, 10), input_dim=3, learning_rate=0.5, radius=None):
        self.dimensions = dimensions
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.grid = np.random.random((dimensions[0], dimensions[1], input_dim))

        if radius is None:
            self.radius = max(dimensions) / 2
        else:
            self.radius = radius

        self.radius_decay = 0.995
        self.learning_rate_decay = 0.995

        self.closest_data_index = -np.ones((self.dimensions[0], self.dimensions[1]), dtype=int)

    def find_bmu(self, sample):
        distances = np.sqrt(((self.grid - sample) ** 2).sum(axis=2))
        return np.unravel_index(distances.argmin(), distances.shape)

    def update_weights(self, bmu, sample):
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                node_position = np.array([x, y])
                bmu_position = np.array(bmu)
                distance = np.linalg.norm(node_position - bmu_position)

                if distance <= self.radius:
                    influence = np.exp(-distance / (2 * (self.radius ** 2)))
                    self.grid[x, y] += influence * self.learning_rate * (sample - self.grid[x, y])
                    self.closest_data_index[x, y] = np.where((data == sample).all(axis=1))[0][0]

    def train(self, data, num_iterations=10000):
        for i in range(num_iterations):
            sample = data[np.random.randint(0, len(data))]
            bmu = self.find_bmu(sample)
            self.update_weights(bmu, sample)

            self.learning_rate *= self.learning_rate_decay
            self.radius *= self.radius_decay

    def plot_grid(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.grid)

        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                ax.text(j, i, self.closest_data_index[i, j], ha='center', va='center', color='w')

        plt.show()

def rgb_to_hex(r, g, b):
    return 'https://csscolor.ru/?hex={:02x}{:02x}{:02x}'.format(r, g, b)

def color_identifier(data):
    x = 2 ** 8
    red = int(data[0] * x)
    green = int(data[1] * x)
    blue = int(data[2] * x)
    hexcolor = rgb_to_hex(red, green, blue)
    return hexcolor

som = SelfOrganizingMap()
np.random.seed(0)
data = np.random.rand(100, 3)

for i in range(len(data)):
    print(color_identifier(data[i]) + " Data number: " + str(i) + " Initial data: " + str(data[i]))

som.train(data, num_iterations=500)
som.plot_grid()

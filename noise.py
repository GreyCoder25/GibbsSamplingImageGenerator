import numpy as np


class Noiser:

    def simple_noise(self, image, num_colors, p):

        self.p = p
        self.p_spec_noise = self.p / (num_colors - 1)
        image_height, image_width = image.shape
        self.image = image.copy()

        for i in range(0, image_height):
            for j in range(0, image_width):
                curr_color = self.image[i, j]
                self.image[i, j] = np.random.choice([i for i in range(0, num_colors)], 1,
                                                    p=[self.p_spec_noise for i in range(0, curr_color)] + [1 - self.p] +
                                                      [self.p_spec_noise for i in range(curr_color + 1, num_colors)])[0]

        return self.image.copy()

    def p_x_cond_k(self, noise_type, x, k):

        if noise_type == 'simple':
            if x == k:
                return 1 - self.p
            return self.p_spec_noise

import numpy as np


class GibbsSamplingImageGenerator:

    def __init__(self, image_width=20, image_height=20, num_colors=3, num_iterations=100):

        self.image_width = image_width
        self.image_height = image_height
        self.num_colors = num_colors
        self.image = np.random.randint(0, self.num_colors, size=(self.image_width, self.image_height))
        self.g_horizontal = np.ones((self.num_colors, self.num_colors))
        self.g_vertical = np.ones((self.num_colors, self.num_colors))
        self.num_iterations = num_iterations
        self.current_iteration = 0

    def iteration_of_generation(self):
        RED, BLUE, GREEN = 0, 1, 2
        COLORS = (RED, BLUE, GREEN)

        if self.current_iteration % 2 == 0:
            for i_start, j_start in zip([0, 1], [0, 1]):
                for i in range(i_start, self.image_height, 2):
                    for j in range(j_start, self.image_width, 2):
                        p_colors = []
                        colors_interval = 0
                        for color in COLORS:
                            p_color = self._get_color_prob(i, j, color)
                            p_colors.append(p_color)
                            colors_interval += p_color
                        rand_point = np.random.rand() * colors_interval
                        # print(p_colors)
                        # print(rand_point, colors_interval)
                        if rand_point <= p_colors[0]:
                            # print(0)
                            self.image[i, j] = RED
                        elif rand_point > p_colors[0] and rand_point <= p_colors[0] + p_colors[1]:
                            # print(1)
                            self.image[i, j] = BLUE
                        else:
                            # print(2)
                            self.image[i, j] = GREEN
        elif self.current_iteration % 2 == 1:
            for i_start, j_start in zip([0, 1], [1, 0]):
                for i in range(i_start, self.image_height, 2):
                    for j in range(j_start, self.image_width, 2):
                        p_colors = []
                        colors_interval = 0
                        for color in COLORS:
                            p_color = self._get_color_prob(i, j, color)
                            p_colors.append(p_color)
                            colors_interval += p_color
                        rand_point = np.random.rand() * colors_interval
                        # print(p_colors)
                        # print(rand_point, colors_interval)
                        if rand_point <= p_colors[0]:
                            # print(0)
                            self.image[i, j] = RED
                        elif rand_point > p_colors[0] and rand_point <= p_colors[0] + p_colors[1]:
                            # print(1)
                            self.image[i, j] = BLUE
                        else:
                            # print(2)
                            self.image[i, j] = GREEN

        self.current_iteration += 1
        # print(self.image, "\n")

    def noise(self):
        print(self.image)
        # for beginning I've chosen the simplest noise algorithm

        RED, BLUE, GREEN = 0, 1, 2
        COLORS = (RED, BLUE, GREEN)
        for i in range(0, self.image_height):
            for j in range(0, self.image_width):
                if self.image[i, j] == RED:
                    self.image[i, j] = np.random.choice([RED, BLUE, GREEN], 1, p=[0.8, 0.1, 0.1])[0]
                elif self.image[i, j] == BLUE:
                    self.image[i, j] = np.random.choice([RED, BLUE, GREEN], 1, p=[0.1, 0.8, 0.1])[0]
                else:
                    self.image[i, j] = np.random.choice([RED, BLUE, GREEN], 1, p=[0.1, 0.1, 0.8])[0]
                # self.image[i, j] = np.random.choice([RED, BLUE, GREEN], 1)[0]

    def set_g(self, color1, color2, value, g_type):
        if g_type == "h":
            self.g_horizontal[color1, color2] = value
            # print('g_horizontal[%d, %d] = %f' % (color1, color2, value))
            # print('g_horizontal:\n', self.g_horizontal, "\n")
        elif g_type == "v":
            self.g_vertical[color1, color2] = value
            # print('g_vertical[%d, %d] = %f' % (color1, color2, value))
            # print('g_vertical:\n', self.g_vertical, "\n")

    def _get_color_prob(self, i, j, color):
        p = 1
        if self._check_coords(i, j - 1):
            p *= self.g_horizontal[self.image[i, j - 1], color]
        if self._check_coords(i, j + 1):
            p *= self.g_horizontal[color, self.image[i, j + 1]]
        if self._check_coords(i - 1, j):
            p *= self.g_vertical[self.image[i - 1, j], color]
        if self._check_coords(i + 1, j):
            p *= self.g_vertical[color, self.image[i + 1, j]]

        return p

    def set_num_iterations(self, num_iter):
        self.num_iterations = num_iter
        print('Number iterations of generation: ', self.num_iterations)

    def set_image_size(self, size):
        self.image_width = self.image_height = size
        print('Image size: ', self.image_width)
        self.image = np.random.randint(0, self.num_colors, size=(self.image_width, self.image_height))

    # def set_num_colors(self, num_colors):
    #     self.num_colors = num_colors


    def _check_coords(self, i, j):
        return (i >= 0 and i < self.image_height) and (j >= 0 and j < self.image_width)

    def execute_all_remaining(self):
        for i in range(self.current_iteration, self.num_iterations):
            self.iteration_of_generation()


class GibbsSamplingImageRecognizer:

    def __init__(self, sampler, num_iterations=1000):
        self.num_iterations = num_iterations
        self.image_width = sampler.image_width
        self.image_height = sampler.image_height
        self.g_horizontal = sampler.g_horizontal
        self.g_vertical = sampler.g_vertical
        self.num_colors = sampler.num_colors
        self.image = sampler.image.copy()
        self.current_iteration = 0

    def set_num_iterations(self, num_iter):
        self.num_iterations = num_iter
        print('Number iterations of recognition: ', self.num_iterations)

    def set_image_size(self, size):
        self.image_width = self.image_height = size
        # print('Image size: ', self.image_width)
        self.image = np.random.randint(0, self.num_colors, size=(self.image_width, self.image_height))

    def iteration_of_recognition(self):
        print(self.image)
        RED, BLUE, GREEN = 0, 1, 2
        COLORS = (RED, BLUE, GREEN)
        for i in range(0, self.image_height):
            for j in range(0, self.image_width):
                p_colors = []
                colors_interval = 0
                for color in COLORS:
                    if self.image[i, j] == color:
                        p_cond = 0.8
                    else:
                        p_cond = 0.1
                    p_color = self._get_color_prob(i, j, color) * p_cond   #the simplest possible noise is used
                    p_colors.append(p_color)
                    colors_interval += p_color
                rand_point = np.random.rand() * colors_interval
                if rand_point <= p_colors[0]:
                    self.image[i, j] = RED
                elif rand_point > p_colors[0] and rand_point <= p_colors[0] + p_colors[1]:
                    self.image[i, j] = BLUE
                else:
                    self.image[i, j] = GREEN

        self.current_iteration += 1

    def _get_color_prob(self, i, j ,color):
        return GibbsSamplingImageGenerator._get_color_prob(self, i, j, color)

    def _check_coords(self, i, j):
        return GibbsSamplingImageGenerator._check_coords(self, i ,j)

    def execute_all_remaining(self):
        for i in range(self.current_iteration, self.num_iterations):
            self.iteration_of_recognition()

import numpy as np


class GibbsSamplingImageGenerator:

    def __init__(self, image_width=20, image_height=20, num_colors=3, num_iterations=100):

        """Initializes image generator. Set's initial values of image(2-D array of integers),
        G-matrices(weight functions of different states of neighboring pixels), number of iterations of
        generation and different image parameters.

        Parameters
        ----------
        image_width : integer, optional (default=20)
            Width of image that will be generated.

        image_height : integer, optional (default=20)
            Height of image that will be generated.

        num_colors: integer, optional (default=3)
            Number of colors in image that will be generated.

        num_iterations : integer, optional (default=100)
            Specifies number of iterations of Gibbs sampler for image generating.
        """

        self.image_width = image_width
        self.image_height = image_height
        self.num_colors = num_colors
        self.image = np.random.randint(0, self.num_colors, size=(self.image_height, self.image_width))
        self.g_horizontal = np.ones((self.num_colors, self.num_colors))
        self.g_vertical = np.ones((self.num_colors, self.num_colors))
        self.num_iterations = num_iterations
        self.current_iteration = 0

    def iteration_of_generation(self):

        """Performs one iteration of Gibbs sampling of the image."""

        COLORS = range(self.num_colors)

        for i in range(0, self.image_height):
            for j in range(0, self.image_width):
                p_colors = []                               # list of probabilities of colors
                colors_interval = 0
                for color in COLORS:                        # Calculation of color's probability distribution
                    p_color = self._get_color_prob(i, j, color)
                    p_colors.append(p_color)
                    colors_interval += p_color
                rand_point = np.random.uniform(0, colors_interval)      # choosing a color from calculated distribution
                p_colors.append(0)              # for next cycle to work

                start_of_interval, end_of_interval = 0, p_colors[0]
                # color = 0
                for color in COLORS:
                    if start_of_interval <= rand_point <= end_of_interval:
                        self.image[i, j] = color
                        break
                    start_of_interval = end_of_interval
                    end_of_interval += p_colors[color + 1]

        self.current_iteration += 1
        # print(self.image, "\n")

    def reset(self):
        """Resets the generated image."""
        self.image = np.random.randint(0, self.num_colors, size=(self.image_height, self.image_width))
        self.current_iteration = 0

    def noise(self):                            # TODO rewrite
        # print(self.image)
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

        """Set's value of the element of the G matrix.

        Parameters
        ----------
        color1, color2 : integer
            Pair of colors that will have given value.

        value : real
            Weight of given pair.

        g_type : string('h' or 'v')
            Type of matrix which element need to be changed.
            'v' means vertical and 'g' means horizontal.
        """

        if g_type == "h":
            self.g_horizontal[color1, color2] = value
            # print('g_horizontal[%d, %d] = %f' % (color1, color2, value))
            # print('g_horizontal:\n', self.g_horizontal, "\n")
        elif g_type == "v":
            self.g_vertical[color1, color2] = value
            # print('g_vertical[%d, %d] = %f' % (color1, color2, value))
            # print('g_vertical:\n', self.g_vertical, "\n")

    def get_g(self, color1, color2, g_type):

        if g_type == "h":
            return self.g_horizontal[color1, color2]
        elif g_type == "v":
            return self.g_vertical[color1, color2]

    def _get_color_prob(self, i, j, color):

        """Estimates the probability of given color to be real color of the pixel with given coordinates.

        Parameters
        ----------
        i, j: integer
            Coordinates of pixel.
        color:
            Color value that need's to be checked.

        Returns
        -------
        Probability of pixel with coordinates [i, j] to
        have given color.

        """

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

    def set_num_colors(self, num_colors):               # TODO possibly can be optimized
        self.num_colors = num_colors
        print("Number of colors:", self.num_colors)
        self.image = np.random.randint(0, self.num_colors, size=(self.image_width, self.image_height))
        self.g_horizontal = np.ones((self.num_colors, self.num_colors))
        self.g_vertical = np.ones((self.num_colors, self.num_colors))
        print(self.g_horizontal, self.g_vertical)

    def _check_coords(self, i, j):
        return (0 <= i < self.image_height) and (0 <= j < self.image_width)

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

    def iteration_of_recognition(self):                     # TODO rewrite
        # print(self.image)
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

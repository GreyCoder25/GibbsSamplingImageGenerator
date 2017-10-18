import numpy as np
import noise


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

        print("Generation iteration %d started" % self.current_iteration)
        COLORS = range(self.num_colors)

        for i in range(0, self.image_height):
            for j in range(0, self.image_width):
                p_colors = []                               # list of probabilities of colors
                colors_interval = 0
                for color in COLORS:                        # Calculation of color's probability distribution
                    p_color = self.get_color_prob(i, j, color)
                    p_colors.append(p_color)
                    colors_interval += p_color
                rand_point = np.random.uniform(0, colors_interval)      # choosing a color from calculated distribution
                p_colors.append(0)              # for next cycle to work

                start_of_interval, end_of_interval = 0, p_colors[0]

                for color in COLORS:
                    if start_of_interval <= rand_point <= end_of_interval:
                        self.image[i, j] = color
                        break
                    start_of_interval = end_of_interval
                    end_of_interval += p_colors[color + 1]

        print("Generation iteration %d finished" % self.current_iteration)

        self.current_iteration += 1

    def reset(self):

        """Resets the generated image by generating new random image."""
        self.image = np.random.randint(0, self.num_colors, size=(self.image_height, self.image_width))
        self.current_iteration = 0
    #
    # def noise(self, noiser):
    #
    #     return noiser.simple_noise(self.image, self.num_colors, 0.2)

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
        elif g_type == "v":
            self.g_vertical[color1, color2] = value

    def get_g(self, color1, color2, g_type):

        if g_type == "h":
            return self.g_horizontal[color1, color2]
        elif g_type == "v":
            return self.g_vertical[color1, color2]

    def get_color_prob(self, i, j, color):

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
        if self.check_coords(i, j - 1):
            p *= self.g_horizontal[self.image[i, j - 1], color]
        if self.check_coords(i, j + 1):
            p *= self.g_horizontal[color, self.image[i, j + 1]]
        if self.check_coords(i - 1, j):
            p *= self.g_vertical[self.image[i - 1, j], color]
        if self.check_coords(i + 1, j):
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

        if 1 < num_colors <= 8:
            self.num_colors = num_colors
            print("Number of colors:", self.num_colors)
            self.image = np.random.randint(0, self.num_colors, size=(self.image_width, self.image_height))
            self.g_horizontal = np.ones((self.num_colors, self.num_colors))
            self.g_vertical = np.ones((self.num_colors, self.num_colors))
            # print(self.g_horizontal, self.g_vertical)

    def check_coords(self, i, j):

        return (0 <= i < self.image_height) and (0 <= j < self.image_width)

    def execute_all_remaining(self):

        for i in range(self.current_iteration, self.num_iterations):
            self.iteration_of_generation()


class GibbsSamplingImageRecognizer:

    def __init__(self, sampler, noiser, num_iterations=1000):

        self.num_iterations = num_iterations
        self.image_width = sampler.image_width
        self.image_height = sampler.image_height
        self.g_horizontal = sampler.g_horizontal
        self.g_vertical = sampler.g_vertical
        self.num_colors = sampler.num_colors
        self.image = np.empty_like(sampler.image)
        self.initial_image = self.image.copy()
        self.current_iteration = 0
        self.noiser = noiser

    def set_image(self, image):

        self.image = image.copy()
        self.initial_image = image.copy()

    def set_num_iterations(self, num_iter):

        self.num_iterations = num_iter
        print('Number iterations of recognition: ', self.num_iterations)

    def set_image_size(self, size):

        self.image_width = self.image_height = size
        self.set_image(np.random.randint(0, self.num_colors, size=(self.image_width, self.image_height)))

    def iteration_of_recognition(self):

        COLORS = range(self.num_colors)

        for i in range(0, self.image_height):
            for j in range(0, self.image_width):
                p_colors = []                               # list of probabilities of colors
                colors_interval = 0
                curr_color = self.image[i, j]
                for color in COLORS:                        # Calculation of color's probability distribution
                    p_color = self.get_color_prob(i, j, color) * self.noiser.p_x_cond_k('simple', curr_color, color)
                    p_colors.append(p_color)
                    colors_interval += p_color
                rand_point = np.random.uniform(0, colors_interval)  # choosing a color from calculated distribution
                p_colors.append(0)                          # for next cycle to work

                start_of_interval, end_of_interval = 0, p_colors[0]

                for color in COLORS:
                    if start_of_interval <= rand_point <= end_of_interval:
                        self.image[i, j] = color
                        break
                    start_of_interval = end_of_interval
                    end_of_interval += p_colors[color + 1]

        self.current_iteration += 1

    def iteration_of_line_recognition(self):

        def q1(i, j, k):

            res = self.noiser.p_x_cond_k('simple', self.image[i, j], k)
            if self.check_coords(i - 1, j):
                res *= self.g_vertical[self.image[i - 1, j], k]
            if self.check_coords(i + 1, j):
                res *= self.g_vertical[k, self.image[i + 1, j]]

            return res

        def p_k1_k2(i, j1, j2, k1, k2, f_left, f_right, q1):

            return f_left[k1, j1] * q1(i, j1, k1) * self.g_horizontal[k1, k2] * q1(i, j2, k2) * f_right[k2, j2]
            # return f_left[k1, j1] * self.g_horizontal[k1, k2] * f_right[k2, j2]

        def generate_k0(row, f_left, f_right, q1):

            P = np.empty((self.num_colors, self.num_colors))
            for i in range(self.num_colors):
                for j in range(self.num_colors):
                    P[i, j] = p_k1_k2(row, 0, 1, i, j, f_left, f_right, q1)

            P = P.sum(axis=1)
            rand_point = np.random.uniform(0, P.sum())
            interval_begin, interval_end = 0, P[0]
            for i in range(self.num_colors):
                if interval_begin <= rand_point <= interval_end:
                    return i
                interval_begin += P[i]
                interval_end += P[i+1]

        # print(np.vectorize(q1)(0, 1, np.arange(self.num_colors)))
        # function body begins here
        for i in range(self.image_height):

            f_left = np.ones((self.num_colors, self.image_width))
            for j in range(2, self.image_width):
                f_left[:, j] = np.dot(f_left[:, j - 1], self.g_horizontal *
                                      np.vectorize(q1)(i, j-1, np.arange(self.num_colors)).reshape(self.num_colors, 1))
                f_left[:, j] = f_left[:, j] / 300

            f_right = np.ones((self.num_colors, self.image_width))
            for j in range(self.image_width - 2, -1, -1):
                f_right[:, j] = np.dot(self.g_horizontal * np.vectorize(q1)(i, j+1, np.arange(self.num_colors)),
                                       f_right[:, j + 1])
                f_right[:, j] = f_right[:, j] / 300

            self.image[i, 0] = generate_k0(i, f_left, f_right, q1)

            for j in range(1, self.image_width):
                p_labels = []
                prev_label = self.image[i, j-1]
                for label in range(self.num_colors):
                    p_label = p_k1_k2(i, j-1, j, prev_label, label, f_left, f_right, q1)
                    p_labels.append(p_label)

                p_prev_label = sum(p_labels)
                for label_index in range(len(p_labels)):
                    p_labels[label_index] /= p_prev_label

                rand_point = np.random.uniform(0, sum(p_labels))
                interval_begin, interval_end = 0, p_labels[0]
                for label in range(self.num_colors):
                    if interval_begin <= rand_point <= interval_end:
                        self.image[i, j] = label
                        break
                    interval_begin += p_labels[label]
                    interval_end += p_labels[label + 1]

        self.current_iteration += 1

    def get_color_prob(self, i, j, color):

        return GibbsSamplingImageGenerator.get_color_prob(self, i, j, color)

    def check_coords(self, i, j):

        return GibbsSamplingImageGenerator.check_coords(self, i, j)

    def reset(self):

        self.image = self.initial_image.copy()
        self.current_iteration = 0

    def execute_all_remaining(self, rec_type):

        if rec_type == 'pixelwise':
            iteration = self.iteration_of_recognition
        elif rec_type == 'line':
            iteration = self.iteration_of_line_recognition

        for i in range(self.current_iteration, self.num_iterations):
            iteration()

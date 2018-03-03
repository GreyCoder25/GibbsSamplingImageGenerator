import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc
import noise
# import time

DTYPE = np.int
GTYPE = np.float

ctypedef np.int_t DTYPE_t
ctypedef np.float_t GTYPE_t

cdef extern from "time.h":
    long int time(int)

cdef extern from "np_mtwister.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos

    double rk_double(rk_state *state)
    void rk_seed(unsigned long seed, rk_state *state)

cdef rk_state *internal_state = <rk_state*>malloc(sizeof(rk_state))
rk_seed(time(0), internal_state)


cdef bint check_coords(int i, int j, int h, int w):

    return (0 <= i < h) and (0 <= j < w)


class GibbsSamplingImageGenerator:

    def __init__(self, image_width=100, image_height=100, num_colors=3, num_iterations=100):

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
        self.num_iterations = num_iterations
        self.current_iteration = 1
        self.image = np.random.randint(0, self.num_colors, size=(self.image_height, self.image_width), dtype=DTYPE)
        self.g_horizontal = np.ones((self.num_colors, self.num_colors), dtype=GTYPE)
        self.g_vertical = np.ones((self.num_colors, self.num_colors), dtype=GTYPE)

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
            p *= self.g_horizontal[color,
                                   self.image[i,
                                              j + 1]]
        if self.check_coords(i - 1, j):
            p *= self.g_vertical[self.image[i - 1, j], color]
        if self.check_coords(i + 1, j):
            p *= self.g_vertical[color, self.image[i + 1, j]]

        return p

    def check_coords(self, i, j):

        return (0 <= i < self.image_height) and (0 <= j < self.image_width)

    def iteration_of_generation(self, num_iters=1):
        self._iteration_of_generation(self.g_vertical, self.g_horizontal, self.image, num_iters)

    def _iteration_of_generation(self, np.ndarray[GTYPE_t, ndim=2] g_v, np.ndarray[GTYPE_t, ndim=2] g_h,
                                 np.ndarray[DTYPE_t, ndim=2] im, n):

        """Performs one iteration of Gibbs sampling of the image."""
        cdef int i, j, color
        p_colors_ndarray = np.ones(self.num_colors + 1, dtype=np.float)
        cdef double [:] p_colors = p_colors_ndarray                           # list of probabilities of colors
        cdef double colors_interval, p_color, rand_point
        cdef double start_of_interval, end_of_interval

        cdef int h = self.image_height
        cdef int w = self.image_width
        cdef int num_col = self.num_colors
        cdef int num_iters = n
        cdef int curr_iter = self.current_iteration

        for _ in range(0, num_iters):
            for i in range(0, h):
                for j in range(0, w):
                    colors_interval = 0
                    for color in range(0, num_col):                 # Calculation of color's probability distribution
                        p_color = 1.0
                        if check_coords(i, j - 1, h, w):
                            p_color *= g_h[im[i, j - 1], color]
                        if check_coords(i, j + 1, h, w):
                            p_color *= g_h[color, im[i, j + 1]]
                        if check_coords(i - 1, j, h, w):
                            p_color *= g_v[im[i - 1, j], color]
                        if check_coords(i + 1, j, h, w):
                            p_color *= g_v[color, im[i + 1, j]]
                        p_colors[color] = p_color
                        colors_interval += p_color

                    rand_point = rk_double(internal_state) * colors_interval
                    # rand_point = np.random.uniform(0, colors_interval)
                    p_colors[num_col] = 0              # for next cycle to work

                    start_of_interval = 0
                    end_of_interval = p_colors[0]

                    for color in range(0, num_col):
                        if start_of_interval <= rand_point <= end_of_interval:
                            im[i, j] = color
                            break
                        start_of_interval = end_of_interval
                        end_of_interval += p_colors[color + 1]

            curr_iter += 1
        self.current_iteration = curr_iter

    def reset(self):

        """Resets the generated image by generating new random image."""
        # np.random.seed(20)
        self.image = np.random.randint(0, self.num_colors, size=(self.image_height, self.image_width))
        self.current_iteration = 1
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

    def set_num_iterations(self, num_iter):

        self.num_iterations = num_iter
        print('Number iterations of generation: ', self.num_iterations)

    def set_image_height(self, height):

        self.image_height = height
        self.image = np.random.randint(0, self.num_colors, size=(self.image_height, self.image_width))

    def set_image_width(self, width):

        self.image_width = width
        self.image = np.random.randint(0, self.num_colors, size=(self.image_height, self.image_width))

    def set_num_colors(self, num_colors):               # TODO possibly can be optimized

        if 1 < num_colors <= 8:
            self.num_colors = num_colors
            print("Number of colors:", self.num_colors)
            self.image = np.random.randint(0, self.num_colors, size=(self.image_width, self.image_height))
            self.g_horizontal = np.ones((self.num_colors, self.num_colors))
            self.g_vertical = np.ones((self.num_colors, self.num_colors))
            # print(self.g_horizontal, self.g_vertical)

    def execute_all_remaining(self):

        self.iteration_of_generation(self.num_iterations - self.current_iteration)


cdef double p_x_cond_k(int x, int k, double p, double p_spec_noise):

     if x == k:
         return 1 -p
     return p_spec_noise

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
        self.color_counters = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=DTYPE)
        self.mean_prob = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=GTYPE)
        self.current_iteration = 1
        self.noiser = noiser

    def set_image(self, image):

        self.image = image.copy()
        self.initial_image = image.copy()

    def set_num_colors(self, num_colors):

        if 1 <= num_colors <= 8:
            self.num_colors = num_colors
            self.g_horizontal = np.ones((self.num_colors, self.num_colors))
            self.g_vertical = np.ones((self.num_colors, self.num_colors))

    def set_num_iterations(self, num_iter):

        self.num_iterations = num_iter
        print('Number iterations of recognition: ', self.num_iterations)

    def set_image_height(self, height):

        GibbsSamplingImageGenerator.set_image_height(self, height)
        self.initial_image = self.image.copy()
        self.color_counters = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=DTYPE)
        self.mean_prob = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=GTYPE)

    def set_image_width(self, width):

        GibbsSamplingImageGenerator.set_image_width(self, width)
        self.initial_image = self.image.copy()
        self.color_counters = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=DTYPE)
        self.mean_prob = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=GTYPE)

    def iteration_of_recognition(self, num_iters=1):
        self._iteration_of_recognition(self.g_vertical, self.g_horizontal, self.initial_image, self.image, num_iters)

    def _iteration_of_recognition(self, np.ndarray[GTYPE_t, ndim=2] g_v, np.ndarray[GTYPE_t, ndim=2] g_h,
                                 np.ndarray[DTYPE_t, ndim=2] init_im, np.ndarray[DTYPE_t, ndim=2] im, n):

        cdef int i, j, color, curr_color
        p_colors_ndarray = np.ones(self.num_colors + 1, dtype=np.float)
        cdef double [:] p_colors = p_colors_ndarray                           # list of probabilities of colors
        cdef double colors_interval, p_color, rand_point
        cdef double start_of_interval, end_of_interval

        cdef int h = self.image_height
        cdef int w = self.image_width
        cdef int num_col = self.num_colors
        cdef int num_iters = n
        cdef int curr_iter = self.current_iteration
        cdef int traverse_type[2][2]
        traverse_type[0][0] = 0; traverse_type[0][1] = 1; traverse_type[1][0] = 1; traverse_type[1][1] = 0

        cdef np.ndarray[GTYPE_t, ndim=3] mean_prob = self.mean_prob
        cdef np.ndarray[DTYPE_t, ndim=3] color_counters = self.color_counters

        cdef double noiser_p = self.noiser.p
        cdef double noiser_p_spec_noise = self.noiser.p_spec_noise

        for _ in range(0, num_iters):
            for traverse_type_index in range(0, 2):
                for i in range(0, h):
                    for j in range(traverse_type[traverse_type_index][i % 2], w, 2):
                        colors_interval = 0
                        curr_color = init_im[i, j]
                        for color in range(0, num_col):                        # Calculation of color's probability distribution
                            p_color = p_x_cond_k(curr_color, color, noiser_p, noiser_p_spec_noise)
                            if check_coords(i, j - 1, h, w):
                                p_color *= g_h[im[i, j - 1], color]
                            if check_coords(i, j + 1, h, w):
                                p_color *= g_h[color, im[i, j + 1]]
                            if check_coords(i - 1, j, h, w):
                                p_color *= g_v[im[i - 1, j], color]
                            if check_coords(i + 1, j, h, w):
                                p_color *= g_v[color, im[i + 1, j]]
                            mean_prob[color, i, j] = (mean_prob[color, i, j] * (curr_iter - 1) + p_color) / curr_iter
                            p_colors[color] = p_color
                            colors_interval += p_color
                        # rand_point = np.random.uniform(0, colors_interval)  # choosing a color from calculated distribution
                        rand_point = rk_double(internal_state) * colors_interval
                        p_colors[num_col] = 0                          # for next cycle to work

                        start_of_interval, end_of_interval = 0, p_colors[0]

                        for color in range(0, num_col):
                            if start_of_interval <= rand_point <= end_of_interval:
                                im[i, j] = color
                                break
                            start_of_interval = end_of_interval
                            end_of_interval += p_colors[color + 1]

            curr_iter += 1
            self.update_color_counters()
        self.current_iteration = curr_iter

    def iteration_of_line_recognition(self, num_iters=1):
        self._iteration_of_line_recognition(self.g_vertical, self.g_horizontal, self.initial_image, self.image, num_iters)

    def _iteration_of_line_recognition(self, np.ndarray[GTYPE_t, ndim=2] g_v, np.ndarray[GTYPE_t, ndim=2] g_h,
                                 np.ndarray[DTYPE_t, ndim=2] init_im, np.ndarray[DTYPE_t, ndim=2] im, n):

        cdef double noiser_p = self.noiser.p
        cdef double noiser_p_spec_noise = self.noiser.p_spec_noise

        cdef int i, j, k, k_, start_row, label, prev_label, curr_iter = self.current_iteration
        cdef int h = self.image_height
        cdef int w = self.image_width
        cdef int num_col = self.num_colors
        cdef int num_iters = n
        cdef double f_sum = 0, l_sum = 0, P_sum = 0, q1_val
        p_labels_ndarray = np.ones(num_col, dtype=np.float)
        cdef double [:] p_labels = p_labels_ndarray
        cdef double interval_begin, interval_end, p_label, rand_point

        cdef np.ndarray[GTYPE_t, ndim=2] f_left = np.ones((num_col, w), dtype=GTYPE)
        cdef np.ndarray[GTYPE_t, ndim=2] f_right = np.ones((num_col, w), dtype=GTYPE)

        cdef np.ndarray[GTYPE_t, ndim=2] P = np.ones((num_col, num_col), dtype=GTYPE)
        cdef np.ndarray[GTYPE_t, ndim=1] P_sums = np.zeros(num_col, dtype=GTYPE)
        cdef np.ndarray[GTYPE_t, ndim=3] mean_prob = self.mean_prob

        # function body begins here
        for _ in range(0, num_iters):
            for start_row in range(0, 2):
                for i in range(start_row, h, 2):
                    for j in range(2, w):
                        f_sum = 0
                        for k in range(0, num_col):
                            f_left[k, j] = 0
                            for k_ in range(0, num_col):
                                q1_val = p_x_cond_k(init_im[i, j - 1], k_, noiser_p, noiser_p_spec_noise)
                                if check_coords(i - 1, j - 1, h, w):
                                    q1_val *= g_v[im[i - 1, j - 1], k_]
                                if check_coords(i + 1, j - 1, h, w):
                                    q1_val *= g_v[k_, im[i + 1, j - 1]]
                                f_left[k, j] += f_left[k_, j-1] * g_h[k_, k] * q1_val
                            f_sum += f_left[k, j]

                        for k in range(0, num_col):
                            f_left[k, j] = f_left[k, j] / f_sum

                    for j in range(w - 2, -1, -1):
                        f_sum = 0
                        for k in range(0, num_col):
                            f_right[k, j] = 0
                            for k_ in range(0, num_col):
                                q1_val = p_x_cond_k(init_im[i, j + 1], k_, noiser_p, noiser_p_spec_noise)
                                if check_coords(i - 1, j + 1, h, w):
                                    q1_val *= g_v[im[i - 1, j + 1], k_]
                                if check_coords(i + 1, j + 1, h, w):
                                    q1_val *= g_v[k_, im[i + 1, j + 1]]
                                f_right[k, j] += f_right[k_, j+1] * g_h[k, k_] * q1_val
                            f_sum += f_right[k, j]
                        for k in range(0, num_col):
                            f_right[k, j] = f_right[k, j] / f_sum

                    for k in range(0, num_col):
                        for k_ in range(0, num_col):
                            P[k, k_] = (f_left[k, 0] * g_h[k, k_] * f_right[k_, 1] *
                                        p_x_cond_k(init_im[i, 0], k, noiser_p, noiser_p_spec_noise) *
                                        p_x_cond_k(init_im[i, 1], k_, noiser_p, noiser_p_spec_noise))
                            if check_coords(i - 1, 0, h, w):
                                P[k, k_] *= g_v[im[i - 1, 0], k]
                            if check_coords(i + 1, 0, h, w):
                                P[k, k_] *= g_v[k, im[i + 1, 0]]
                            if check_coords(i - 1, 1, h, w):
                                P[k, k_] *= g_v[im[i - 1, 1], k_]
                            if check_coords(i + 1, 1, h, w):
                                P[k, k_] *= g_v[k_, im[i + 1, 1]]

                    P_sum = 0
                    for k in range(0, num_col):
                        P_sums[k] = 0
                        for k_ in range(0, num_col):
                            P_sums[k] += P[k, k_]
                        P_sum += P_sums[k]
                    # rand_point = np.random.uniform(0, P_sum)
                    rand_point = rk_double(internal_state) * P_sum
                    interval_begin, interval_end = 0, P_sums[0]

                    for label in range(0, num_col):
                        if interval_begin <= rand_point <= interval_end:
                            im[i, 0] = label
                            break
                        interval_begin += P_sums[label]
                        interval_end += P_sums[label+1]

                    for label in range(0, num_col):
                        q1_val = p_x_cond_k(init_im[i, 0], label, noiser_p, noiser_p_spec_noise)
                        if check_coords(i - 1, 0, h, w):
                            q1_val *= g_v[im[i - 1, 0], label]
                        if check_coords(i + 1, 0, h, w):
                            q1_val *= g_v[label, im[i + 1, 0]]
                        p_k_val = f_left[label, 0] * q1_val * f_right[label, 0]
                        # self.update_mean_prob(label, i, 0, self.current_iteration, p_k_val)
                        mean_prob[label, i, 0] = (mean_prob[label, i, 0] * (curr_iter - 1) + p_k_val) / curr_iter

                    for j in range(1, w):
                        prev_label = im[i, j-1]
                        l_sum = 0
                        for label in range(0, num_col):
                            p_label = (f_left[prev_label, j-1] * g_h[prev_label, label] * f_right[label, j] *
                                        p_x_cond_k(init_im[i, j-1], prev_label, noiser_p, noiser_p_spec_noise) *
                                        p_x_cond_k(init_im[i, j], label, noiser_p, noiser_p_spec_noise))
                            if check_coords(i - 1, j - 1, h, w):
                                p_label *= g_v[im[i - 1, j - 1], prev_label]
                            if check_coords(i + 1, j - 1, h, w):
                                p_label *= g_v[prev_label, im[i + 1, j - 1]]
                            if check_coords(i - 1, j, h, w):
                                p_label *= g_v[im[i - 1, j], label]
                            if check_coords(i + 1, j, h, w):
                                p_label *= g_v[label, im[i + 1, j]]
                            l_sum += p_label
                            p_labels[label] = p_label

                        # p_prev_label = np.sum(p_labels)
                        # for label_index in range(len(p_labels)):
                        #     if p_prev_label > 0:
                        #         p_labels[label_index] /= p_prev_label

                        # rand_point = np.random.uniform(0, l_sum)
                        rand_point = rk_double(internal_state) * l_sum
                        interval_begin, interval_end = 0, p_labels[0]
                        for label in range(0, num_col):
                            if interval_begin <= rand_point <= interval_end:
                                im[i, j] = label
                                break
                            interval_begin += p_labels[label]
                            interval_end += p_labels[label + 1]

                        # updating posteriors of labels in current pixel
                        for label in range(num_col):
                            q1_val = p_x_cond_k(init_im[i, j], label, noiser_p, noiser_p_spec_noise)
                            if check_coords(i - 1, j, h, w):
                                q1_val *= g_v[im[i - 1, j], label]
                            if check_coords(i + 1, j, h, w):
                                q1_val *= g_v[label, im[i + 1, j]]
                            p_k_val = f_left[label, j] * q1_val * f_right[label, j]
                            # self.update_mean_prob(label, i, j, self.current_iteration, p_k_val)
                            mean_prob[label, i, j] = (mean_prob[label, i, j] * (curr_iter - 1) + p_k_val) / curr_iter

            curr_iter += 1
            self.update_color_counters()
        self.current_iteration = curr_iter

    def get_color_prob(self, i, j, color):

        return GibbsSamplingImageGenerator.get_color_prob(self, i, j, color)

    def check_coords(self, i, j):

        return GibbsSamplingImageGenerator.check_coords(self, i, j)

    def update_color_counters(self):

        for color in range(self.num_colors):
            self.color_counters[color] += (self.image == color)

    def update_mean_prob(self, label, i, j, iter_number, p_curr):

        self.mean_prob[label, i, j] = (self.mean_prob[label, i, j] * (iter_number - 1) + p_curr) / iter_number

    def get_max_freq_image(self):

        return np.argmax(self.color_counters, axis=0)

    def get_max_prob_image(self):

        return np.argmax(self.mean_prob, axis=0)

    def reset(self):

        self.image = self.initial_image.copy()
        self.color_counters = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=DTYPE)
        self.mean_prob = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=GTYPE)
        self.current_iteration = 1

    def execute_all_remaining(self, rec_type):

        if rec_type == 'pixelwise':
            iteration = self.iteration_of_recognition
        elif rec_type == 'line':
            iteration = self.iteration_of_line_recognition

        # for i in range(self.current_iteration, self.num_iterations):
        iteration(self.num_iterations - self.current_iteration)

    def set_g(self, color1, color2, value, g_type):

        GibbsSamplingImageGenerator.set_g(self, color1, color2, value, g_type)
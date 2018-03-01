import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import model
import noise
import tkinter as tk
import pickle
import time

import numpy as np

mpl.use("TkAgg")


class ImageGeneratorApp(tk.Tk):

    """
    Main class of the image generating application.
    """

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side="top", expand=True, fill="both")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}                                    # dictionary of pages in graphical interface

        try:
            with open('settings.pickle', 'rb') as f:
                self.sampler = pickle.load(f)
                self.noiser = pickle.load(f)
                self.recognizer = pickle.load(f)
        except FileNotFoundError:
            self.sampler = model.GibbsSamplingImageGenerator()
            self.noiser = noise.Noiser()
            self.recognizer = model.GibbsSamplingImageRecognizer(self.sampler, self.noiser)

        for F in (MainPage, SettingsPage, GraphPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainPage)

    def show_frame(self, controller):

        frame = self.frames[controller]
        frame.tkraise()

    def save(self):

        with open('settings.pickle', 'wb') as f:
            for object_ in [self.sampler, self.noiser, self.recognizer]:
                pickle.dump(object_, f)


class MainPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.num_rows = 3
        self.num_columns = 6
        for r in range(self.num_rows):
            self.grid_rowconfigure(r, weight=1)
            for c in range(self.num_columns):
                self.grid_columnconfigure(c, weight=1)

        self.graph_names = ['Generated image', 'Noisy image', 'Recognized image']
        self.buttons_names = ['Next iteration', 'Execute all remaining', 'Noise']
        c = 0

        settings_button = tk.Button(self, text="Settings", command=lambda: controller.show_frame(SettingsPage))
        settings_button.grid(row=0, column=self.num_columns-2)

        quit_button = tk.Button(self, text="Quit", command=self.quit)
        quit_button.grid(row=0, column=self.num_columns-1)

        reset_button = tk.Button(self, text="Reset generation", command=self.reset_generation)
        reset_button.grid(row=0, column=self.num_columns - 6)

        reset_button = tk.Button(self, text="Reset recognition", command=self.reset_recognition)
        reset_button.grid(row=0, column=self.num_columns - 3)

        show_graph_button = tk.Button(self, text="Test and show graph", command=self.test_and_show_graph)
        show_graph_button.grid(row=0, column=self.num_columns - 5)

        graph_button = tk.Button(self, text="Graph", command=self.show_graph_page)
        graph_button.grid(row=0, column=self.num_columns - 4)

        # buttons for generated image graph
        tk.Button(self, text="Next iteration", command=self.next_generating_iteration).grid(row=2,
                                                                                            column=0, sticky="nsew")
        tk.Button(self, text="Execute all remaining", command=self.execute_all_gen_remaining).grid(row=2,
                                                                                                column=1,sticky="nsew")

        # Noise button
        tk.Button(self, text="Noise", command=self.show_noisy_image).grid(row=2, column=2, sticky="nsew")

        # buttons for recognized image graph
        tk.Button(self, text="Next pixelwise iteration", command=self.next_pixelwise_recognition_iteration).grid(row=2, column=3,
                                                                                             sticky="nsew")

        tk.Button(self, text="Next line iteration", command=self.next_line_recognition_iteration).grid(row=2,
                                                                                                   column=4, sticky="nsew")
        # show image graphs
        self.figure = self.image_graphs_init()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=1, columnspan=self.num_columns)

    def image_graphs_init(self):

        IMAGE_WIDTH = self.controller.sampler.image_width
        IMAGE_HEIGHT = self.controller.sampler.image_height

        fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=3)

        for ax, graph_name in zip(axes, self.graph_names):
            ax.set_ylim(IMAGE_HEIGHT, 0)
            ax.set_xlim(0, IMAGE_WIDTH)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(graph_name)
            ax.xaxis.set_ticks_position('top')

        self.image_init(axes[0])
        return fig

    def image_init(self, ax, pix_vals_type='gen'):

        COLORS = ('red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'white')

        if pix_vals_type == 'gen':
            pix_vals = self.controller.sampler.image
        elif pix_vals_type == 'rec':
            pix_vals = self.controller.recognizer.image
        elif pix_vals_type == 'max_freq':
            pix_vals = self.controller.recognizer.get_max_freq_image()
        elif pix_vals_type == 'max_prob':
            pix_vals = self.controller.recognizer.get_max_prob_image()
            # print(pix_vals)
            # print(self.controller.recognizer.mean_prob)
        cmap = mpl.colors.ListedColormap(COLORS[:self.controller.sampler.num_colors])
        ax.pcolormesh(pix_vals, cmap=cmap)

    def reset_generation(self):

        self.controller.sampler.reset()
        self.update_generated_image()

    def reset_recognition(self):

        self.controller.recognizer.reset()
        self.update_recognized_image()

    def change_image_size(self, size, param):

        for ax in self.canvas.figure.axes:
            if param == 'h':
                ax.set_ylim(size, 0)
            elif param == 'w':
                ax.set_xlim(0, size)

        self.image_graphs_init()
        self.image_init(self.canvas.figure.axes[0])
        self.canvas.draw()

    def update_generated_image(self):

        self.image_init(self.canvas.figure.axes[0])
        self.canvas.draw()

    def update_recognized_image(self):

        self.image_init(self.canvas.figure.axes[2], 'max_prob')
        self.canvas.draw()

    def show_noisy_image(self):

        self.controller.recognizer.set_image(self.controller.noiser.simple_noise(self.controller.sampler.image,
                                                                                 self.controller.recognizer.num_colors, 0.4))

        self.image_init(self.canvas.figure.axes[1], 'rec')
        self.image_init(self.canvas.figure.axes[2], 'rec')
        self.canvas.draw()

    def next_generating_iteration(self):

        start = time.time()
        self.controller.sampler.iteration_of_generation()
        finish = time.time()
        print("Time: %f" % (finish - start))
        self.update_generated_image()

    def next_pixelwise_recognition_iteration(self):
        # for i in range(20):
        #     for j in range(1):
        self.controller.recognizer.iteration_of_recognition()
        self.update_recognized_image()

    def next_line_recognition_iteration(self):

        # for i in range(1000):
        # for i in range(20):
        #     for j in range(1):
        self.controller.recognizer.iteration_of_line_recognition()
        self.update_recognized_image()

    def execute_all_gen_remaining(self):

        start = time.time()
        self.controller.sampler.execute_all_remaining()
        finish = time.time()
        print("Generation time: %f" % (finish - start))
        self.update_generated_image()

    def test_and_show_graph(self):

        num_iterations = self.controller.recognizer.num_iterations
        recognizer = self.controller.recognizer
        sampler = self.controller.sampler
        noiser = self.controller.noiser
        noise_level = 0.6
        # save_file = 'monotonous5_2labels.pickle'
        number_of_experiments = 1

        for _ in range(number_of_experiments):
            # sampler.reset()
            # recognizer.reset()

            # for i in range(sampler.num_iterations):
            #     sampler.iteration_of_generation()

            # recognizer.set_image(noiser.simple_noise(sampler.image, recognizer.num_colors, noise_level))

            pixelwise_max_freq = np.empty(9*num_iterations + 1)
            pixelwise_aver_prob = np.empty(9*num_iterations + 1)
            pixelwise_max_freq[0] = (sampler.image != recognizer.image).sum()
            pixelwise_aver_prob[0] = (sampler.image != recognizer.image).sum()
            line_max_freq = np.empty(num_iterations + 1)
            line_aver_prob = np.empty(num_iterations + 1)
            line_max_freq[0] = pixelwise_max_freq[0]
            line_aver_prob[0] = pixelwise_aver_prob[0]

            pixelwise_total_time = 0
            line_total_time = 0

            for i in range(1, 9*num_iterations + 1):
                start = time.time()
                recognizer.iteration_of_recognition()
                finish = time.time()
                print('Time of pixelwise max_freq perfoming iteration %d: %f' % (i, finish - start))
                pixelwise_total_time += finish - start
                pixelwise_max_freq[i] = (sampler.image != recognizer.get_max_freq_image()).sum()
                pixelwise_aver_prob[i] = (sampler.image != recognizer.get_max_prob_image()).sum()
            recognizer.reset()

            pixelwise_average_time = pixelwise_total_time / (9*num_iterations)
            print("Pixelwise max_freq average time: %f" % pixelwise_average_time)

            for i in range(1, num_iterations + 1):
                start = time.time()
                recognizer.iteration_of_line_recognition()
                finish = time.time()
                print('Time of line perfoming iteration %d: %f' % (i, finish - start))
                line_total_time += finish - start
                line_max_freq[i] = (sampler.image != recognizer.get_max_freq_image()).sum()
                line_aver_prob[i] = (sampler.image != recognizer.get_max_prob_image()).sum()
            recognizer.reset()

            # with open(save_file, 'ab') as f:
            #     for data_array in (pixelwise_max_freq, line_max_freq, pixelwise_aver_prob, line_aver_prob):
            #         pickle.dump(data_array, f)

        line_average_time = line_total_time / num_iterations
        print("Line average time: %f" % line_average_time)

        # average_times = (pixelwise_average_time, line_average_time, pixelwise_average_time, line_average_time)

        # graph_page = self.controller.frames[GraphPage]
        # graph_page.get_data(average_times, pixelwise_max_freq, line_max_freq, pixelwise_aver_prob, line_aver_prob)
        # self.show_graph_page()

    def show_graph_page(self):

        self.controller.show_frame(GraphPage)

    # def execute_all_rec_remaining(self):
    #
    #     self.controller.recognizer.execute_all_remaining()
    #     self.update_recognized_image()

    def quit(self):

        self.controller.save()
        tk.Frame.quit(self)


class SettingsPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

        main_page_button = tk.Button(self, text="Main page", command=lambda: controller.show_frame(MainPage))
        main_page_button.pack(side="right", expand=True)

        quit_button = tk.Button(self, text="Quit", command=self.quit)
        quit_button.pack(side="right", expand=True)

        self.control_panel = tk.Frame(self)
        self.control_panel_init(self.control_panel)
        self.control_panel.pack(side="left", expand=True)

    def set_g(self, g_entries, g_type):

        i = 0
        for entries_row in g_entries:
            j = 0
            for g in entries_row:
                value = float(g.get())
                self.controller.sampler.set_g(i, j, value, g_type)
                self.controller.recognizer.set_g(i, j, value, g_type)
                j += 1
            i += 1

    def set_num_colors(self, entry):

        num_colors = int(entry.get())
        self.controller.sampler.set_num_colors(num_colors)
        self.controller.recognizer.set_num_colors(num_colors)
        self.control_panel.destroy()
        self.control_panel = tk.Frame(self)
        self.control_panel_init(self.control_panel)
        self.control_panel.pack(side="left", expand=True)
        self.controller.frames[MainPage].update_generated_image()

    def set_num_iterations(self, object_, entry):

        num_iter = int(entry.get())
        object_.set_num_iterations(num_iter)

    def set_image_height(self, entry):

        height = int(entry.get())
        self.controller.sampler.set_image_height(height)
        self.controller.recognizer.set_image_height(height)
        self.controller.frames[MainPage].change_image_size(height, 'h')
        print('Image height: %d' % height)

    def set_image_width(self, entry):

        width = int(entry.get())
        self.controller.sampler.set_image_width(width)
        self.controller.recognizer.set_image_width(width)
        self.controller.frames[MainPage].change_image_size(width, 'w')
        print('Image width: %d' % width)

    def control_panel_init(self, control_panel):

        num_colors = self.controller.sampler.num_colors
        num_gen_iterations = self.controller.sampler.num_iterations
        num_rec_iterations = self.controller.recognizer.num_iterations
        image_height = self.controller.sampler.image_height
        image_width = self.controller.sampler.image_width
        num_rows = 2 * (self.controller.sampler.num_colors + 3)
        num_columns = self.controller.sampler.num_colors + 3

        label_names = ['Number of generating iterations:', 'Image height:',
                       'Image width:', 'Number of recognition iterations:', 'Num colors:']
        COLOR_NAMES = ('Red', 'Blue', 'Green', 'Cyan', 'Magenta', 'Yellow', 'Black', 'White')
        label_grid_coords = [(0, num_columns), (3, num_columns), (6, num_columns), (9, num_columns), (12, num_columns)]

        entries = []
        for label_name, label_coords in zip(label_names, label_grid_coords):
            tk.Label(control_panel, text=label_name).grid(row=label_coords[0], column=label_coords[1])
            ent = tk.Entry(control_panel)
            ent.grid(row=label_coords[0] + 1, column=label_coords[1])
            entries.append(ent)

        # entries_texts = ('100', '20', '1000', str(num_colors))
        # entries_actions = (
        #     self.set_num_iterations,
        #     self.set_image_size,
        #     self.set_num_iterations,
        #     self.set_num_colors
        # )
        # for entry, text, action in zip(entries, entries_texts, entries_actions):
        #     entry.insert(0, text)
        #     entry.bind('<Return>', lambda event: action(entry))

        entries[0].insert(0, str(num_gen_iterations))
        entries[0].bind('<Return>', lambda event: self.set_num_iterations(self.controller.sampler, entries[0]))
        entries[1].insert(0, str(image_height))
        entries[1].bind('<Return>', lambda event: self.set_image_height(entries[1]))
        entries[2].insert(0, str(image_width))
        entries[2].bind('<Return>', lambda event: self.set_image_width(entries[2]))
        entries[3].insert(0, str(num_rec_iterations))
        entries[3].bind('<Return>', lambda event: self.set_num_iterations(self.controller.recognizer, entries[3]))
        entries[4].insert(0, str(num_colors))
        entries[4].bind('<Return>', lambda event: self.set_num_colors(entries[4]))

        g_horizontal_entries = []
        g_vertical_entries = []
        g_entries = [g_vertical_entries, g_horizontal_entries]

        for gs, start_row, g_type in zip(g_entries, (0, num_colors + 3), ('v', 'h')):
            for row in range(start_row, start_row + num_colors):
                tk.Label(control_panel, text=COLOR_NAMES[row - start_row]).grid(row=row + 1, column=0)
                tk.Label(control_panel, text=COLOR_NAMES[row - start_row]).grid(row=0, column=row + 1 - start_row)
                entries_row = []
                for column in range(num_columns - 3):
                    ent = tk.Entry(control_panel)
                    ent.insert(0, str(self.controller.sampler.get_g(row - start_row, column, g_type)))
                    ent.grid(row=row + 1, column=column + 1)
                    entries_row.append(ent)
                gs.append(entries_row)

        set_g_button = tk.Button(control_panel, text="Set vertical g's", command=lambda: self.set_g(g_vertical_entries, "v"))
        set_g_button.grid(row=num_colors + 3, column=2)

        set_g_button = tk.Button(control_panel, text="Set horizontal g's", command=lambda: self.set_g(g_horizontal_entries, "h"))
        set_g_button.grid(row=2 * (num_colors + 3), column=2)

    def quit(self):

        self.controller.save()
        tk.Frame.quit(self)


class GraphPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.graph_names = ["Pixelwise max-freq", "Line max_freq", "Pixelwise aver-prob", "Line aver-prob", "Min-cut"]
        self.graph_styles = ['r-', 'b-', 'm-', 'c-', 'g-']
        self.data_lists = []

        main_page_button = tk.Button(self, text="Main page", command=lambda: controller.show_frame(MainPage))
        main_page_button.pack(side="right", expand=True)

        self.figure = self.axes_init()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack()

    def get_data(self, average_times, *data_lists):

        print("Data lists: ", data_lists)
        self.average_times = average_times
        self.data_lists = data_lists
        self.graph_update()

    def axes_init(self):

        Y_LIMIT = 1000
        X_LIMIT = 20

        fig, ax = plt.subplots(figsize=(10, 3), nrows=1, ncols=1)
        ax.set_ylim(0, Y_LIMIT)
        ax.set_xlim(0, X_LIMIT)
        ax.set_xticks(np.arange(X_LIMIT))
        self.set_graph(ax)

        return fig

    def set_graph(self, ax):
        ax.cla()
        args = []
        kvargs = {}
        for index, data in enumerate(self.data_lists):
            args.append(np.arange(len(data)) * self.average_times[index])
            # args.append(np.arange(len(data)))
            args.append(data)
            args.append(self.graph_styles[index])
            kvargs['label'] = self.graph_names[index]
            ax.plot(*args, **kvargs)
            args = []
            kvargs = {}

        ax.legend(loc=1)

    def graph_update(self):

        self.set_graph(self.canvas.figure.axes[0])
        self.canvas.draw()



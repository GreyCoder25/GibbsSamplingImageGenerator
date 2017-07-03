import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import model
import tkinter as tk

class ImageGeneratorApp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side="top", expand=True, fill="both")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.sampler = model.GibbsSamplingImageGenerator()
        self.recognizer = model.GibbsSamplingImageRecognizer(self.sampler)

        for F in (MainPage, SettingsPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainPage)


    def show_frame(self, controller):

        frame = self.frames[controller]
        frame.tkraise()


class MainPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.num_rows = 3
        self.num_columns = 5
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

        reset_button = tk.Button(self, text="Reset", command=self.reset)
        reset_button.grid(row=0, column=self.num_columns - 3)

        # buttons for generated image graph
        tk.Button(self, text="Next iteration", command=self.next_generating_iteration).grid(row=2,
                                                                                            column=0, sticky="nsew")
        tk.Button(self, text="Execute all remaining", command=self.execute_all_gen_remaining).grid(row=2,
                                                                                                column=1,sticky="nsew")

        # Noise button
        tk.Button(self, text="Noise", command=self.show_noisy_image).grid(row=2, column=2, sticky="nsew")

        # buttons for recognized image graph
        tk.Button(self, text="Next iteration", command=self.next_recognition_iteration).grid(row=2, column=3,
                                                                                                                 sticky="nsew")
        tk.Button(self, text="Execute all remaining", command=self.execute_all_rec_remaining).grid(row=2,
                                                                                                        column=4, sticky="nsew")
        # show image graphs
        self.figure = self.image_graphs_init()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=1, columnspan=self.num_columns)

    def image_graphs_init(self):

        IMAGE_WIDTH = 20
        IMAGE_HEIGHT = 20

        fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=3)

        for ax, graph_name in zip(axes, self.graph_names):
            ax.set_ylim(IMAGE_HEIGHT, 0)
            ax.set_xlim(0, IMAGE_WIDTH)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(graph_name)
            ax.xaxis.set_ticks_position('top')

        self.pix_vals_gen = self.controller.sampler.image
        self.pix_vals_rec = self.controller.recognizer.image
        self.image_init(axes[0])
        return fig

    def image_init(self, ax, pix_vals_type='gen'):

        if pix_vals_type == 'gen':
            pix_vals = self.pix_vals_gen
        elif pix_vals_type == 'rec':
            pix_vals = self.pix_vals_rec
        cmap = mpl.colors.ListedColormap(['red', 'blue', 'green'])
        p = ax.pcolormesh(pix_vals, cmap=cmap)

    def reset(self):
        pass

    def change_image_size(self, size):

        for ax in self.canvas.figure.axes:
            ax.set_ylim(size, 0)
            ax.set_xlim(0, size)

        self.pix_vals_gen = self.controller.sampler.image
        self.image_init(self.canvas.figure.axes[0])
        self.canvas.draw()

    def update_generated_image(self):
        self.pix_vals_gen = self.controller.sampler.image
        self.image_init(self.canvas.figure.axes[0])
        self.canvas.draw()

    def update_recognized_image(self):
        self.pix_vals_rec = self.controller.recognizer.image
        self.image_init(self.canvas.figure.axes[2], 'rec')
        self.canvas.draw()

    def show_noisy_image(self):
        self.controller.sampler.noise()
        self.pix_vals_gen = self.controller.sampler.image
        self.controller.recognizer.image = self.controller.sampler.image
        self.image_init(self.canvas.figure.axes[1])
        self.image_init(self.canvas.figure.axes[2])
        self.canvas.draw()

    def next_generating_iteration(self):
        for i in range(10):
            self.controller.sampler.iteration_of_generation()
            self.update_generated_image()

    def next_recognition_iteration(self):
        self.controller.recognizer.iteration_of_recognition()
        self.update_recognized_image()

    def execute_all_gen_remaining(self):
        self.controller.sampler.execute_all_remaining()
        self.update_generated_image()

    def execute_all_rec_remaining(self):
        self.controller.recognizer.execute_all_remaining()
        self.update_recognized_image()


class SettingsPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

        main_page_button = tk.Button(self, text="Main page", command=lambda: controller.show_frame(MainPage))
        main_page_button.pack(side="right", expand=True)

        quit_button = tk.Button(self, text="Quit", command=self.quit)
        quit_button.pack(side="right", expand=True)

        control_panel = tk.Frame(self)
        self.control_panel_init(control_panel)
        control_panel.pack(side="left", expand=True)

    def set_g(self, g_entries, g_type):
        i = 0
        for entries_row in g_entries:
            j = 0
            for g in entries_row:
                value = float(g.get())
                self.controller.sampler.set_g(i, j, value, g_type)
                j += 1
            i += 1

    def set_num_iterations(self, object, entry):
        num_iter = int(entry.get())
        object.set_num_iterations(num_iter)

    def set_image_size(self, entry):
        size = int(entry.get())
        self.controller.sampler.set_image_size(size)
        self.controller.recognizer.set_image_size(size)
        self.controller.frames[MainPage].change_image_size(size)

    def control_panel_init(self, control_panel):
        num_rows = 12
        num_columns = 6
        RED = 0
        BLUE = 1
        GREEN = 2
        COLORS = (RED, BLUE, GREEN)

        label_names = ['Number of generating iterations:', 'Image size:', 'Number of recognition iterations:']
        color_names = ['Red', 'Blue', 'Green']
        label_grid_coords = [(0, 6), (3, 6), (6, 6)]
        entries = []
        for label_name, label_coords in zip(label_names, label_grid_coords):
            tk.Label(control_panel, text=label_name).grid(row=label_coords[0], column=label_coords[1])
            ent = tk.Entry(control_panel)
            ent.grid(row=label_coords[0] + 1, column=label_coords[1])
            entries.append(ent)

        entries[0].insert(0, '100')
        entries[0].bind('<Return>', lambda event: self.set_num_iterations(self.controller.sampler, entries[0]))
        entries[1].insert(0, '20')
        entries[1].bind('<Return>', lambda event : self.set_image_size(entries[1]))
        entries[2].insert(0, '1000')
        entries[2].bind('<Return>', lambda event: self.set_num_iterations(self.controller.recognizer, entries[2]))

        g_horizontal_entries = []
        g_vertical_entries = []
        g_entries = [g_vertical_entries, g_horizontal_entries]

        for gs, start_row in zip(g_entries, [0, 6]):
            for row in range(start_row, start_row + 3):
                tk.Label(control_panel, text=color_names[row - start_row]).grid(row=row + 1, column=0)
                tk.Label(control_panel, text=color_names[row - start_row]).grid(row=0, column=row + 1 - start_row)
                entries_row = []
                for column in range(num_columns - 3):
                    ent = tk.Entry(control_panel)
                    ent.insert(0, '1')
                    ent.grid(row=row + 1, column=column + 1)
                    entries_row.append(ent)
                gs.append(entries_row)

        set_g_button = tk.Button(control_panel, text="Set vertical g's", command=lambda: self.set_g(g_vertical_entries, "v"))
        set_g_button.grid(row=6, column=2)

        set_g_button = tk.Button(control_panel, text="Set horizontal g's", command=lambda: self.set_g(g_horizontal_entries, "h"))
        set_g_button.grid(row=12, column=2)




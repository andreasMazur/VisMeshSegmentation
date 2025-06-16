from deepview import DeepView
from deepview.Selector import SelectFromCollection
from .recommendation_functions import *


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from deepview.config import max_iter

max_iter = 500
class DeepViewLabelRevisit(DeepView):

    def __init__(self, *args, **kwargs):
        self.changed_indices = None
        super().__init__(*args, **kwargs)


    def _init_plots(self):
        '''
        Initialises matplotlib artists and plots.
        '''
        if self.interactive:
            plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.ax.set_title(self.title)
        self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
            interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []

        class_label_display = (self.class_dict if self.class_dict is not None else self.classes)
        for c in range(self.n_classes):
            color = self.cmap(c/(self.n_classes-1))
            plot = self.ax.plot([], [], 'o', label=class_label_display[c],
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        for c in range(self.n_classes):
            color = self.cmap(c/(self.n_classes-1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                fillstyle='none', ms=12, mew=2.5, zorder=1)
            self.sample_plots.append(plot[0])

        # set the mouse-event listeners
        if self.use_selector:
            self.fig.canvas.mpl_connect('key_press_event', self.show_sample)
        else:
            self.fig.canvas.mpl_connect('pick_event', self.show_sample)
            self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
        self.disable_synth = False
        self.ax.set_axis_off()
        self.ax.legend()


    def recommend_label_correction(self,k,percentage):
        changed_labels, all_indices, keep_indices = recommend_KNN_based(self.embedded, self.y_true,
                                                                               self.y_true, k,
                                                                               recommendation_percentage=percentage)
        self.changed_indices = keep_indices
        return keep_indices, changed_labels

    def show(self):
        '''
        Shows the current plot.
        '''
        if not hasattr(self, 'fig'):
            self._init_plots()

        x_min, y_min, x_max, y_max = self._get_plot_measures()

        self.cls_plot.set_data(self.classifier_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'batch size: %d - n: %d - $\lambda$: %.2f - res: %d'
        desc = params_str % (self.batch_size, self.n, self.lam, self.resolution)
        self.desc.set_text(desc)

        for c in range(self.n_classes):
            data = self.embedded[self.y_true==c]
            self.sample_plots[c].set_data(data.transpose())

        change_indices = self.recommend_label_correction(100)
        for c in range(self.n_classes):
            # data = self.embedded[np.logical_and(self.true==c, self.background_at!=c)]
            data = self.embedded[change_indices]
            self.sample_plots[self.n_classes+c].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        if self.use_selector:
            self.selector = SelectFromCollection(self.ax, self.embedded)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

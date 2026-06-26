import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from deepview import DeepView
from deepview.Selector import SelectFromCollection
from deepview.config import max_iter

from improve_mesh_segmentation.comparison_methods.recommendation import recommend_knn_based

max_iter = 500


class DeepViewLabelRevisit(DeepView):
    def __init__(self, *args, **kwargs):
        self.changed_indices = None
        super().__init__(*args, **kwargs)
        self.mesh_preds = None

    def _init_plots(self):
        if self.interactive:
            plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.ax.set_title(self.title)
        self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(
            np.zeros([5, 5, 3]),
            interpolation='gaussian',
            zorder=0,
            vmin=0,
            vmax=1,
        )

        self.sample_plots = []
        class_label_display = self.class_dict if self.class_dict is not None else self.classes
        for c in range(self.n_classes):
            color = self.cmap(c / (self.n_classes - 1))
            plot = self.ax.plot([], [], 'o', label=class_label_display[c], color=color, zorder=2,
                                picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        for c in range(self.n_classes):
            color = self.cmap(c / (self.n_classes - 1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color, fillstyle='none', ms=12, mew=2.5, zorder=1)
            self.sample_plots.append(plot[0])

        if self.use_selector:
            self.fig.canvas.mpl_connect('key_press_event', self.show_sample)
        else:
            self.fig.canvas.mpl_connect('pick_event', self.show_sample)
            self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
        self.disable_synth = False
        self.ax.set_axis_off()
        self.ax.legend()

    def recommend_label_correction(self, k, percentage):
        changed_labels, all_indices, keep_indices = recommend_knn_based(
            self.embedded,
            self.y_true,
            self.y_true,
            k,
            recommendation_percentage=percentage,
        )
        self.changed_indices = keep_indices
        return keep_indices, changed_labels

    def compute_grid(self):
        if self.verbose:
            print('Computing decision regions ...')
        x_min, y_min, x_max, y_max = self._get_plot_measures()
        xs = np.linspace(x_min, x_max, self.resolution)
        ys = np.linspace(y_min, y_max, self.resolution)
        self.grid = np.array(np.meshgrid(xs, ys))
        grid = np.swapaxes(self.grid.reshape(self.grid.shape[0], -1), 0, 1)
        grid_samples = self.inverse(grid)

        mesh_preds = self._predict_batches(grid_samples)
        mesh_preds = mesh_preds + 1e-8
        self.mesh_preds = mesh_preds

        self.mesh_classes = mesh_preds.argmax(axis=1)
        mesh_max_class = max(self.mesh_classes)

        color = self.cmap(self.mesh_classes / mesh_max_class)
        h = -(mesh_preds * np.log(mesh_preds)).sum(axis=1) / np.log(self.n_classes)
        h = (h / h.max()).reshape(-1, 1)
        h = np.clip(h * 1.2, 0, 1)
        color = color[:, 0:3]
        color = (1 - h) * (0.5 * color) + h * np.ones(color.shape, dtype=np.uint8)
        return color.reshape(self.resolution, self.resolution, 3)

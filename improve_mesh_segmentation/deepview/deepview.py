from deepview.DeepView import DeepView

import numpy as np
import matplotlib.pyplot as plt
import warnings


class DeepViewMesh(DeepView):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def show_sample(self, event):
        '''
        Invoked when the user clicks on the plot. Determines the
        embedded or synthesised sample at the click location and
        passes it to the data_viz method, together with the prediction,
        if present a groun truth label and the 2D click location.
        '''

        # when there is an artist attribute, a
        # concrete sample was clicked, otherwise
        # show the according synthesised image
        if self.use_selector == False and hasattr(event, 'artist'):
            artist = event.artist
            ind = event.ind
            xs, ys = artist.get_data()
            point = [xs[ind][0], ys[ind][0]]
            sample, p, t = self.get_artist_sample(point)
            title = '%s <-> %s' if p != t else '%s --- %s'
            title = title % (self.classes[p], self.classes[t])
            self.disable_synth = True
        elif self.use_selector and event.key == "enter":
            indices = self.selector.ind
            sample, p, t = self.get_artist_sample(indices)
            self.disable_synth = True

        elif not self.disable_synth:
            # workaraound: inverse embedding needs more points
            # otherwise it doens't work --> [point]*5
            point = np.array([[event.xdata, event.ydata]] * 5)

            # if the outside of the plot was clicked, points are None
            if None in point[0]:
                return

            sample = self.inverse(point)[0]
            sample += abs(sample.min())
            sample /= sample.max()
            title = 'Synthesised at [%.1f, %.1f]' % tuple(point[0])
            p, t = self.get_mesh_prediction_at(*point[0]), None
        else:
            self.disable_synth = False
            return

        is_image = self.is_image(sample)
        rank_perm = np.roll(range(len(sample.shape)), -1)
        sample_T = sample.transpose(rank_perm)
        is_transformed_image = self.is_image(sample_T)

        if self.use_selector == False and self.data_viz is not None:
            self.data_viz(sample, point, p, t)
            return
        if self.use_selector and self.data_viz is not None:
            self.data_viz(sample, p, t, self.cmap)
            return
        # try to show the image, if the shape allows it
        elif is_image:
            img = sample - sample.min()
        elif is_transformed_image:
            img = sample_T - sample_T.min()
        else:
            warnings.warn("Data visualization not possible, as the data points have"
                          "no image shape. Pass a function in the data_viz argument,"
                          "to enable custom data visualization.")
            return

        f, a = plt.subplots(1, 1)
        a.imshow(img / img.max())
        a.set_title(title)

import numpy as np
from ..utilities import hash_images


class FingerprintPlot:
    """Create plots of fingerprint ranges.
    Initialize with an Amp calculator object.
    """

    def __init__(self, calc):
        self._calc = calc

    def __call__(self, images, name='fingerprints.pdf'):
        """Creates a violin plot of fingerprints for each element type in
        the fed images; saves to specified filename."""

        from matplotlib import pyplot
        from matplotlib.backends.backend_pdf import PdfPages

        self.compile_fingerprints(images)

        self.figures = {}
        for element in self.data.keys():
            self.figures[element] = pyplot.figure(figsize=(11., 8.5))
            fig = self.figures[element]
            ax = fig.add_subplot(211)
            ax.violinplot(self.data[element])
            ax.set_ylabel('raw value')
            ax.set_xlim([0, self.data[element].shape[1] + 1])
            if hasattr(self._calc.model.parameters, 'fprange'):
                ax2 = fig.add_subplot(212)
                fprange = self._calc.model.parameters.fprange[element]
                fprange = np.array(fprange)
                fprange.transpose()
                d = self.data[element]
                scaled = ((d - fprange[:, 0]) /
                          (fprange[:, 1] - fprange[:, 0]) * 2.0 - 1.0)
                ax2.violinplot(scaled)
                ax2.set_ylabel('scaled value')
                ax2.set_xlim([0, self.data[element].shape[1] + 1])
                ax2.set_ylim([-1.05, 1.05])
                ax2.set_xlabel('fingerprint')
            else:
                ax.set_xlabel('fingerprint')
                fig.text(0.5, 0.25,
                         '(No fprange in model; therefore no scaled '
                         'fingerprints shown.)',
                         ha='center')
            fig.text(0.5, 0.95, element, ha='center')

        with PdfPages(name) as pdf:
            for fig in self.figures.values():
                pdf.savefig(fig)
                pyplot.close(fig)

    def compile_fingerprints(self, images):
        """Calculates or looks up fingerprints and compiles them, per
        element, for the images."""
        data = self.data = {}
        images = hash_images(images)
        self._calc.descriptor.calculate_fingerprints(images)
        for hash in images.keys():
            fingerprints = self._calc.descriptor.fingerprints[hash]
            for element, fingerprint in fingerprints:
                if element not in data:
                    data[element] = []
                data[element].append(fingerprint)
                print(element, len(fingerprint))
        for element in data.keys():
            data[element] = np.array(data[element])

import numpy as np
from scipy.stats import beta

import platform

from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from .utils import *

class Landshaft:
    """
    A main class, it provide general lowlands algorithm for multimodality detection,
    based on quantile-respectful density estimation. The algorithm was proposed by
    Andrey Akinshin. See for more details: https://aakinshin.net/posts/lowland-multimodality-detection/

    Parameters
    ----------
    x : array-like
        Input data for analysis.
    q_num : int
        Number of quantiles to use.
    threshold : float, optional
        Threshold for determining lowland ponds, by default 0.5
    """
    
    def __init__(self, x: np.ndarray, q_num: int, threshold: float = 0.5) -> None:
        self.x = np.sort(x)
        self.q_num = q_num
        self.threshold = threshold


    @classmethod
    def _hd(cls, x: np.ndarray, q: float = 0.5, eps=1e-6) -> float:
        """
        Calculate the Harrell-Davis quantile estimate.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        q : float, optional
            Quantile to compute, by default 0.5

        Returns
        -------
        float
            The Harrell-Davis quantile estimate.
        """
        if q == 0:
            return x[0]
        if q > 1-eps and q <1+eps:
            return x[-1]
        
        n = x.shape[0]
        m1 = (n + 1) * q
        m2 = (n + 1) * (1 - q)
        vec = np.arange(n)
        beta_dist = beta(m1, m2)
        w = beta_dist.cdf((vec + 1) / n) - beta_dist.cdf((vec) / n)
        return np.sum(w * x)
    
    def get_quantiles(self) -> None:
        """
        Calculate quantiles of the input data and store Harrell-Davis estimates.
        """
        step = 1 / self.q_num

        quantiles = np.arange(0.0, 1.0 + step/10, step).tolist()

        if platform.system() != "Windows":
            global hd_v
            def hd_v(q):
                return Landshaft._hd(self.x, q, eps=step/10)
            
            with Pool(processes=cpu_count()) as pool:
                res = list(pool.map(hd_v, quantiles, chunksize=10))

            self.hds = np.array(res)
        else:
            hd_v = np.vectorize(lambda q: Landshaft._hd(self.x, q, eps=step/10))
            self.hds = hd_v(quantiles)

    def get_ground(self) -> None:
        """
        Create Ground object with bins defined by quantiles.
        """
        
        vals, densities, widths = (
            self.hds[:-1],
            1 / (self.hds[1:] - self.hds[:-1]) / len(self.hds[1:]),
            (self.hds[1:] - self.hds[:-1])
        )
        # handling equal hds
        idxs = (self.hds[1:] - self.hds[:-1]) > 1e-8 * self.hds[:-1].mean()
        print(idxs)
        vals = vals[idxs]
        densities = densities[idxs]
        widths = widths[idxs]

        self.ground = Ground(np.array([Bin(val, width, height) for val, width, height in zip(vals, widths, densities)]))

    def get_extremums(self) -> None:
        """
        Identify extremum points in the ground object.
        """
        density_diff = np.diff(self.ground.get_bins_height())
        is_turn = (np.sign(density_diff[1:]) * np.sign(density_diff[:-1]) <= 0) & (np.sign(density_diff[:-1]) != 0)
        idxs = np.nonzero(is_turn)[0] + 1
        signs = np.sign(density_diff[:-1])
        if density_diff[0] < 0:
            idxs = np.concatenate([np.array([0]), idxs])
            signs = np.concatenate([np.array([1]), signs]) 
        if density_diff[-1] > 0:
            idxs = np.concatenate([idxs, np.array([len(density_diff)])])
            signs = np.concatenate([signs, np.array([1])])
        self.extremums = ExtremumArray(np.array([Extremum(bin, sign) for bin, sign in zip(self.ground[idxs].bins, signs[idxs - 1])]))

    def get_peaks_and_ponds(self) -> None:
        """
        Identify peaks and ponds in the extremum points.
        """
        self.peaks = self.extremums[self.extremums.signs() > 0]
        
        ponds = [
            Pond(self.peaks[i], self.peaks[i + 1],
                 self.ground[(self.peaks[i].start < self.ground.get_bins_start()) & (self.peaks[i + 1].start > self.ground.get_bins_start())],
                 threshold=self.threshold)
            for i in range(self.peaks.shape[0] - 1)
        ]
        self.ponds = ponds

    def merge_highland_ponds(self) -> None:
        """
        Merge ponds in highland areas.
        """
        for i in range(len(self.ponds)):
            for j in range(i + 1, len(self.ponds)):
                if not (self.ponds[i].is_lowland() or self.ponds[j].is_lowland() or self.ponds[i].merged) and \
                        self.ponds[i].start_peak.height > self.ponds[j].start_peak.height and \
                        self.ponds[i].start_peak.height > self.ponds[i].end_peak.height:
                    self.ponds[i].merge(self.ponds[j])
                else:
                    break
        
        self.ponds = [pond for pond in self.ponds if not pond.merged]

    def get_modes(self) -> tuple['ExtremumArray', 'ExtremumArray']:
        """
        Identify modes and minimum points between modes.
        """
        lowland_ponds = [pond for pond in self.ponds if pond.is_lowland()]
        if not lowland_ponds:
            modes = ExtremumArray([self.peaks[np.argmax(self.peaks.get_bins_height())]
                                   if isinstance(self.peaks[np.argmax(self.peaks.get_bins_height())], Extremum ) 
                                   else self.peaks[np.argmax(self.peaks.get_bins_height())]
                                   ])
            min_between_modes = None
        else:    
            between_lowland = np.array(
                [[self.ground.get_bins_start()[0], lowland_ponds[0].start_peak.start + lowland_ponds[0].start_peak.width / 2]] +
                [[lowland_ponds[i].end_peak.start, lowland_ponds[i + 1].start_peak.start + lowland_ponds[i + 1].start_peak.width / 2]
                for i in range(len(lowland_ponds) - 1)] +
                [[lowland_ponds[-1].end_peak.start, (self.ground.get_bins_start() + self.ground.get_bins_width() / 2)[-1]]]
            )

            peaks_start = self.peaks.get_bins_start().reshape(-1, 1)
            condition = (peaks_start >= between_lowland[:, 0]) & (peaks_start < between_lowland[:, 1])
            candidates_idx = condition.sum(axis=1).astype(bool)
            candidates_group = condition[candidates_idx].argmax(axis=1)

            a_max_height = [np.argmax(x) for x in np.split(self.peaks.get_bins_height()[candidates_idx],
                                                            np.unique(candidates_group, return_index=True)[1][1:])]

            modes = ExtremumArray([x[max_idx] for x, max_idx in zip(np.split(self.peaks[candidates_idx],
                                                                np.unique(candidates_group, return_index=True)[1][1:]), a_max_height)])

            min_between_modes = ExtremumArray([
                self.extremums[(pond.start_peak.start < self.extremums.get_bins_start()) & 
                            (self.extremums.get_bins_start() < pond.end_peak.start)][
                    np.argmin(self.extremums[(pond.start_peak.start < self.extremums.get_bins_start()) & 
                                            (self.extremums.get_bins_start() < pond.end_peak.start)].get_bins_height())
                ] for pond in lowland_ponds
            ])

        self.modes, self.min_between_modes = modes, min_between_modes

    def build_landshaft(self) -> None:
        """
        The main function of class providing general logic of algorithm.
        Constructs the landshaft by calculating quantiles, building the ground,
        identifying extremums, peaks, and ponds, merging highland ponds, and determining modes.

        This method follows a series of steps to build the landshaft:
        
        - Calculate quantiles using `get_quantiles()`.
        - Construct the ground from the quantiles using `get_ground()`.
        - Identify extremums in the ground using `get_extremums()`.
        - Determine peaks and ponds using `get_peaks_and_ponds()`.
        - Merge highland ponds based on certain conditions using `merge_highland_ponds()`.
        - Finally, identify modes using `get_modes()`.

        Returns
        -------
        None
        """
        self.get_quantiles()
        self.get_ground()
        self.get_extremums()
        self.get_peaks_and_ponds()
        self.merge_highland_ponds()
        self.get_modes()

    def plot_hds(self) -> None:
        """
        Plot the Harrell-Davis estimates.
        """
        bins_start = self.ground.get_bins_start()
        plt.ion()
        plt.show()
        plt.bar((np.arange(len(bins_start)) + 1) / len(bins_start), bins_start, 1 / len(bins_start),
                facecolor=(1, 1, 1, 0),
                edgecolor=(0, 0, 1, 0.5),
                linewidth=0.25)
        plt.draw()
        plt.pause(0.1)

        if not is_notebook():
            input("Press [enter] to continue.")
        plt.close()
    
    def plot_ground(self, is_terminal: bool = True) -> plt.Axes:
        """
        Plot the ground landshaft.

        This method plots the ground landshaft based on bin starts, widths, and heights.
        It waits for user input and then closes the plot and returns None if `is_terminal` is True.
        And it does not close the plot and returns a pyplot.Axes object for further plotting otherwise.

        Parameters
        ----------
        is_terminal : bool, optional
            If True, the function will wait for user input and then close the plot.
            If False it does not close the plot and returns a pyplot.Axes object. 
            The default is False.

        Returns
        -------
        plt.Axes
            The Axes object of the plot.
        """
        bins_x = self.ground.get_bins_start()
        widths = self.ground.get_bins_width()
        density = self.ground.get_bins_height()
        
        plt.ion()
        plt.bar(bins_x + widths / 2, density, widths, 
                facecolor=(0.4, 0.2, 0, 0.6), 
                edgecolor=(1, 1, 1, 0))
        
        if is_terminal:
            plt.draw()
            plt.pause(0.1)
            if not is_notebook():
                try:
                    input("Press [enter] to continue.")
                except EOFError:
                    plt.savefig("peaks_and_lowlands.png")
            plt.close()
        else:
            return plt.gca()

    def plot_ponds_and_peaks(self) -> None:
        """
        Plot the ponds and peaks on the landscape.

        This method plots the density of the ground and overlays the positions of 
        the ponds and peaks. Ponds are colored based on whether they are lowland 
        or highland, and peaks are marked with green asterisks. Minimum points 
        between modes are marked with red asterisks.
        """
        ax = self.plot_ground(is_terminal=False)
        modes, mins = self.modes, self.min_between_modes
        
        bins_x = [pond.ground_underwater.get_bins_start()[0] for pond in self.ponds]
        widths = [
            pond.ground_underwater.get_bins_start()[-1] 
            - pond.ground_underwater.get_bins_start()[0] 
            + pond.ground_underwater.get_bins_width()[-1] 
            for pond in self.ponds
        ]
        
        heights = [pond.level for pond in self.ponds]
        colors = [(0, 0, 1, 0.6) if pond.is_lowland() else (0, 0, 1, 0.2) for pond in self.ponds]
        
        ax.bar(bins_x, heights, widths, color=colors, align='edge')
        ax.scatter(modes.get_bins_start() + modes.get_bins_width()/2, modes.get_bins_height(), marker="*", color="green")
        if not (mins) is None:
            ax.scatter(mins.get_bins_start() + mins.get_bins_width()/2, mins.get_bins_height(), marker="*", color="red")
        
        plt.draw()
        plt.pause(0.1)
        if not is_notebook():
            try:
                input("Press [enter] to continue.")
            except EOFError:
                plt.savefig("peaks_and_lowlands.png")
        plt.close()

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False  
    except NameError:
        return False 
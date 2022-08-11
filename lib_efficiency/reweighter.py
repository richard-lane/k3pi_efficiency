"""
Implementation for the efficiency reweighter

"""
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
from hep_ml.reweight import GBReweighter, BinsReweighter

from lib_efficiency import time_fitter


class TrainingError(Exception):
    """
    Useful because we might be multiprocessing; raise this if anything
    went wrong in a child process

    """

    def __init__(self):
        """
        Error training the BDT or something

        """
        super().__init__(
            "Error training the BDT; see above for child process tracebacks"
        )


class TimeFitReweighter:
    """
    Do the reweighting by performing a fit to decay times

    Does the reweighting AmpGen -> MC (i.e. the weights here apply the
    efficiency, they don't correct for it) - this is because this seems
    to be more stable

    Get the weights by assuming the true decay time distribution is
    e^-t

    """

    def __init__(self):
        """
        List of attributes

        """
        self.fitter = None

    def fit(self, _, times: np.ndarray):
        """
        Perform the fit

        Dummy var for API consistency with BinsReweighter.fit

        """
        # These initial parameters seem to do the right thing, mostly
        self.fitter = time_fitter.fit(times, (0.21, 1.0, 2.0, 1.0, 2.0, 1.0))

    def _fitted_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Return (normalised) pdf values at each time passed in

        """
        assert (
            self.fitter is not None
        ), "need to call .fit() before we have a fitted pdf"
        return time_fitter.normalised_pdf(x, *self.fitter.values)[1]

    def predict_weights(self, times: np.ndarray) -> np.ndarray:
        """
        Weights to apply the efficiency

        Returns 0 for times below the fitted minimum

        """
        return self._fitted_pdf(times) / np.exp(-times)


class TimeWeighter:
    """
    The time reweighter

    Or via a histogram division

    Basically this class just holds either a BinsReweighter or TimeFitReweighter
    object, and uses this to perform the time weighting

    """

    def __init__(self, min_t: float, fit: bool):
        """
        Tell us whether we're doing a fit to the decay times

        :param min_t: time below which to set all weights to 0 anyway
        :param fit: whether we want to reweight using the decay time fit
                    or by using a BinsReweighter

        """
        self.min_t = min_t
        self.fitter = (
            TimeFitReweighter() if fit else BinsReweighter(n_bins=20000, n_neighs=10)
        )

    def fit(self, mc_times: np.ndarray, ampgen_times: np.ndarray):
        """
        ampgen times arg is unused if we're doing a fit

        """
        # Reweight AmpGen to MC to avoid getting huge weights
        # If we're doing a histogram division, only use times above the minimum
        if isinstance(self.fitter, BinsReweighter):
            ag_t = ampgen_times[ampgen_times > self.min_t]
            mc_t = mc_times[mc_times > self.min_t]
            print(
                f"{self.fitter.n_bins} bins\n"
                f"{len(ag_t)}, {len(mc_t)} times above minimum.\n"
                f"Avg of {len(ag_t) / self.fitter.n_bins}, "
                f"{len(mc_t) / self.fitter.n_bins} per bin"
            )
            self.fitter.fit(ag_t, mc_t)
        # If we're fitting fit all the times; not just the ones above the min time
        else:
            self.fitter.fit(ampgen_times, mc_times)

    def apply_efficiency(self, times):
        """
        Predict weights to apply the efficiency

        """
        above_min = times > self.min_t
        retval = np.zeros_like(times)
        retval[above_min] = self.fitter.predict_weights(times[above_min])

        return retval

    def correct_efficiency(self, times):
        """
        Weights to correct for the efficiency

        """
        above_min = times > self.min_t
        retval = np.zeros_like(times)
        retval[above_min] = 1 / self.fitter.predict_weights(times[above_min])

        return retval


class EfficiencyWeighter:
    """
    Holds a time reweighter that either does a histogram division or a fit to the decay times
    Holds also a BDT reweighter that deals with the phsp efficiency, but uses the time reweighter
    to keep the correlations between time and phase space

    """

    def __init__(
        self, target: np.ndarray, original: np.ndarray, fit: bool, min_t: float
    ):
        """
        Perform time weighting and train the BDT

        First finds the decay time efficiency e(t), then trains BDT to reweight mc -> AmpGen * e(t)

        :param target: shape (N, 6) array of phsp points; 6th element of each should be decay time
        :param original: shape (N, 6) array of phsp points; 6th element of each should be decay time
        :param fit: whether to find the decay time efficiency by performing a fit. Does a histogram
                    division if False.
        :param min_t: minimum time, below which weights are set to 0.
                      The histogram division only considers times above this, but the fit fits to all
                      times.

        """
        self._time_weighter = TimeWeighter(min_t, fit)
        self._time_weighter.fit(original[:, 5], target[:, 5])

        self._phsp_weighter = GBReweighter(
            # n_estimators=650, max_depth=6, learning_rate=0.2, min_samples_leaf=800
            # n_estimators=10
        )

        # Weight original -> target, but weight the target such that
        self._phsp_weighter.fit(
            original=original,
            target=target,
            target_weight=self._time_weighter.apply_efficiency(target[:, 5]),
        )

    def time_weights(self, times: np.ndarray) -> np.ndarray:
        """
        Find weights to take mc -> AmpGen

        """
        return self._time_weighter.correct_efficiency(times)

    def phsp_weights(self, phsp_points: np.ndarray) -> np.ndarray:
        """
        Find the weights needed to reweight phsp part of mc -> AmpGen

        """
        return self._phsp_weighter.predict_weights(phsp_points)

    def weights(self, points):
        return self.phsp_weights(points) * self.time_weights(points[:, 5])


class Binned_Reweighter:
    """
    Several Efficiency_Weighters that each operate on a separate subset of the data

    """

    def _create_reweighter(self, i):
        """Create the i'th reweighter and populate the list of reweighters with it"""
        print(f"creating {i}")
        self._reweighters[i] = Efficiency_Weighter(
            self._ag_points[self._ag_indices == i],
            self._mc_points[self._mc_indices == i],
        )

    def _create_reweighters(self):
        """Create all the reweighters, add them to a list"""
        procs = [
            Process(target=self._create_reweighter, args=(i,))
            for i in range(self._n_bins)
        ]

        # Only run 8 processes at a time
        n_parallel = 8
        n_done = 0
        while n_done < self._n_bins:
            for p in procs[n_done : n_done + n_parallel]:
                p.start()
            for p in procs[n_done : n_done + n_parallel]:
                p.join()

            # Check exit codes
            for p in procs[n_done : n_done + n_parallel]:
                if p.exitcode > 0:
                    raise TrainingError

            n_done += n_parallel

    def _check_underflow(self, indices: np.ndarray) -> None:
        """
        Check bin indices for underflow

        """
        if (indices < 0).any():
            raise ValueError("underflow")

    def _check_overflow(self, indices: np.ndarray) -> None:
        """
        Check bin indices for overflow

        """
        if (indices == (self._n_bins)).any():
            raise ValueError("overflow")

    def __init__(
        self, time_bins: np.ndarray, mc_points: np.ndarray, ag_points: np.ndarray
    ):
        """
        Using the provided time bins, create an Efficiency_Weighter in each

        Bins should cover the entire range of interest - from min time (leftmost edge)
        to the max time (rightmost edge)

        """
        self._mc_points = mc_points
        self._ag_points = ag_points

        self._n_bins = len(time_bins) - 1
        self._reweighter_bins = time_bins

        # Bin times
        self._mc_indices = np.digitize(mc_points[:, 5], self._reweighter_bins) - 1
        self._ag_indices = np.digitize(ag_points[:, 5], self._reweighter_bins) - 1

        # Check if any times have over/underflowed our bins
        self._check_underflow(self._mc_indices)
        self._check_underflow(self._ag_indices)
        self._check_overflow(self._mc_indices)
        self._check_overflow(self._ag_indices)

        # Check that all bins are populated
        assert (
            len(np.unique(self._mc_indices)) == self._n_bins
        ), "some bins contain no MC"
        assert (
            len(np.unique(self._ag_indices)) == self._n_bins
        ), "some bins contain no ampgen"

        # Find the overall scaling we need to apply to the points
        _, mc_counts = np.unique(self._mc_indices, return_counts=True)
        _, ag_counts = np.unique(self._ag_indices, return_counts=True)

        self._ratios = ag_counts / mc_counts
        self._ratios /= np.mean(self._ratios)

        # Create reweighters
        m = Manager()
        self._reweighters = m.list([None for _ in range(self._n_bins)])
        self._create_reweighters()

        # Convert manager list to a regular list so we can pickle an instance of this class safely
        self._reweighters = list(self._reweighters)

    def hist(self):
        """
        Plot histogram of MC time population in the provided bins; saves to bins.png

        """
        sub_bins = np.linspace(0.0, np.max(self._mc_points[:, 5]), 1000)
        fig, ax = plt.subplots()
        for i in range(self._n_bins):
            ax.hist(
                self._mc_points[:, 5][self._mc_indices == i], bins=sub_bins, alpha=0.5
            )
        fig.savefig("bins.png")
        plt.close(fig)

    def weights(self, points):
        # Find indices
        indices = np.digitize(points[:, 5], self._reweighter_bins) - 1

        # Init array of 0s
        # This ensures that any points outside of our time bins get assigned a weight of exactly 0.0
        w = np.zeros(len(points))

        for i in range(self._n_bins):
            tmp = self._reweighters[i].weights(points[indices == i])

            # Multiply by the ratio to get the correct relative weighting
            w[indices == i] = tmp * self._ratios[i] / np.mean(tmp)

        return w

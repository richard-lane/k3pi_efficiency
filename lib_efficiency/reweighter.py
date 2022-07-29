import sys
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
from hep_ml.reweight import GBReweighter, BinsReweighter


class TrainingError(Exception):
    def __init__(self):
        """
        Error training the BDT or something

        """
        super().__init__(
            "Error training the BDT; see above for child process tracebacks"
        )


class Efficiency_Weighter:
    """
    Holds a 5d BDT and information about the decay times

    Used for approximating the efficiency

    """

    def __init__(self, target: np.ndarray, original: np.ndarray):
        """
        Train BDT, perform histogram division to set all the class members

        First performs histogram division, then trains BDT to reweight mc -> AmpGen * e(t)

        :param target: shape (N, 6) array of phsp points; 6th element of each should be decay time
        :param original: shape (N, 6) array of phsp points; 6th element of each should be decay time

        """
        self._time_weighter = BinsReweighter()
        # This is a bit strange - we're reweighting AmpGen -> MC to avoid getting really big weights
        self._time_weighter.fit(target=original[:, 5], original=target[:, 5])

        self._phsp_weighter = GBReweighter(
            # n_estimators=650, max_depth=6, learning_rate=0.2, min_samples_leaf=800
            # n_estimators=10
        )

        # TODO if we do the weighting the other way, need to weight the target not the original
        self._phsp_weighter.fit(
            original=original,
            target=target,
            target_weight=self._time_weighter.predict_weights(target[:, 5]),
        )

    def time_weights(self, times: np.ndarray) -> np.ndarray:
        """
        Find weights to take mc -> AmpGen

        """
        return 1 / self._time_weighter.predict_weights(times)

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

        Bins should cover the entire range of interest - from min time (leftmost edge) to the max time (rightmost edge)

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

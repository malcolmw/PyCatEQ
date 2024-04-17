import functools
import numpy as np
import pykonal
import scipy.optimize


def get_arrival_dict(event_id, arrivals):
    arrivals = arrivals.loc[event_id]
    return dict(zip(arrivals.index, arrivals['time']))


class EQLocator:
    """
    EQLocator(stations, tt_inv, coord_sys='spherical')

    A class to locate earthquakes.
    """
    def __init__(self, traveltime_inventory: str):
        self._arrivals = dict()
        self._traveltimes = dict()
        self._residual_rvs = dict()
        self._tt_inv = pykonal.inventory.TraveltimeInventory(
            traveltime_inventory,
            mode="r"
        )


    def __del__(self) -> None:
        self._tt_inv.f5.close()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()


    def add_arrivals(self, arrivals: dict) -> bool:
        self._arrivals = {**self.arrivals, **arrivals}
        return True

    def clear_arrivals(self) -> bool:
        self.arrivals = {}
        return True


    @property
    def arrivals(self) -> dict:
        return self._arrivals

    @arrivals.setter
    def arrivals(self, value: dict):
        self._arrivals = value


    @property
    def tt_inv(self) -> object:
        return self._tt_inv


    @property
    def residual_rvs(self) -> dict:
        return self._residual_rvs

    @residual_rvs.setter
    def residual_rvs(self, value: dict):
        self._residual_rvs = value


    @property
    def traveltimes(self) -> dict:
        return self._traveltimes

    @traveltimes.setter
    def traveltimes(self, value: dict):
        self._traveltimes = value


    def read_traveltimes(self, min_coords=None,  max_coords=None) -> bool:

        tt_inv = self.tt_inv
        self.traveltimes = {
            index: tt_inv.read(
                "/".join(index),
                min_coords=min_coords,
                max_coords=max_coords
            ) for index in self.arrivals
        }

        return True


    def locate(self, initial, delta, norm='l2'):
        """
        Locate event using a grid search and Differential Evolution
        Optimization to minimize the residual RMS.
        """

        min_coords = initial - delta
        max_coords = initial + delta
        bounds = np.stack([min_coords, max_coords]).T
        if norm.upper() == 'EDT': bounds = bounds[:3]

        self.read_traveltimes(
            min_coords=min_coords[:3],
            max_coords=max_coords[:3]
        )

        soln = scipy.optimize.differential_evolution(
            functools.partial(self.likelihood, norm=norm, const=-1),
            bounds
        )
        if norm.upper() == 'EDT':
            t0 = self.estimate_origin_time(soln.x)
            x = np.concatenate([soln.x, [t0]])
        else:
            x = soln.x

        return x

    def estimate_origin_time(self, hypocenter):
        keys = list(self.arrivals.keys())
        t_obs = np.array([self.arrivals[key] for key in keys])
        tt = np.array([
            self.traveltimes[key].value(hypocenter, null=np.inf) for key in keys
        ])
        return np.median(t_obs - tt)

    def likelihood(self, hypocenter, norm='L2', const=1):
        '''
        Returns the likelihood for a set of hypocenter coordinates.
        **Note that the EDT likelihood is independent of origin time!

        Arguments
        =========
        hypocenter :: np.array(shape=(4,)) :: Hypocenter coordinates
            to compute likelihood for.

        Keyword arguments
        =================
        norm :: str :: Specifies whether to use the least-squares L2
            likelihood ('L2') or the Equalt Differential Time likelihood
            ('EDT'). Default value is 'L2'.
        const :: float :: Constant by which to scale likelihood. Set to
            -1 to return negative likelihood for optimization. Default
            value is 1.

        Returns
        =======
        likelihood :: float :: Likelihood of specifid hypocenter
            coordinates, scaled by `const`.
        '''
        arrivals = self.arrivals
        traveltimes = self.traveltimes
        keys = list(arrivals.keys())
        t_obs = np.array([arrivals[key] for key in keys])
        tt = np.array([
            traveltimes[key].value(hypocenter[:3], null=np.inf) for key in keys
        ])
        if np.any(np.isinf(tt)):
            return -const * np.inf
        if norm.upper() == 'L2':
            t0 = hypocenter[3]
            return const * np.exp(-1 / 2 * np.sum(np.square(t0 + tt - t_obs)))
        elif norm.upper() == 'EDT':
            t_obs_a = np.repeat(t_obs, len(t_obs))
            t_obs_b = np.tile(t_obs, len(t_obs))
            tt_a = np.repeat(tt, len(tt))
            tt_b = np.tile(tt, len(tt))
            p = np.square((t_obs_a - t_obs_b) - (tt_a - tt_b))
            p = np.exp(-p / 2)
            p = p / np.sqrt(2)
            p = np.sum(p)
            p = np.power(p, len(keys))
            return const * p
        else: raise NotImplementedError(
            f'Unrecognized norm {norm}. Choose one of \'EDT\' or \'L2\'.'
        )

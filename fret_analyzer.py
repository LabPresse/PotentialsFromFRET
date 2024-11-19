
# Import libraries
import time
import copy
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace
from matplotlib import gridspec
from joblib import Parallel, delayed
from algorithms import HistoryH5


# Kernel function
@nb.njit(cache=True)
def kernel_FRET(x1, x2, sig, ell, d_dx=False, dd_ddx=False):

    sig2 = sig ** 2
    ell2 = ell ** 2
    ell4 = ell ** 4
    num_1, num_dims = x1.shape
    num_2, ________ = x2.shape

    if not (d_dx or dd_ddx):
        K = np.zeros((num_1, num_2))
        for i in range(num_1):
            for j in range(num_2):
                K[i, j] = sig2 * np.exp(-.5 * np.sum((x1[i, :] - x2[j, :]) ** 2) / ell2)
    else:
        K = np.zeros((num_1 * num_dims, num_2))
        for i in range(num_1):
            for j in range(num_2):
                Kij = sig2 * np.exp(-.5 * np.sum((x1[i, :] - x2[j, :]) ** 2) / ell2)
                for d in range(num_dims):
                    if d_dx:
                        K[i + num_1 * d, j] = - Kij * (x1[i, d] - x2[j, d]) / ell2
                    elif dd_ddx:
                        K[i + num_1 * d, j] = - Kij * (ell2 - (x1[i, d] - x2[j, d]) ** 2) / ell4

    return K

# Units are nanoseconds (ns), nanometers (nm), attograms (ag)
PARAMETERS = {
    # Variables
    'P': None,          # (#)      Probability
    'z': 5000,          # (ag/ns)  Friction
    'kx': None,         # (1/ns)   Excitation rate
    'ka': None,         # (1/ns)   Acceptor background rate
    'kd': None,         # (1/ns)   Donor background rate
    'f_data': None,     # (pN)     Force at data points
    'u_indu': None,     # (pN*nm)  Potential at inducing points
    'x_data': None,     # (nm)     Positions at data time levels

    # Experiment
    'dt': 1,              # (ns)     Time step
    'R0': 5,              # (nm)     FRET distance
    'kT': 4.114,          # (pN*nm)  Experiment temperature
    'traj_mask': None,    # (#)      Index of trajectories
    'crosstalk': None,    # (#)      Cross talk matrix (cols from, rows to)
    'detectoreff': None,  # (#)      Detector efficiency matrix (cols from, rows to)

    # Priors
    'u_indu_fixed': None,  # Hyperparameter
    'x0_mean': None,       # Hyperparameter
    'x0_vars': None,       # Hyperparameter
    'z_shape': 2,          # Hyperparameter
    'z_scale': None,       # Hyperparameter
    'kx_shape': 2,         # Hyperparameter
    'kx_scale': None,      # Hyperparameter
    'kd_shape': 2,         # Hyperparameter
    'kd_scale': None,      # Hyperparameter
    'ka_shape': 2,         # Hyperparameter
    'ka_scale': None,      # Hyperparameter

    # Covariance matrix
    'sig': 1,             # Hyperparameter for force
    'ell': 1,              # Hyperparameter for force (in units of nm)
    'eps': .01,            # Numerical stability parameter for matrix inversion
    'x_grid': None,        # Positions of grid points
    'x_indu': None,        # Positions of inducing points
    'K_grid_indu': None,        # Covariances between potential at grid and potential at indu
    'K_indu_indu': None,        # Covariances between potential at indu and potential at indu
    'K_indu_indu_inv': None,
    'K_indu_indu_unfixed_inv': None,  # Covariance between potential at indu given fixed indu

    # Ground truth parameters
    'force': None,
    'potential': None,

    # Numbers
    'num_data': None,  # Number of time levels
    'num_dims': 1,     # Number of data dimensions
    'num_traj': None,  # Number of trajectories
    'num_indu': 100,   # Number of inducing points
    'num_grid': 1000,  # Number of grid points

    # Sampler parameters
    'seed': 0,             # Random number generator seed
    'z_prop_shape': 100,         # Proposal distribution shape for friction
    'x_prop_shape': 100,         # Proposal distribution width (STD) for position
    'kx_prop_shape': 100,        # Proposal distribution shape for excitation
    'kd_prop_shape': 100,        # Proposal distribution shape for background
    'ka_prop_shape': 100,        # Proposal distribution shape for background
    'parallelize': False,  # If true then parallelize the sampler
}


class FRETAnalyzer:

    @staticmethod
    def simulate_data(parameters=None, **kwargs):

        # Set default parameters
        default_parameters = {
            # Variables
            'z': 5000,             # (ag/ns))       Friction
            'kx': 10,              # (1/ns)         Fluorophore excitation rate
            'ka': .5,              # (1/ns)         Acceptor photon background rate
            'kd': 0,               # (1/ns)         Donor photon background rate
            'force': None,         # (pN)           Force function
            # Constants
            'crosstalk': np.array([[0.75, 0.00], [0.25, 1.00]]),  # Cross talk matrix
            'detectoreff': np.array([1, .34]),                    # Detection efficiency
            'dt': 1,               # (ns)           Chosen time step for Langevin dynamics
            'kT': 4.114,           # (pN*nm)        Experiment temperature
            'R0': 5,               # (nm)           FRET distance
            'num_data': None,      # Number of time levels
            'num_data_per': 100,   # Number of time levels per trajectory
            'num_dims': 1,         # Number of data dimensions
            'num_traj': 10,        # Number of trajectories
            'seed': 0,             # RNG seed
        }

        # Set parameters
        if parameters is None:
            parameters = {**default_parameters, **kwargs}
        else:
            parameters = {**default_parameters, **parameters, **kwargs}
        if parameters['num_data'] is None:
            parameters['num_data'] = parameters['num_traj'] * parameters['num_data_per']

        # Set up variables
        variables = FRETAnalyzer.initialize_variables(None, parameters)
        z = variables.z
        kx = variables.kx
        ka = variables.ka
        kd = variables.kd
        force = variables.force
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_traj = variables.num_traj
        num_data_per = variables.num_data_per
        seed = variables.seed
        
        # Set rng
        np.random.seed(seed)

        # Calculate values
        kick = 2 * dt * kT / z
        f = force
        if f is None:
            V = 2 * kT  # well depth
            s = R0  # location of minimum
            f = lambda x_: 4 * V * (3 * (s ** 12) / (x_ ** 13) - 3 * (s ** 6) / (x_ ** 7))
            U = lambda x_: -kT * 4 * V * (3 * (s ** 12) / (12 * x_ ** 12) - 3 * (s ** 6) / (6 * x_ ** 6))

        # Sample trajectory
        x = np.zeros((num_data, num_dims))
        traj_mask = np.zeros(num_data, dtype=int)
        for t in range(num_traj):
            # Find start id
            id = t * num_data_per
            traj_mask[id:id + num_data_per] = t
            # Sample trajectory
            x_mean = np.linspace(.25*R0, 5*R0)[np.argmin(-np.cumsum(f(np.linspace(.25*R0, 5*R0))))]   # start at minimum of potential
            x[id, :] = stats.truncnorm.rvs(a=-x_mean, b=np.inf, loc=x_mean, scale=R0/1000)
            for n in range(1, num_data_per):
                x_mean = x[id + n - 1, :] + dt / z * f(x[id + n - 1, :])
                x[id + n, :] = stats.truncnorm.rvs(a=-x_mean, b=np.inf, loc=x_mean, scale=np.sqrt(kick))

        # Sample data
        data = np.zeros((num_data, 2))
        for n in range(num_data):
            FRET = 1 / (1 + (x[n] / R0) ** 6)
            mu = dt * detectoreff @ crosstalk @ np.array([kx*(1-FRET) + kd, kx*FRET + ka])
            data[n, 0] = stats.poisson.rvs(mu=mu[0])  # background donor
            data[n, 1] = stats.poisson.rvs(mu=mu[1])  # background acceptor

        # Update parameters
        variables.x = x
        variables.traj_mask = traj_mask
        variables.force = f
        variables.potential = U
        variables = variables.__dict__  # Convert to dictionary

        return data, variables

    @staticmethod
    def initialize_variables(data, parameters):

        # Eet up variables
        variables = SimpleNamespace(**parameters)

        # Extract variables
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        sig = variables.sig
        eps = variables.eps
        ell = variables.ell
        z = variables.z
        kx = variables.kx
        ka = variables.ka
        kd = variables.kd
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars
        kx_scale = variables.kx_scale
        kd_scale = variables.kd_scale
        ka_scale = variables.ka_scale
        kx_shape = variables.kx_shape
        ka_shape = variables.ka_shape
        kd_shape = variables.kd_shape
        z_shape = variables.z_shape
        crosstalk = variables.crosstalk
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_indu = variables.num_indu
        num_grid = variables.num_grid
        traj_mask = variables.traj_mask
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff

        # Data required variables
        if data is not None:

            # Data shape
            num_data, _ = data.shape
            variables.num_data = num_data

            # Trajectory mask
            if traj_mask is None:
                traj_mask = np.zeros(num_data, dtype=int)
            else:
                traj_mask = traj_mask.astype(int)
            num_traj = np.max(traj_mask) + 1
            variables.traj_mask = traj_mask
            variables.num_traj = num_traj

        # Probability
        P = - np.inf
        variables.P = P

        # Crosstalk matrix
        if crosstalk is None:
            crosstalk = np.eye(2)
        if detectoreff is None:
            detectoreff = np.ones(2)
        variables.crosstalk = crosstalk
        variables.detectoreff = detectoreff

        # Friction
        z_scale = z / z_shape
        variables.z_scale = z_scale

        # Rates
        if kx is None:
            kx = np.mean(data) / dt / np.mean(detectoreff)
        if kx_scale is None:
            kx_scale = kx / kx_shape
        if kd is None:
            kd = np.mean(data) / dt / detectoreff[0]
        if kd_scale is None:
            kd_scale = kd / kd_shape
        if ka is None:
            ka = np.mean(data) / dt / detectoreff[1]
        if ka_scale is None:
            ka_scale = ka / ka_shape
        variables.kx = kx + 1e-20
        variables.kx_scale = kx_scale
        variables.kd = kd + 1e-20
        variables.kd_scale = kd_scale
        variables.ka = ka + 1e-20
        variables.ka_scale = ka_scale

        # Potential
        f_data = np.zeros((num_data, num_dims))
        u_indu = np.zeros((num_indu ** num_dims, 1))
        u_grid = np.zeros((num_grid ** num_dims, 1))
        variables.f_data = f_data
        variables.u_indu = u_indu
        variables.u_grid = u_grid

        # Positions
        if x0_mean is None:
            x0_mean = 3*R0
        if x0_vars is None:
            x0_vars = (10 * R0) ** 2
        x_data = 3*R0*np.random.rand(num_data, num_dims)
        x_indu = np.zeros((num_indu ** num_dims, num_dims))
        x_grid = np.zeros((num_grid ** num_dims, num_dims))
        u_fixed = np.ones((num_indu ** num_dims), dtype=bool)
        for d in range(num_dims):
            temp_indu = np.linspace(-R0, 5 * R0, num_indu)
            temp_grid = np.linspace(.01*R0, 5 * R0, num_grid)
            x_indu[:, d] = np.tile(np.repeat(temp_indu, num_indu ** (num_dims - d - 1)), num_indu ** d)
            x_grid[:, d] = np.tile(np.repeat(temp_grid, num_grid ** (num_dims - d - 1)), num_grid ** d)
            u_fixed = u_fixed * (temp_indu > 2*R0)
        variables.x0_mean = x0_mean
        variables.x0_vars = x0_vars
        variables.x_data = x_data
        variables.x_indu = x_indu
        variables.x_grid = x_grid
        variables.u_fixed = u_fixed

        # Covariance matrices
        kernel = lambda x1, x2, d_dx=False, dd_ddx=False: kernel_FRET(x1, x2, sig, ell, d_dx=d_dx, dd_ddx=dd_ddx)
        K_grid_indu = kernel(x_grid, x_indu)
        K_indu_indu = kernel(x_indu, x_indu)
        K_indu_indu_inv = np.linalg.inv(K_indu_indu + eps * np.eye(num_indu ** num_dims))
        variables.kernel = kernel
        variables.K_grid_indu = K_grid_indu
        variables.K_indu_indu = K_indu_indu
        variables.K_indu_indu_inv = K_indu_indu_inv
        return variables

    @staticmethod
    def sample_potential(data, variables):

        dt = variables.dt
        kT = variables.kT
        z = variables.z
        eps = variables.eps
        x_data = variables.x_data
        x_indu = variables.x_indu
        kernel = variables.kernel
        u_fixed = variables.u_fixed
        K_indu_indu_inv = variables.K_indu_indu_inv
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_indu = variables.num_indu
        num_traj = variables.num_traj
        traj_mask = variables.traj_mask

        # Calculate displacements
        ids = np.where(traj_mask[1:] == traj_mask[:-1])[0]
        y_data = x_data[1:, :] - x_data[:-1, :]
        y_data = y_data[ids, :].reshape((-1, 1), order='F')

        # Calculate mean and covarince matrix
        K_data_indu = - kT * kernel(x_data[ids, :], x_indu, d_dx=True)
        K_tilde = np.linalg.inv(
            K_indu_indu_inv 
            + dt / (2 * z * kT) * K_indu_indu_inv @ (K_data_indu.T @ K_data_indu) @ K_indu_indu_inv
        )
        K_tilde_chol = np.linalg.cholesky(K_tilde)
        mu_tilde = 1 / (2 * kT) * (K_tilde @ (K_indu_indu_inv @ (K_data_indu.T @ y_data)))

        # Sample potential
        u_indu = mu_tilde + K_tilde_chol @ np.random.randn(num_indu ** num_dims, 1)

        # # Find mean and covariance for unfixed given fixed
        # u_unfixed = ~u_fixed
        # num_fixed = np.sum(u_fixed)
        # num_unfixed = np.sum(u_unfixed)
        # K_tilde_fixed_inv = np.linalg.inv(K_tilde[np.ix_(u_fixed, u_fixed)] + eps*np.eye(num_fixed))
        # mu_tilde_unfixed = (
        #     mu_tilde[u_unfixed] 
        #     - K_tilde[np.ix_(u_unfixed, u_fixed)] @ K_tilde_fixed_inv @ mu_tilde[u_fixed]
        # )
        # K_tilde_unfixed = (
        #     K_tilde[np.ix_(u_unfixed, u_unfixed)] 
        #     - K_tilde[np.ix_(u_unfixed, u_fixed)] @ K_tilde_fixed_inv @ K_tilde[np.ix_(u_fixed, u_unfixed)]
        # )
        # K_tilde_unfixed_chol = np.linalg.cholesky(K_tilde_unfixed)

        # # Sample U
        # u_indu = np.zeros((num_indu ** num_dims, 1))
        # u_indu[u_unfixed, :] = (
        #     mu_tilde_unfixed + K_tilde_unfixed_chol @ np.random.randn(num_unfixed, 1)
        # )

        # Calculate force
        f_data = np.zeros((num_data, num_dims))
        f_data[ids, :] = (K_data_indu @ (K_indu_indu_inv @ u_indu)).reshape((-1, num_dims), order='F')

        # Update variables
        variables.u_indu = u_indu
        variables.f_data = f_data

        return

    @staticmethod
    def sample_trajectory(data, variables):

        if np.random.rand() < .5:
            FRETAnalyzer.sample_trajectory_MH(data, variables)
        else:
            FRETAnalyzer.sample_trajectory_HMC(data, variables)

        return

    @staticmethod
    def sample_trajectory_HMC(data, variables):

        # get variables
        z = variables.z
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars
        x_data = variables.x_data
        x_indu = variables.x_indu
        u_indu = variables.u_indu
        f_data = variables.f_data
        kernel = variables.kernel
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_traj = variables.num_traj
        K_indu_indu_inv = variables.K_indu_indu_inv
        traj_mask = variables.traj_mask
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff
        parallelize = variables.parallelize

        # set up variables
        x_data = x_data.copy()
        f_data = f_data.copy()
        kick = 2 * dt * kT / z
        K_inv_U = K_indu_indu_inv @ u_indu.reshape(-1, 1, order='F')

        def sample_positions(ids, x_data, f_data):

            # Select variables
            n0 = ids[0]
            nf = ids[-1]
            t = traj_mask[n0]
            num_ids = len(ids)
            data_ids = data[ids, :]

            # Set up x0 and kicks
            x_prev = np.zeros((num_ids, num_dims))
            kicks = kick * np.ones((num_ids, num_dims))
            if (n0 == 0) or traj_mask[n0-1] != t:
                x_prev[0, :] = x0_mean
                kicks[0, :] = x0_vars
            else:
                x_prev[0, :] = x_data[n0 - 1, :] + dt / z * f_data[n0 - 1, :]
            x_prev = x_prev.reshape((-1, 1), order='F')
            kicks = kicks.reshape((-1, 1), order='F')

            # Set up x final
            if (nf == num_data-1) or traj_mask[nf+1] != t:
                C = np.zeros((num_ids, num_ids))
                x_next = np.zeros((num_ids, num_dims))
            else:
                C = np.diag([0] * (num_ids - 1) + [1])
                x_next = np.zeros((num_ids, num_dims))
                x_next[-1, :] = x_data[nf + 1, :]
            x_next = x_next.reshape((-1, 1), order='F')
            C = np.kron(np.eye(num_dims), C)

            # set up dynamics matrixes
            B = np.diag(np.ones(num_ids - 1), -1)
            A = np.eye(num_ids) - B
            B = np.kron(np.eye(num_dims), B)
            A = np.kron(np.eye(num_dims), A)

            # HMC proposal distribution
            h = stats.expon.rvs(scale=.001)
            M = np.ones((num_ids * num_dims, 1))
            M_inv = 1 / M
            num_steps = stats.poisson.rvs(mu=50)

            def dp_dh(q_):
                if np.any(q_ <= 0) or np.any(q_ > 1e10):
                    # dont bother with calculations if we know the sample will be rejected
                    y_ = 0
                else:
                    f = (- kT * kernel(q_, x_indu, d_dx=True) @ K_inv_U).reshape((-1, num_dims), order='F')
                    df = np.diag((- kT * kernel(q_, x_indu, dd_ddx=True) @ K_inv_U)[:, 0])
                    B_f = np.vstack([np.zeros((1, 1)), f[:-1, :]])  # faster calculation of B @ f
                    # B_df = np.diag(df[:-1, 0], -1)  # faster calculation of B @ df
                    FRET = 1 / (1 + (q_[:, 0] / R0) ** 6)
                    dFRET_dx = -6 * q_[:, 0] ** 5 / (R0 ** 6) * FRET ** 2
                    mu = dt * np.diag(detectoreff) @ crosstalk @ np.vstack([kx*(1-FRET) + kd, kx*FRET + ka])
                    dmu_dx = dt * np.diag(detectoreff) @ crosstalk @ np.vstack([-kx*dFRET_dx, kx*dFRET_dx])
                    y_ = (
                        (dmu_dx[0, :] * (1 - data_ids[:, 0] / mu[0, :]))[:, None]
                        + (dmu_dx[1, :] * (1 - data_ids[:, 1] / mu[1, :]))[:, None]
                        - (A - dt / z * B @ df).T @ (A @ q_ - dt / z * B_f - x_prev) / kicks
                        - (C + dt / z * C @ df).T @ (C @ (q_ + dt / z * f) - x_next) / kicks
                    )
                    if np.any(np.isnan(y_)):
                        print('ohno')
                return y_

            def probability(q_, p_):
                if np.any(q_ <= 0) or np.any(q_ > 1e10):
                    prob = -np.inf
                else:
                    f = (- kT * kernel(q_, x_indu, d_dx=True) @ K_inv_U).reshape((-1, num_dims), order='F')
                    FRET = 1 / (1 + (q_[:, 0] / R0) ** 6)
                    mu = dt * np.diag(detectoreff) @ crosstalk @ np.vstack([kx*(1-FRET) + kd, kx*FRET + ka])
                    prob = (
                        np.sum(stats.poisson.logpmf(data_ids[:, 0], mu[0, :]))
                        + np.sum(stats.poisson.logpmf(data_ids[:, 1], mu[1, :]))
                        + np.sum(stats.norm.logpdf(q_, loc=B @ (q_ + dt / z * f) + x_prev, scale=np.sqrt(kicks)))  # prior on x
                        + np.sum(stats.norm.logpdf(x_next, loc=C @ (q_ + dt / z * f), scale=np.sqrt(kicks)))       # prior on final x
                        + np.sum(stats.norm.logpdf(p_, loc=0, scale=np.sqrt(M)))                                   # prior on p
                    )
                return prob

            # run HMC
            q = x_data[ids, :].copy().reshape((-1, 1), order='F')
            p = stats.norm.rvs(loc=np.zeros(M.shape), scale=np.sqrt(M))
            P_old = probability(q, p)
            for _ in range(num_steps):
                p = p + h / 2 * dp_dh(q)
                q = q + h * p * M_inv
                p = p + h / 2 * dp_dh(q)
            P_new = probability(q, p)

            # accept or reject
            acc_prob = P_new - P_old
            if acc_prob > np.log(np.random.rand()):
                x_data_ids = q[:, :].reshape((-1, num_dims), order='F')
                f_data_ids = (- kT * kernel(x_data_ids, x_indu, d_dx=True) @ K_inv_U).reshape((-1, num_dims), order='F')
            else:
                x_data_ids = x_data[ids, :].copy()
                f_data_ids = f_data[ids, :].copy()

            return ids, x_data_ids, f_data_ids

        # Sample trajectories
        acceptance_rate = np.zeros(2)
        for t in range(num_traj):

            # Split trajectory into sections
            num_per_sec = 10
            idt = np.where(traj_mask == t)[0]
            section_ids = np.zeros(len(idt), dtype=int)
            section_ids[range(0, len(idt), num_per_sec)] += 1
            section_ids[:] = np.cumsum(section_ids) - 1
            num_sections = section_ids[-1]

            # Loop through sampling two sections at a time while leaving the third constant
            for i in range(3):
                sections = [np.where((section_ids == j) | (section_ids == j + 1))[0] for j in range(i, num_sections, 3)]
                
                # Sample each section
                if parallelize:
                    results = Parallel(n_jobs=-1)(
                        delayed(sample_positions)(ids, x_data, f_data) for ids in sections
                    )
                else:
                    results = [sample_positions(ids, x_data, f_data) for ids in sections]

                # Update positions
                for ids, x_new, f_new in results:
                    acceptance_rate += x_new.size, np.sum(x_new != x_data[ids, :])
                    x_data[ids, :] = x_new
                    f_data[ids, :] = f_new

        # update variables
        variables.x_data = x_data
        variables.f_data = f_data

        # print acceptance ratio
        print('(Hx{}%)'.format(round(100 * acceptance_rate[1]/acceptance_rate[0])), end='')

        return

    @staticmethod
    def sample_trajectory_MH(data, variables):
        
        # Extract variables
        z = variables.z
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars
        x_data = variables.x_data
        x_indu = variables.x_indu
        x_prop_shape = variables.x_prop_shape
        f_data = variables.f_data
        u_indu = variables.u_indu
        kernel = variables.kernel
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_traj = variables.num_traj
        K_indu_indu_inv = variables.K_indu_indu_inv
        traj_mask = variables.traj_mask
        parallelize = variables.parallelize
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff

        # Initialize variables
        x_data = x_data.copy()
        f_data = f_data.copy()
        kick = 2 * dt * kT / z
        K_inv_U = K_indu_indu_inv @ u_indu.reshape(-1, 1, order='F')

        # Create sample function
        def sample_position(n, x_data, f_data):

            # Select variables
            t = traj_mask[n]
            if (n == 0) or (traj_mask[n-1] != t):
                mu_prev = x0_mean
                sigma_prev = np.sqrt(x0_vars)
            else:
                mu_prev = x_data[n-1, :] + dt/z*f_data[n-1, :]
                sigma_prev = np.sqrt(kick)
            if (n == num_data-1) or traj_mask[n+1] != t:
                x_next = None
            else:
                x_next = x_data[n+1, :]

            # Create probability function
            def probability(x_, f_):

                # No x should be less than 0
                if np.any(x_ <= 0):
                    prob = -np.inf
                    return prob

                # Compute likelihood
                FRET = 1 / (1 + (x_ / R0) ** 6)
                mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx*(1-FRET) + kd, kx*FRET + ka])
                lhood = (
                    np.sum(stats.poisson.logpmf(data[n, 0], mu[0]))
                    + np.sum(stats.poisson.logpmf(data[n, 1], mu[1]))
                )

                # Compute prior
                prior = np.sum(stats.norm.logpdf(x_, loc=mu_prev, scale=sigma_prev))
                if x_next is not None:
                    prior += np.sum(stats.norm.logpdf(x_next, loc=x_+dt/z*f_, scale=np.sqrt(kick)))

                prob = lhood + prior
                return prob

            # Get old positions
            x_old = x_data[n, :].copy()
            f_old = f_data[n, :].copy()

            # Sample new positions multiple times
            for _ in range(10):
                x_new = stats.gamma.rvs(a=x_prop_shape, scale=x_old/x_prop_shape, size=x_old.size)
                f_new = (- kT * kernel(x_new[None, :], x_indu, d_dx=True) @ K_inv_U)[:, 0]
                acc_prob = (
                    probability(x_new, f_new)
                    - probability(x_old, f_old)
                    + np.sum(stats.gamma.logpdf(x_old, a=x_prop_shape, scale=x_new/x_prop_shape))
                    - np.sum(stats.gamma.logpdf(x_new, a=x_prop_shape, scale=x_old/x_prop_shape))
                )
                if acc_prob > np.log(np.random.rand()):
                    x_old = x_new
                    f_old = f_new

            return n, x_old, f_old

        # Paralellize sampling trajectories
        ratio = np.zeros(2)  # total, accepted
        for m in range(2):
            if parallelize:
                results = Parallel(n_jobs=-1)(
                    delayed(sample_position)(n, x_data, f_data) for n in range(m, num_data, 2)
                )
            else:
                results = [sample_position(n, x_data, f_data) for n in range(m, num_data, 2)]
            for n, x_new, f_new in results:
                ratio += x_new.size, np.sum(x_new != x_data[n, :])
                x_data[n, :] = x_new
                f_data[n, :] = f_new

        # Update variables
        variables.x_data = x_data
        variables.f_data = f_data

        # Print acceptance ratio
        print('(Mx{}%)'.format(round(100 * ratio[1]/ratio[0])), end='')

        return

    @staticmethod
    def sample_friction(data, variables):
        
        # Extract variables
        z = variables.z
        dt = variables.dt
        kT = variables.kT
        x_data = variables.x_data
        f_data = variables.f_data
        traj_mask = variables.traj_mask
        num_traj = variables.num_traj
        z_shape = variables.z_shape
        z_scale = variables.z_scale
        z_prop_shape = variables.z_prop_shape

        # Set up probability
        def probability(z_):
            prob = stats.gamma.logpdf(z_, a=z_shape, scale=z_scale)  # prior
            for t in range(num_traj):
                ids = np.where(traj_mask == t)[0]
                dx = x_data[ids[1:], :] - x_data[ids[:-1], :]
                f = f_data[ids[:-1], :]
                kick = 2 * dt * kT / z_
                prob += (
                    np.sum(stats.norm.logpdf(dx, loc=dt*f/z_, scale=np.sqrt(kick)))  # lhood
                )
            return prob

        # Sample many times
        for _ in range(10):
            z_old = z
            z_new = stats.gamma.rvs(a=z_prop_shape, scale=z_old/z_prop_shape)
            acc_prob = (
                probability(z_new)
                - probability(z_old)
                + stats.gamma.logpdf(z_old, a=z_prop_shape, scale=z_new/z_prop_shape)
                - stats.gamma.logpdf(z_new, a=z_prop_shape, scale=z_old/z_prop_shape)
            )
            if acc_prob > np.log(np.random.rand()):
                z = z_new

        # Update variables
        variables.z = z

        return

    @staticmethod
    def sample_rates(data, variables):
        
        # Extract variables
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        x_data = variables.x_data
        f_data = variables.f_data
        traj_mask = variables.traj_mask
        num_traj = variables.num_traj
        kx_shape = variables.kx_shape
        kx_scale = variables.kx_scale
        kx_prop_shape = variables.kx_prop_shape
        kd_shape = variables.kd_shape
        kd_scale = variables.kd_scale
        kd_prop_shape = variables.kd_prop_shape
        ka_shape = variables.ka_shape
        ka_scale = variables.ka_scale
        ka_prop_shape = variables.ka_prop_shape
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff

        # Caclulate constants
        FRET = 1 / (1 + (x_data[:, 0] / R0) ** 6)

        # Sample kx
        def probability(kx_):
            mu = dt * np.diag(detectoreff) @ crosstalk @ np.vstack([kx_*(1-FRET) + kd, kx_*FRET + ka])
            prob = (
                stats.gamma.logpdf(kx_, a=kx_shape, scale=kx_scale)
                + np.sum(stats.poisson.logpmf(data[:, 0], mu[0, :]))
                + np.sum(stats.poisson.logpmf(data[:, 1], mu[1, :])) 
            )
            return prob
        for _ in range(10):
            kx_old = kx
            kx_new = stats.gamma.rvs(a=kx_prop_shape, scale=kx_old/kx_prop_shape)
            acc_prob = (
                probability(kx_new)
                - probability(kx_old)
                + stats.gamma.logpdf(kx_old, a=kx_prop_shape, scale=kx_new/kx_prop_shape)
                - stats.gamma.logpdf(kx_new, a=kx_prop_shape, scale=kx_old/kx_prop_shape)
            )
            if acc_prob > np.log(np.random.rand()):
                kx = kx_new

        # Sample kd
        if kd_scale > 0:
            def probability(kd_):
                mu = dt * np.diag(detectoreff) @ crosstalk @ np.vstack([kx*(1-FRET) + kd_, kx*FRET + ka])
                prob = (
                    stats.gamma.logpdf(kd_, a=kd_shape, scale=kd_scale)
                    + np.sum(stats.poisson.logpmf(data[:, 0], mu[0, :]))
                    + np.sum(stats.poisson.logpmf(data[:, 1], mu[1, :])) 
                )
                return prob
            for _ in range(10):
                kd_old = kd
                kd_new = stats.gamma.rvs(a=kd_prop_shape, scale=kd_old/kd_prop_shape)
                acc_prob = (
                    probability(kd_new)
                    - probability(kd_old)
                    + stats.gamma.logpdf(kd_old, a=kd_prop_shape, scale=kd_new/kd_prop_shape)
                    - stats.gamma.logpdf(kd_new, a=kd_prop_shape, scale=kd_old/kd_prop_shape)
                )
                if acc_prob > np.log(np.random.rand()):
                    kd = kd_new
        
        # Sample ka
        if ka_scale > 0:
            def probability(ka_):
                mu = dt * np.diag(detectoreff) @ crosstalk @ np.vstack([kx*(1-FRET) + kd, kx*FRET + ka_])
                prob = (
                    stats.gamma.logpdf(ka_, a=ka_shape, scale=ka_scale)
                    + np.sum(stats.poisson.logpmf(data[:, 0], mu[0, :]))
                    + np.sum(stats.poisson.logpmf(data[:, 1], mu[1, :])) 
                )
                return prob
            for _ in range(10):
                ka_old = ka
                ka_new = stats.gamma.rvs(a=ka_prop_shape, scale=ka_old/ka_prop_shape)
                acc_prob = (
                    probability(ka_new)
                    - probability(ka_old)
                    + stats.gamma.logpdf(ka_old, a=ka_prop_shape, scale=ka_new/ka_prop_shape)
                    - stats.gamma.logpdf(ka_new, a=ka_prop_shape, scale=ka_old/ka_prop_shape)
                )
                if acc_prob > np.log(np.random.rand()):
                    ka = ka_new

        # Update variables
        variables.kx = kx
        variables.kd = kd
        variables.ka = ka

        return

    @staticmethod
    def posterior(data, variables, **kwargs):

        # Incorporate kwarg args into variables
        variables = copy.copy(variables)
        for key, val in kwargs.items():
            setattr(variables, key, val)

        # Extract variables
        z = variables.z
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        eps = variables.eps
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars
        z_shape = variables.z_shape
        z_scale = variables.z_scale
        kx_shape = variables.kx_shape
        kx_scale = variables.kx_scale
        kd_shape = variables.kd_shape
        kd_scale = variables.kd_scale
        ka_shape = variables.ka_shape
        ka_scale = variables.ka_scale
        u_indu = variables.u_indu
        f_data = variables.f_data
        x_data = variables.x_data
        x_indu = variables.x_indu
        K_indu_indu = variables.K_indu_indu
        num_traj = variables.num_traj
        num_indu = variables.num_indu
        traj_mask = variables.traj_mask
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff

        # Calculate constants
        kick = 2 * dt * kT / z

        # Prior
        prior = (
            stats.gamma.logpdf(z, a=z_shape, scale=z_scale)
            + stats.gamma.logpdf(kx, a=kx_shape, scale=kx_scale)
            + stats.multivariate_normal.logpdf(
                u_indu[:, 0],
                np.zeros(num_indu),
                cov=(K_indu_indu+eps*np.eye(num_indu))
            )
        )
        if kd_scale > 0:
            prior += stats.gamma.logpdf(kd, a=kd_shape, scale=kd_scale)
        if ka_scale > 0:
            prior += stats.gamma.logpdf(ka, a=ka_shape, scale=ka_scale)

        # Dynamics
        dynamics = 0
        for t in range(num_traj):
            ids = np.where(traj_mask == t)[0]
            x0 = x_data[ids[0], :]
            dx = x_data[ids[1:], :] - x_data[ids[:-1], :]
            f = f_data[ids[:-1], :]
            dynamics += (
                np.sum(stats.norm.logpdf(x0, loc=x0_mean, scale=np.sqrt(x0_vars)))  # prior on x[0, :]
                + np.sum(stats.norm.logpdf(dx, loc=dt*f/z, scale=np.sqrt(kick)))    # prior on x[1:, :]
            )
        
        # Likelihood
        FRET = 1 / (1 + (x_data[:, 0] / R0) ** 6)
        mu = dt * np.diag(detectoreff) @ crosstalk @ np.vstack([kx*(1-FRET) + kd, kx*FRET + ka])
        lhood = (
            np.sum(stats.poisson.logpmf(data[:, 0], mu[0, :]))
            + np.sum(stats.poisson.logpmf(data[:, 1], mu[1, :]))
        )

        # Retrun probability
        prob = prior + dynamics + lhood
        # print(f'prior={round(prior)},dynamics={round(dynamics)},lhood={round(lhood)},prob={round(prob)}')

        return prob

    @staticmethod
    def plot_data(data, inputs=None, ax=None):

        # Check data shape
        if data.shape[1] > data.shape[0]:
            data = data.T
        num_data = data.shape[0]

        # Check inputs
        if inputs is None:
            # In no input is specified, use the default inputs
            variables = None
            dt = 1
            num_traj = 1
            traj_mask = np.zeros(num_data, dtype=int)
        else:
            # Extract variables
            if type(inputs) == dict:
                variables = SimpleNamespace(inputs)
            elif type(inputs) == SimpleNamespace:
                variables = inputs
            elif type(inputs) == HistoryH5:
                variables = inputs.get('MAP')
            dt = variables.dt
            num_traj = variables.num_traj
            traj_mask = variables.traj_mask

        # Calculate values
        times = np.arange(num_data) * dt * 1e-9

        # Set up ax
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Data
        for t in range(num_traj):
            idt = np.where(traj_mask == t)[0]
            ax.step(times[idt], data[idt, 0], color='g', where='pre', label='Green photons')
            ax.step(times[idt], data[idt, 1], color='r', where='pre', label='Red photons')
            if num_traj > 1:
                ax.axvline(times[idt[0]], color='k', linewidth=2, label='New trajectory')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        return ax

    @staticmethod
    def plot_trajectory(data, inputs, groundtruth=None, ax=None,):

        # Check inputs
        if type(inputs) == dict:
            variables = SimpleNamespace(inputs)
        elif type(inputs) == SimpleNamespace:
            variables = inputs
        elif type(inputs) == HistoryH5:
            history = inputs
            variables = history.get('MAP')

        # Set up ax
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Set up ground truth
        if groundtruth is not None:
            gt = SimpleNamespace(**{**PARAMETERS, **groundtruth})
        else:
            gt = None

        # Extract variables
        z = variables.z
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        x_data = variables.x_data
        x_indu = variables.x_indu
        x_grid = variables.x_grid
        u_indu = variables.u_indu
        K_grid_indu = variables.K_grid_indu
        K_indu_indu_inv = variables.K_indu_indu_inv
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_traj = variables.num_traj
        traj_mask = variables.traj_mask

        # Calculate values
        times = np.arange(num_data) * dt * 1e-9

        # Plot trajectory
        for t in range(num_traj):
            idt = np.where(traj_mask == t)[0]
            if gt and (gt.x_data is not None):
                ax.plot(times[idt], gt.x_data[idt, :], color='r', label='Ground truth')
            ax.plot(times[idt], x_data[idt, :], color='b', label='Inferred trajectory')
            if num_traj > 1:
                ax.axvline(times[idt[0]], color='k', linewidth=2, label='New trajectory')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        ax.set_ylim([0, 3*R0])

        return ax

    @staticmethod
    def plot_potential(data, inputs, groundtruth=None, ax=None, xzero=None, ulim=None):

        # Check inputs
        if type(inputs) == dict:
            variables = SimpleNamespace(inputs)
        elif type(inputs) == SimpleNamespace:
            variables = inputs
        elif type(inputs) == HistoryH5:
            history = inputs
            variables = history.get('MAP')
            probs = history.get('P')
            last = min([*np.where(probs == 0)[0], probs.shape[0]])
            burn = int(.9 * last)

        # Set up ax
        if ax is None:
            _, ax = plt.subplots(1, 1)
        
        # Set up ground truth
        if groundtruth is not None:
            gt = SimpleNamespace(**{**PARAMETERS, **groundtruth})
        else:
            gt = None

        # Extract variables
        z = variables.z
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        x_data = variables.x_data
        x_grid = variables.x_grid
        u_indu = variables.u_indu
        K_grid_indu = variables.K_grid_indu
        K_indu_indu_inv = variables.K_indu_indu_inv
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_traj = variables.num_traj
        traj_mask = variables.traj_mask

        # Calculate values
        if xzero is None:
            xzero = R0
        i0 = np.argmin(np.abs(x_grid - xzero))
        if type(inputs) == HistoryH5:
            u_hist_indu = history.get('u_indu', burn=burn, last=last)
            u_hist_grid = K_grid_indu @ K_indu_indu_inv @ u_hist_indu[:, :, 0].T
            u_hist_grid -= u_hist_grid[i0, :]
            u_grid_mean = np.mean(u_hist_grid, axis=1)
            u_grid_std = np.std(u_hist_grid, axis=1)
        else:
            u_grid = (K_grid_indu @ K_indu_indu_inv @ u_indu).reshape(-1, order='F')
            u_grid -= u_grid[i0]

        # Plot Potential
        if gt and (gt.force is not None):
            U_gt = np.cumsum(gt.force(x_grid[::-1]))[::-1] * (x_grid[1] - x_grid[0]) / kT
            U_gt -= U_gt[i0]
            ax.plot(U_gt, x_grid, color='r', label='Ground truth')
        if type(inputs) == HistoryH5:
            ax.plot(u_grid_mean, x_grid, color='b', label='FRET-SKIPPER')
            ax.fill_betweenx(
                x_grid[:, 0],
                u_grid_mean - u_grid_std,
                u_grid_mean + u_grid_std,
                color='skyblue',
                alpha=.5,
                label='Uncertainty',
            )
        else:
            ax.plot(u_grid, x_grid, color='b', label='Sampled potential')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        if ulim is not None:
            ax.set_xlim(ulim)
        plt.tight_layout()

        return ax

    @staticmethod
    def plot_variables(data, inputs, title=None, groundtruth=None, fig=None, **kwargs):

        # Set up data
        if data.shape[1] > data.shape[0]:
            data = data.T

        # Set up figure
        if fig is None:
            fig = plt.gcf()
        fig.clf()
        fig.set_size_inches(12, 6)
        ax = np.empty((2, 2), dtype=object)
        gs = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)
        ax[0, 0] = fig.add_subplot(gs[0, :-1])
        ax[1, 0] = fig.add_subplot(gs[1, :-1], sharex=ax[0, 0])
        ax[1, 1] = fig.add_subplot(gs[1, -1], sharey=ax[1, 0])

        # Data
        ax[0, 0].set_title('Data')
        ax[0, 0].set_ylabel('Photons (#)')
        FRETAnalyzer.plot_data(data, inputs, ax=ax[0, 0], **kwargs)

        # Trajectory
        ax[1, 0].set_title(f'Trajectory')
        ax[1, 0].set_ylabel('Position (nm)')
        ax[1, 0].set_xlabel('Time (s)')
        FRETAnalyzer.plot_trajectory(data, inputs, groundtruth=groundtruth, ax=ax[1, 0], **kwargs)

        # Potential
        ax[1, 1].set_title('Potential')
        ax[1, 1].set_xlabel('Potential (kT)')
        FRETAnalyzer.plot_potential(data, inputs, groundtruth=groundtruth, ax=ax[1, 1], **kwargs)

        # Finialize figure
        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        plt.pause(.1)

        return ax
    
    @staticmethod
    def learn_potential(data, parameters=None, num_iter=1000, 
                        saveas=None, plot_status=False, log=False, **kwargs):

        # Print status
        print("Starting inference")
        
        # Set up log
        if not log:
            pass
        elif log is True:
            if saveas is not None:
                log = saveas.split('/')[-1] + '.log'
            else:
                log = f'log{np.random.randint(1e6)}.log'
        elif not log.lower().endswith('.log'):
            log = f'{log}.log'

        # Set up data
        if data.shape[1] > data.shape[0]:
            data = data.T

        # Set up parameters
        if parameters is None:
            parameters = {**PARAMETERS, **kwargs}
        else:
            parameters = {**PARAMETERS, **parameters, **kwargs}

        # Set up variables
        variables = FRETAnalyzer.initialize_variables(data, parameters)
        print('Parameters:')
        for key in sorted(parameters.keys()):
            text = str(getattr(variables, key)).replace('\n', ', ')
            text = '--{} = {}'.format(key, text)
            if len(text) > 80: text = text[:77] + '...'
            print(text)
            if log:
                with open(log, 'a') as handle:
                    handle.write(text + '\n')

        # Set up history
        MAP = copy.deepcopy(variables)
        if saveas is not None:
            history = HistoryH5(
                save_name=saveas,
                variables=variables,
                num_iter=num_iter,
                fields=[
                    'z',
                    'kx',
                    'kd',
                    'ka',
                    # 'x_data',
                    'u_indu',
                    'P',
                ],
            )

        # Gibbs sampler
        for i in range(num_iter):

            # Print status
            print(f'Iteration {i+1} of {num_iter} [', end='')
            t = time.time()

            # Sample variables
            FRETAnalyzer.sample_potential(data, variables)
            print('%', end='')
            FRETAnalyzer.sample_friction(data, variables)
            print('%', end='')
            FRETAnalyzer.sample_rates(data, variables)
            print('%', end='')
            FRETAnalyzer.sample_trajectory(data, variables)
            print('%', end='')
            
            # Save sample
            variables.P = FRETAnalyzer.posterior(data, variables)
            if variables.P >= MAP.P:
                MAP = copy.deepcopy(variables)
            if saveas is not None:
                history.checkpoint(variables, i)
            print('%', end='')

            # Plot
            if plot_status:
                FRETAnalyzer.plot_variables(data, variables)
                print('%', end='')

            # Print status
            print(f'%] ({(time.time()-t):.2f} s) (prob={variables.P:.3e})'.format())
            if log:
                with open(log, 'a') as handle:
                    handle.write(f'Iteration {i+1} of {num_iter} ({round(time.time()-t, 2)}s)\n')

        # Return output
        print('Sampling complete')
        if saveas is not None:
            return MAP, history
        else:
            return MAP



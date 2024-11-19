
# Import libraries
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy import stats
from types import SimpleNamespace
from matplotlib import gridspec
from algorithms import FFBS


# Units are nanoseconds (ns), nanometers (nm), attograms (ag)
PARAMETERS = {
    # Variables
    'P': None,          # (#)     Probability
    's': None,          # (#)     States at each time level
    'r': None,          # (nm)    Position of each state
    'U': None,          # (pN*nm) Potential energy of each state
    'kx': None,         # (1/ns)  Excitation rate
    'ka': None,         # (1/ns)  Background rate for acceptor photons
    'kd': None,         # (1/ns)  Background rate for donor photons
    'pi': None,         # (#)     Transition matrix

    # Experiment
    'dt': 1,              # (ns)           Time step
    'R0': 5,              # (nm)           FRET distance
    'kT': 4.114,          # (ag*nm^2/ns^2) Experiment temperature
    'detectoreff': None,  #                Detection efficiencies
    'crosstalk': None,    #                Cross talk matrix

    # Priors
    'r_shape': 2,      # Hyperparameter
    'r_scale': None,   # Hyperparameter
    'kx_shape': 2,     # Hyperparameter
    'kx_scale': None,  # Hyperparameter
    'kd_shape': 2,     # Hyperparameter
    'kd_scale': None,  # Hyperparameter
    'ka_shape': 2,     # Hyperparameter
    'ka_scale': None,  # Hyperparameter
    'pi_conc': None,   # Hyperparameter

    # Num
    'num_traj': None,    # Number of trajectories
    'num_data': None,    # Number of time levels
    'num_states': 2,     # Number of states

    # Sampler parameters
    'seed': 0,            # Random number generator seed
    'r_prop': 100,        # Proposal distribution shape for state position
    'kx_prop': 100,       # Proposal distribution shape for excitation
    'kd_prop': 100,       # Proposal distribution shape for background
    'ka_prop': 100,       # Proposal distribution shape for background
    'traj_mask': None,    # Mask for trajectory
}


class FRETAnalyzerHMM:

    @staticmethod
    def simulate_data(parameters=None, **kwargs):

        # Set default parameters
        default_parameters = {
            # Variables
            'r': None,             # (nm)           State locations
            'pi': None,            # (#)            Transition matrix
            'kx': 10,              # (1/ns)         Fluorophore excitation rate
            'ka': .5,              # (1/ns)         Acceptor photon background rate
            'kd': 0,               # (1/ns)         Donor photon background rate
            # Constants
            'crosstalk': np.array([[0.75, 0.00], [0.25, 1.00]]),  # Cross talk matrix
            'detectoreff': np.array([1, .34]),                    # Detection efficiency
            'dt': 1,               # (ns)           Chosen time step for Langevin dynamics
            'kT': 4.114,           # (pN*nm)        Experiment temperature
            'R0': 5,               # (nm)           FRET distance
            'num_data': None,      # Number of time levels
            'num_traj': 10,        # Number of trajectories
            'num_states': 2,       # Number of data dimensions
            'num_data_per': 100,   # Number of time levels per trajectory
            'seed': 0,             # RNG seed
        }

        # Set parameters
        if parameters is None:
            parameters = {**default_parameters, **kwargs}
        else:
            parameters = {**default_parameters, **parameters, **kwargs}
        if parameters['num_data'] is None:
            parameters['num_data'] = parameters['num_traj'] * parameters['num_data_per']
        parameters = {**PARAMETERS, **parameters}

        # Set up variables
        variables = FRETAnalyzerHMM.initialize_variables(None, parameters)
        r = variables.r
        pi = variables.pi
        kx = variables.kx
        ka = variables.ka
        kd = variables.kd
        dt = variables.dt
        R0 = variables.R0
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff
        num_data = variables.num_data
        num_traj = variables.num_traj
        num_states = variables.num_states
        num_data_per = variables.num_data_per
        seed = variables.seed

        # Set RNG
        np.random.seed(seed)

        # Sample trajectory
        s = np.zeros(num_data)
        traj_mask = np.zeros(num_data, dtype=int)
        for t in range(num_traj):
            # Find start id
            id = t * num_data_per
            traj_mask[id:id + num_data_per] = t
            # Sample trajectory
            s[id] = np.random.choice(num_states, p=pi[-1, :])
            for n in range(1, num_data_per):
                s[id + n] = np.random.choice(num_states, p=pi[s[id + n - 1], :])

        # Sample data
        data = np.zeros((num_data, 2))
        for n in range(num_data):
            FRET = 1 / (1 + (r[s[n]] / R0) ** 6)
            mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx*(1-FRET) + kd, kx*FRET + ka])
            data[n, 0] = stats.poisson.rvs(mu=mu[0])  # background donor
            data[n, 1] = stats.poisson.rvs(mu=mu[1])  # background acceptor

        # Update variables
        variables.s = s
        variables.traj_mask = None
        variables = variables.__dict__  # Convert to dictionary

        return data, variables


    @staticmethod
    def initialize_variables(data, parameters) -> SimpleNamespace:

        # Set up variables
        variables = SimpleNamespace(**parameters)

        # Extract variables
        r = variables.r
        pi = variables.pi
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        r_shape = variables.r_shape
        r_scale = variables.r_scale
        pi_conc = variables.pi_conc
        kx_shape = variables.kx_shape
        kx_scale = variables.kx_scale
        kd_shape = variables.kd_shape
        kd_scale = variables.kd_scale
        ka_shape = variables.ka_shape
        ka_scale = variables.ka_scale
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff
        num_data = variables.num_data
        num_traj = variables.num_traj
        num_states = variables.num_states
        traj_mask = variables.traj_mask

        # Data required parameters
        if data is not None:

            # Set number of data
            num_data = data.shape[0]
            variables.num_data = num_data

            # Trajectory mask
            if traj_mask is None:
                traj_mask = np.zeros(num_data, dtype=int)
            num_traj = np.max(traj_mask) + 1
            variables.num_traj = num_traj
            variables.traj_mask = traj_mask

        # Detector variables
        if crosstalk is None:
            crosstalk = np.eye(2)
        if detectoreff is None:
            detectoreff = np.ones(2)
        variables.crosstalk = crosstalk
        variables.detectoreff = detectoreff

        # Photon rates
        if kx is None:
            kx = .5 * np.mean(data[:, 0]) / dt / np.mean(detectoreff)
        if kx_scale is None:
            kx_scale = kx / kx_shape
        if kd is None:
            kd = .5 * np.mean(data[:, 1]) / dt / detectoreff[0]
        if kd_scale is None:
            kd_scale = kd / kd_shape
        if ka is None:
            ka = .5 * np.mean(data[:, 1]) / dt / detectoreff[1]
        if ka_scale is None:
            ka_scale = ka / ka_shape
        variables.kx = kx
        variables.kx_scale = kx_scale
        variables.kd = kd
        variables.kd_scale = kd_scale
        variables.ka = ka
        variables.ka_scale = ka_scale

        # States
        s = np.zeros(num_data, dtype=int)
        variables.s = s

        # Positions
        if r is None:
            r = np.ones(num_states) * R0
        if r_scale is None:
            r_scale =  r / r_shape
        variables.r = r
        variables.r_scale = r_scale

        # Transition matrix
        if pi is None:
            pi = np.ones((num_states + 1, num_states))
            pi[:-1, :] += np.eye(num_states) * 100
            for k in range(num_states + 1):
                pi[k, :] /= np.sum(pi[k, :])
        if pi_conc is None:
            pi_conc = np.ones((num_states + 1, num_states))
            pi_conc[:-1, :] += np.eye(num_states) * 10
        variables.pi = pi_conc
        variables.pi_conc = pi_conc

        # Energies
        U = FRETAnalyzerHMM.calculate_energy(variables)
        variables.U = U
        
        # Probability
        P = - np.inf
        variables.P = P

        return variables


    @staticmethod
    def sample_states(data, variables) -> SimpleNamespace:

        # Extract variables
        s = variables.s
        r = variables.r
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        pi = variables.pi
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        kd = variables.kd
        ka = variables.ka
        num_data = variables.num_data
        num_traj = variables.num_traj
        num_states = variables.num_states
        traj_mask = variables.traj_mask
        detectoreff = variables.detectoreff
        crosstalk = variables.crosstalk
        
        # Initialize states
        s = np.zeros(num_data, dtype=int)

        # Loop through trajectories
        for t in range(num_traj):
            ids = np.where(traj_mask == t)[0]

            # Set up log likelihood matrix
            lhood = np.zeros((num_states, len(ids)))
            for k in range(num_states):
                FRET = 1 / (1 + (r[k] / R0) ** 6)
                mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx*(1-FRET) + kd, kx*FRET + ka])
                lhood[k, :] = (
                    stats.poisson.logpmf(data[ids, 0], mu=mu[0])
                    + stats.poisson.logpmf(data[ids, 1], mu=mu[1])
                )

            # Softmax for numerical stability
            lhood = np.exp(lhood - np.max(lhood, axis=0))

            # Sample states using FFBS
            s[ids] = FFBS(lhood, pi)

        # Update variables
        variables.s = s

        pass


    @staticmethod
    def sample_transition_matrix(data, variables) -> SimpleNamespace:

        # Extract variables
        s = variables.s
        pi_conc = variables.pi_conc
        num_traj = variables.num_traj
        num_states = variables.num_states
        traj_mask = variables.traj_mask

        # Count the number of each transition that occurs
        counts = np.zeros((num_states + 1, num_states))
        for t in range(num_traj):
            s_t = s[traj_mask == t]
            counts[-1, s[0]] += 1
            for i in range(num_states):
                for j in range(num_states):
                    counts[i, j] = np.sum((s_t[:-1] == i) * (s_t[1:] == j))

        # Sample
        pi = np.zeros((num_states + 1, num_states))
        for k in range(num_states + 1):
            pi[k, :] = stats.dirichlet.rvs(counts[k, :] + pi_conc[k, :])
        
        # Update variables
        variables.pi = pi
        variables.U = FRETAnalyzerHMM.calculate_energy(variables)

        return


    @staticmethod
    def sample_positions(data, variables) -> SimpleNamespace:

        # Extract variables
        r = variables.r
        s = variables.s
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        pi = variables.pi
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        num_data = variables.num_data
        num_traj = variables.num_traj
        num_states = variables.num_states
        traj_mask = variables.traj_mask
        r_shape = variables.r_shape
        r_scale = variables.r_scale
        r_prop = variables.r_prop
        detectoreff = variables.detectoreff
        crosstalk = variables.crosstalk

        # Loop through the states
        for k in range(num_states):

            # Get ids of data in state k
            ids = s == k

            # Set up probability
            def probability(r_):
                FRET = 1 / (1 + (r_ / R0) ** 6)
                mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx*(1-FRET) + kd, kx*FRET + ka])
                prob = (
                        stats.gamma.logpdf(r_, a=r_shape, scale=r_scale[k])  # prior
                        + np.sum(stats.poisson.logpmf(data[ids, 0], mu=mu[0]))  # likelihood no FRET
                        + np.sum(stats.poisson.logpmf(data[ids, 1], mu=mu[1]))  # likelihood FRET
                )
                return prob
            
            # Sample multiple times
            for _ in range(10):

                # Propose new value
                r_old = r[k].copy()
                r_new = stats.gamma.rvs(a=r_prop, scale=r_old / r_prop)

                # Accept or reject
                P_old = probability(r_old)
                P_new = probability(r_new)
                acc_prob = (
                        P_new - P_old
                        + stats.gamma.logpdf(r_old, a=r_prop, scale=r_new / r_prop)
                        - stats.gamma.logpdf(r_new, a=r_prop, scale=r_old / r_prop)
                )
                if acc_prob > np.log(np.random.rand()):
                    r[k] = r_new

        # Update variables
        variables.r = r

        return


    @staticmethod
    def sample_rates(data, variables):

        # Extract constants
        r = variables.r
        s = variables.s
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        pi = variables.pi
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        num_data = variables.num_data
        num_traj = variables.num_traj
        num_states = variables.num_states
        traj_mask = variables.traj_mask
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff
        kx_shape = variables.kx_shape
        kx_scale = variables.kx_scale
        kx_prop = variables.kx_prop
        kd_shape = variables.kd_shape
        kd_scale = variables.kd_scale
        kd_prop = variables.kd_prop
        ka_shape = variables.ka_shape
        ka_scale = variables.ka_scale
        ka_prop = variables.ka_prop

        # Sample excitation rate
        def probability(kx_):
            prob = stats.gamma.logpdf(kx_, a=kx_shape, scale=kx_scale)  # prior
            for k in range(num_states):
                ids = s == k
                FRET = 1 / (1 + (r[k] / R0) ** 6)
                mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx_*(1-FRET) + kd, kx_*FRET + ka])
                prob += (
                    np.sum(stats.poisson.logpmf(data[ids, 0], mu=mu[0]))  # likelihood no FRET
                    + np.sum(stats.poisson.logpmf(data[ids, 1], mu=mu[1]))  # likelihood FRET
                )
            return prob
        for _ in range(10):
            kx_old = kx
            kx_new = stats.gamma.rvs(a=kx_prop, scale=kx_old/kx_prop)
            P_old = probability(kx_old)
            P_new = probability(kx_new)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(kx_old, a=kx_prop, scale=kx_new/kx_prop)
                - stats.gamma.logpdf(kx_new, a=kx_prop, scale=kx_old/kx_prop)
            )
            if acc_prob > np.log(np.random.rand()):
                kx = kx_new

        # Sample donor background
        if kd_scale > 0:
            def probability(kd_):
                prob = stats.gamma.logpdf(kd_, a=kd_shape, scale=kd_scale)  # prior
                for k in range(num_states):
                    ids = s == k
                    FRET = 1 / (1 + (r[k] / R0) ** 6)
                    mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx*(1-FRET) + kd_, kx*FRET + ka])
                    prob += (
                        np.sum(stats.poisson.logpmf(data[ids, 0], mu=mu[0]))  # likelihood no FRET
                        + np.sum(stats.poisson.logpmf(data[ids, 1], mu=mu[1]))  # likelihood FRET
                    )
                return prob
            for _ in range(10):
                kd_old = kd
                kd_new = stats.gamma.rvs(a=kd_prop, scale=kd_old/kd_prop)
                P_old = probability(kd_old)
                P_new = probability(kd_new)
                acc_prob = (
                    P_new - P_old
                    + stats.gamma.logpdf(kd_old, a=kd_prop, scale=kd_new/kd_prop)
                    - stats.gamma.logpdf(kd_new, a=kd_prop, scale=kd_old/kd_prop)
                )
                if acc_prob > np.log(np.random.rand()):
                    kd = kd_new
            
        # Sample acceptor background
        if ka_scale > 0:
            def probability(ka_):
                prob = stats.gamma.logpdf(ka_, a=ka_shape, scale=ka_scale)  # prior
                for k in range(num_states):
                    ids = s == k
                    FRET = 1 / (1 + (r[k] / R0) ** 6)
                    mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx*(1-FRET) + kd, kx*FRET + ka_])
                    prob += (
                        np.sum(stats.poisson.logpmf(data[ids, 0], mu=mu[0]))  # likelihood no FRET
                        + np.sum(stats.poisson.logpmf(data[ids, 1], mu=mu[1]))  # likelihood FRET
                    )
                return prob
                for _ in range(10):
                    ka_old = ka
                    ka_new = stats.gamma.rvs(a=ka_prop, scale=ka_old/ka_prop)
                    P_old = probability(kd_old)
                    P_new = probability(kd_new)
                    acc_prob = (
                        P_new - P_old
                        + stats.gamma.logpdf(ka_old, a=ka_prop, scale=ka_new/ka_prop)
                        - stats.gamma.logpdf(ka_new, a=ka_prop, scale=ka_old/ka_prop)
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

        # Set kwarg args into variables
        variables = copy.copy(variables)
        for key, val in kwargs.items():
            setattr(variables, key, val)

        # Get variables
        s = variables.s
        r = variables.r
        kx = variables.kx
        kd = variables.kd
        ka = variables.ka
        pi = variables.pi
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        r_shape = variables.r_shape
        r_scale = variables.r_scale
        pi_conc = variables.pi_conc
        kx_shape = variables.kx_shape
        kx_scale = variables.kx_scale
        kd_shape = variables.kd_shape
        kd_scale = variables.kd_scale
        ka_shape = variables.ka_shape
        ka_scale = variables.ka_scale
        num_data = variables.num_data
        num_traj = variables.num_traj
        num_states = variables.num_states
        traj_mask = variables.traj_mask
        crosstalk = variables.crosstalk
        detectoreff = variables.detectoreff

        # Prior
        prob = (
            np.sum(stats.gamma.logpdf(r, a=r_shape, scale=r_scale))   # Prior on r
            + np.sum(stats.gamma.logpdf(kx, a=kx_shape, scale=kx_scale))  # Prior on kx
        )
        if kd_scale > 0:
            prob += np.sum(stats.gamma.logpdf(kd, a=kd_shape, scale=kd_scale))  # Prior on kd
        if ka_scale > 0:
            prob += np.sum(stats.gamma.logpdf(ka, a=ka_shape, scale=ka_scale))  # Prior on ka
        for k in range(num_states + 1):
            prob += stats.dirichlet.logpdf(pi[k, :], pi_conc[k, :])  # Prior on pi

        # Dyanmics
        counts = np.zeros((num_states + 1, num_states))
        for t in range(num_traj):
            s_t = s[traj_mask == t]
            counts[-1, s[0]] += 1
            for i in range(num_states):
                for j in range(num_states):
                    counts[i, j] = np.sum((s_t[:-1] == i) * (s_t[1:] == j))
        prob += np.sum(counts * np.log(pi))

        # Likelihood
        for k in range(num_states):
            ids = s == k
            FRET = 1 / (1 + (r[k] / R0) ** 6)
            mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx*(1-FRET) + kd, kx*FRET + ka])
            prob += (
                np.sum(stats.poisson.logpmf(data[ids, 0], mu=mu[0]))  # likelihood no FRET
                + np.sum(stats.poisson.logpmf(data[ids, 1], mu=mu[1]))  # likelihood FRET
            )

        return prob


    @staticmethod
    def calculate_energy(variables):

        # get variables
        dt = variables.dt
        kT = variables.kT
        pi = variables.pi

        # calculate eigenvector
        q = pi[:-1, :].T
        eigval, eigvec = scipy.linalg.eig(q)

        # get eigenvector with eigenvalue 1
        id = np.argmin(np.abs(eigval - 1))
        P = eigvec[:, id]
        P = P / np.sum(P)

        # calculate energies in units of kT
        energies = - np.log(P)
        energies -= np.min(energies)

        return energies


    @staticmethod
    def plot_variables(data, variables, ground_truth=None):

        # Set up data
        if data.shape[1] > data.shape[0]:
            data = data.T

        # Extract variables
        s = variables.s
        r = variables.r
        kx = variables.kx
        pi = variables.pi
        dt = variables.dt
        kT = variables.kT
        R0 = variables.R0
        kd = variables.kd
        ka = variables.ka
        num_data = variables.num_data
        num_traj = variables.num_traj
        num_states = variables.num_states
        traj_mask = variables.traj_mask
        detectoreff = variables.detectoreff
        crosstalk = variables.crosstalk

        # Calculate trajectory
        times = np.arange(num_data) * dt
        trace = np.zeros((num_data, 2))
        for k in range(num_states):
            ids = s == k
            FRET = 1 / (1 + (r[k] / R0) ** 6)
            mu = dt * np.diag(detectoreff) @ crosstalk @ np.array([kx*(1-FRET) + kd, kx*FRET + ka])
            trace[ids, 0] = mu[0]  # no FRET
            trace[ids, 1] = mu[1]  # FRET

        # set up figure
        fig = plt.gcf()
        fig.clf()
        fig.set_size_inches(12, 6)
        ax = np.empty((2, 1), dtype=object)
        gs = gridspec.GridSpec(nrows=2, ncols=1, figure=fig)
        ax[0, 0] = fig.add_subplot(gs[0, :])
        ax[1, 0] = fig.add_subplot(gs[1, :], sharex=ax[0, 0])

        # plot data and trace for donor
        ax[0, 0].set_title('Donor photons')
        ax[0, 0].set_title('Trajectory')
        ax[0, 0].set_ylabel('Photons (#)')
        for t in range(num_traj):
            ids = traj_mask == t
            if num_traj > 1:
                ax[0, 0].axvline(times[ids][0], color='k', linewidth=2, label='New dataset')
            ax[0, 0].step(times[ids], data[ids, 0], color='g', where='pre', label='Data (green)')
            ax[0, 0].plot(times[ids], trace[ids, 0], color='b', label='Sampled trajectory')

        # plot data and trace for acceptor
        ax[1, 0].set_title('Acceptor photons')
        ax[1, 0].set_title('Trajectory')
        ax[1, 0].set_xlabel('Time (ns)')
        ax[1, 0].set_ylabel('Photons (#)')
        for t in range(num_traj):
            ids = traj_mask == t
            if num_traj > 1:
                ax[1, 0].axvline(times[ids][0], color='k', linewidth=2, label='New dataset')
            ax[1, 0].step(times[ids], data[ids, 1], color='r', where='pre', label='Data (red)')
            ax[1, 0].plot(times[ids], trace[ids, 1], color='b', label='Sampled trajectory')

        # set up legend
        handles = [handle for axes in fig.axes for handle in axes.get_legend_handles_labels()[0]]
        labels = [label for axes in fig.axes for label in axes.get_legend_handles_labels()[1]]
        by_label = dict(zip(labels, handles))
        ax[0, -1].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1),)

        # tighten figure
        plt.tight_layout()
        plt.pause(.1)

        return


    @staticmethod
    def plot_results(data, history, ground_truth=None, x_zero=None):
        pass


    @staticmethod
    def learn_potential(
        data, parameters=None, num_iter=1000, 
        plot_status=False, log=False, **kwargs
    ):
        
        # Set up log
        if not log:
            pass
        elif log is True:
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
        variables = FRETAnalyzerHMM.initialize_variables(data, parameters)
        print('parameters:')
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

        # Gibbs sampler
        for i in range(num_iter):

            # Print status
            print(f'Iteration {i+1} of {num_iter} [', end='')
            t = time.time()

            # Sample variables
            FRETAnalyzerHMM.sample_states(data, variables)
            print('%', end='')
            FRETAnalyzerHMM.sample_positions(data, variables)
            print('%', end='')
            FRETAnalyzerHMM.sample_rates(data, variables)
            print('%', end='')
            FRETAnalyzerHMM.sample_transition_matrix(data, variables)
            print('%', end='')
            
            # Save sample
            variables.P = FRETAnalyzerHMM.posterior(data, variables)
            if variables.P >= MAP.P:
                MAP = copy.deepcopy(variables)
            print('%', end='')

            # Plot
            if plot_status:
                FRETAnalyzerHMM.plot_variables(data, variables)
                print('%', end='')

            # Print status
            print('%] ({} s)'.format(round(time.time()-t, 2)))
            if log:
                with open(log, 'a') as handle:
                    handle.write(f'Iteration {i+1} of {num_iter} ({round(time.time()-t, 2)}s)\n')

        # Return output
        print('Sampling complete')
        return MAP

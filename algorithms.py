
# Import libraries
import h5py
import copy
import numba as nb
import numpy as np
import dill as pickle


# Forward Filter algorithm
@nb.njit(cache=True)
def forward_filter_nb(likelihood_matrix, transition_matrix):
    """
    This function calculates the forward filter used for sampling the
    states.

    :param likelihood_matrix: A matrix containing the likelihood value
    at each timepoint for each state
    :param transition_matrix: The transition matrix. The final row is
    the initial starting probability array.
    :return: The forward filter.
    """

    # extract values
    lhood = likelihood_matrix
    pi0 = transition_matrix[-1, :]
    pis = transition_matrix[:-1, :]
    num_states, num_data = lhood.shape

    # find initial probability
    forward = np.zeros((num_states, num_data))
    forward[:, 0] = lhood[:, 0] * pi0
    forward[:, 0] /= np.sum(forward[:, 0])
    for n in range(1, num_data):
        """
        The probability of each state at each time level is the 
        likelihood times the probability of the state marginalized over 
        the state of the previous time step
        """
        forward[:, n] = lhood[:, n] * (pis.T @ forward[:, n - 1])
        forward[:, n] /= np.sum(forward[:, n])

    return forward


# Backward Sample algorithm
@nb.njit(cache=True)
def backward_sample_nb(forward_filter, transition_matrix):
    """
    This funciton takes a forward filter and a transition matrix and
    outputs a sampled state trajectory.

    :param forward_filter: The forward filter.
    :param transition_matrix: The transition matrix where the final
    row is the initial state probability.
    :return: A sampled state trajectory.
    """

    # get values
    forward = forward_filter
    pis = transition_matrix[:-1, :]
    num_states, num_data = forward.shape

    # start by sampling the final state from the forward filter
    states = np.zeros(num_data, dtype=np.int32)
    s = np.searchsorted(np.cumsum(forward[:, -1]), np.random.rand())
    states[-1] = s
    for m in range(1, num_data):
        """
        Every preceeding state is sampled from the forward filter
        times the probability of the preceeding state given
        the following step.
        """
        n = num_data - m - 1
        backward = forward[:, n] * pis[:, s]
        backward /= np.sum(backward)
        s = np.searchsorted(np.cumsum(backward), np.random.rand())
        states[n] = s

    return states


# Forward Filter Backwards Sample algorithm
def FFBS(likelihood_matrix, transition_matrix):
    """
    "Forward Filter Backwards Sample"
    This algorithm calculates the forward filter then
    draws a sample from it using the backward sample
    algorithm.

    :param likelihood_matrix: The likelihood of each state
    at each time level.
    :param transition_matrix: The transition matrix where
    the final row is the initial state probability.
    :return: A sampled state trajectory.
    """
    return backward_sample_nb(
        forward_filter_nb(
            likelihood_matrix, transition_matrix
        ), 
        transition_matrix
    )

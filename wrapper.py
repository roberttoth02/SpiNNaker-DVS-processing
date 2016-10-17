from ShonTopology3 import ShonTopology3
from spynnaker.pyNN import *
import numpy as np


# TODO
# Construct training sequence


# Simulation parameters
simulation_params = {'timestep':   1, # (ms) Target timestep
                     'min_delay':  1, # (ms) Minimum timestep
                     'max_delay': 10, # (ms) Maximum timestep
                     'length':   300} # (ms) Simulation length

# STDP Learning rule parameters
stdp_params = {'tau_plus': 25.0,
               'tau_min':  30.0,
               'w_min':     0.0,
               'w_max':     1.0,
               'A_plus':   0.10,
               'A_min':    0.125}

# Conductance based leaky integrate and fire model parameters
cell_params = {'cm':         0.5, # (nF) Membrane capacitance
               'tau_m':       20, # (ms) Membrane time constant = Rm*Cm,
               'tau_refrac':   5, # (ms) Refractory period
               'tau_syn_E': 10.0, # (ms) Decay time of exhib. synaptic conductance
               'tau_syn_I': 40.0, # (ms) Decay time of inhib. synaptic conductance
               'e_rev_E':      0, # (mV) Reversal potential for excitatory input
               'e_rev_I':    -80, # (mV) Reversal potential for inhibitory input
               'v_reset':  -50.0, # (mV) Reset potential after spike
               'v_rest':   -60.0, # (mV) Resting potential
               'v_thresh': -40.0, # (mV) Spike threshold
               'i_offset':  0.35} # (nA) Offset current


# Population sizes
pop_size = {'input': 256, # Number of neurons in input layers
            'output': 40} # Number of neurons in output layer

# Synaptic delays
delays = {'on_static':    1, # (ms)
          'off_static':   1, # (ms)
          'on_stdp':      1, # (ms)
          'off_stdp':     1, # (ms)
          'rec_i_static': 1, # (ms)
          'rec_e_static': 1} # (ms)

# Weights
rng = NumpyRNG(seed=4242)
weights = {'on_static': RandomDistribution('uniform', [10, 0.004], rng=rng),    # (uS)
           'off_static': RandomDistribution('uniform', [10, 0.004], rng=rng),   # (uS)
           'rec_i_static': 0.0055,                                              # (uS)
           'rec_e_static': RandomDistribution('uniform', [0, 0.005], rng=rng),  # (uS)
           'stdp_on_init': RandomDistribution('uniform', [0, 0.004], rng=rng),  # (uS)
           'stdp_off_init': RandomDistribution('uniform', [0, 0.004], rng=rng)} # (uS)


# Runs a simulation
simulation = ShonTopology3(simulation_params=simulation_params, mode='training',
                           cell_params=cell_params, stdp_params=stdp_params,
                           input_pop_size=100, output_pop_size=20,
                           weights=weights, delays=delays,
                           input_sequence=[])

simulation.run_simulation()

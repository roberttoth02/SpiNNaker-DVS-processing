__author__ = "roberttoth"

from spynnaker.pyNN import *
import numpy as np
from itertools import permutations as perm
import scipy.io
import pylab

# The class implements the network topology 3 from Shon, A.P. et al (2004)
# Motion detection and prediction through spike-timing dependent plasticity.
# Network: Computation in Neural Systems, 15(3), pp. 179-198.
# intended for deployment on the SpiNNaker platform

# The file includes a class-independent helper method for creating recurrent projections
# twoWayRecurrentProjection()


# As of October 2016, conductance based neurons with multiplicative stdp learning rule
# are not available by default on SpiNNaker
# https://groups.google.com/forum/#!msg/spinnakerusers/tPWiPqTkQq4/BnOQYq8NAwAJ

class ShonTopology3:
    def __init__(self, simulation_params, mode, cell_params, stdp_params,
                 input_pop_size, output_pop_size, weights, delays, input_sequence):

        self.simulation_params = simulation_params
        self.mode = mode
        self.cell_params = cell_params
        self.stdp_param = stdp_params
        self.input_pop_size = input_pop_size
        self.output_pop_size = output_pop_size
        self.weights = weights
        self.delays = delays
        self.input_sequence = input_sequence

    # Defines network topology
    def create_network(self):
        #
        # Populations
        #
        #input_pop_on = Population(self.input_pop_size, SpikeSourceArray,
        #{'spike_times': self.input_sequence})
        #input_pop_off = Population(self.input_pop_size, SpikeSourceArray,
        #{'spike_times': self.input_sequence})
        input_pop_off = Population(self.input_pop_size, IF_cond_exp, self.cell_params)
        input_pop_on = Population(self.input_pop_size, IF_cond_exp, self.cell_params)
        output_pop = Population(self.output_pop_size, IF_cond_exp, self.cell_params)

        #
        # Synapses
        #
        
        # Static inhibitory recurrent connections between output neurons
        rec_e_static = Projection(output_pop, output_pop,
                                  AllToAllConnector(weights=self.weights['rec_i_static'],
                                                    delays=self.delays['rec_i_static'],
                                                    allow_self_connections=False),
                                  target='inhibitory')
                                  
        # Static excitatory recurrent connections between output neurons
        rec_e_static = Projection(output_pop, output_pop,
                                  AllToAllConnector(weights=self.weights['rec_e_static'],
                                                    delays=self.delays['rec_e_static'],
                                                    allow_self_connections=False),
                                  target='excitatory')

        # Static inhibitory synapses from source to output neurons
        on_static = Projection(input_pop_on, output_pop,
                               AllToAllConnector(weights=self.weights['on_static'],
                                                 delays=self.delays['on_static'],
                                                 allow_self_connections=False),
                               target='inhibitory')

        off_static = Projection(input_pop_off, output_pop,
                                AllToAllConnector(weights=self.weights['off_static'],
                                                  delays=self.delays['off_static'],
                                                  allow_self_connections=False),
                                target='inhibitory')

        # STDP Synapses
        if(self.mode == "training"):
            t_dep = SpikePairRule(tau_plus=self.stdp_param["tau_plus"],
                                  tau_minus=self.stdp_param["tau_min"],
                                  nearest=True)

            w_dep = MultiplicativeWeightDependence(w_min=self.stdp_param["w_min"],
                                                   w_max=self.stdp_param["w_max"],
                                                   A_plus=self.stdp_param["A_plus"],
                                                   A_minus=self.stdp_param["A_min"])

            stdp_model = STDPMechanism(timing_dependence=t_dep, weight_dependence=w_dep)

            # Dynamic excitatory synapses from source to output neurons
            on_stdp = Projection(input_pop_on, output_pop,
                                 AllToAllConnector(weights=self.weights['stdp_on_init'],
                                                   delays=self.delays['on_stdp'],
                                                   allow_self_connections=False),
                                 synapse_dynamics=SynapseDynamics(slow=stdp_model),
                                 target='excitatory')

            off_stdp = Projection(input_pop_off, output_pop,
                                  AllToAllConnector(weights=self.weights['stdp_off_init'],
                                                    delays=self.delays['off_stdp'],
                                                    allow_self_connections=False),
                                  synapse_dynamics=SynapseDynamics(slow=stdp_model),
                                  target='excitatory')

        elif(self.mode == "testing"):
            # TODO
            # use FromListConnector to recreate a network from already trained and saved weights
            # using static synapses
            sys.exit("Simulation mode not implemented")

        else:
            sys.exit("Invalid simulation mode")

        return (input_pop_on, input_pop_off, output_pop), (on_stdp, off_stdp)

    # Controls simulation and data recording
    def run_simulation(self):
        # Initialise spiNNaker settings
        setup(timestep=1, min_delay=1, max_delay=10.0)

        # Set up the network
        populations, stdp_projections = self.create_network()

        # Initialize recording of spikes on each population
        for i in populations:
            i.record()

        # Run the simulation for the length of training (given in milliseconds)
        run(self.simulation_params['length'])

        # Extract and save recorded data
        self.save_Weights(stdp_projections)
        self.display_spikes(populations)     # TODO

        self.plot_spikes(populations)

        # End of spiNNaker session
        end()

    def save_weights(self, stdp_projections):
        trained_weights_on = stdp_projections[0].getWeights(format='array')
        trained_weights_off = stdp_projections[1].getWeights(format='array')
        scipy.io.savemat('trained_weights.mat',
                         {'trained_weights_on': trained_weights_on,
                         'trained_weights_off': trained_weights_off})

    def display_spikes(self, populations):
        for i in populations:
            i.getSpikes(compatible_output=True)

    def plot_spikes(self, populations, title='Some plot'):
        spikes = populations[2].getSpikes(compatible_output=True)
        size = populations[2].size
        
        if spikes is None:
            print "No spikes received"
            return None
        
        pylab.figure()
        
        ax = pylab.plt.subplot(111, xlabel='Time/ms',
                               ylabel='Neurons #', title=title)
        
        pylab.xlim((0, self.simulation_params['length']))
        pylab.ylim((-0.5, size - 0.5))
        
        lines = pylab.plot([i[1] for i in spikes],
                           [i[0] for i in spikes], ".")
        
        pylab.axvline(32500, 0, 1, linewidth=4, color='c',
                      alpha=0.75, linestyle='dashed')
        pylab.axvline(64500, 0, 1, linewidth=4, color='c',
                      alpha=0.75, linestyle='dashed')
        pylab.axvline(96500, 0, 1, linewidth=4, color='c',
                      alpha=0.75, linestyle='dashed')
        pylab.setp(lines, markersize=10, color='r')
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
            
        pylab.show()


# Returns a projection object, that defines two-way all-to-all connectivity within a layer
# without self-connections
# Unused, as it turned out AllToAllConnector is able to handle having the same
# population as both input and output
def twoWayRecurrentProjection(population, syn_type, weight, delays):
    conn = np.array(list(perm(np.arange(population.size), 2)))
    print(conn)
    conn = np.concatenate((conn, weight * np.ones((population.size, 1), dtype='int')))
    print(conn)
    conn = np.concatenate((conn, delays * np.ones((population.size, 1), dtype='int')))
    projection = Projection(population, population, FromListConnector(conn_list=conn),
                            target=syn_type)
    return projection

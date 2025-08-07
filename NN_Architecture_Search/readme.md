# NN Architecture Search

This is the NN architecture search segment of the CrossNAS framework. The CrossNAS framework adopts a PIM simulation framework MNSIM-2.0 for the simulation of latency and energy. You need to copy the MNSIM-2.0 directory to run the search.

1. _train_supernet.py_ => Trains the supernet architecture comprising all possible neural architectures in the search space
2. _evol_search.py_ => Performs an evolutionary subnet search based on the trained supernet
3. _retrain_best_choice.py_ => Retrains the best architecture choice from the subnet search stage

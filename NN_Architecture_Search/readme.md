# NN Architecture Search

This is the NN architecture search segment of the CrossNAS framework.

1. _train_supernet.py_ => Trains the supernet architecture comprising all possible neural architectures in the search space
2. _evol_search.py_ => Performs an evolutionary subnet search based on the trained supernet
3. _retrain_best_choice.py_ => Retrains the best architecture choice from the subnet search stage

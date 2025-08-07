# CrossNAS

Welcome to the CrossNAS framework!

This is the GitHUB directory for the paper **CrossNAS: A Cross-Layer Neural Architecture Search Framework for PIM Systems**. The paper is available in the Proceedings of the Great Lakes Symposium on VLSI 2025 and can be accessed using this link https://dl.acm.org/doi/10.1145/3716368.3735178

The **CrossNAS** framework has two segments.
1. _NN_Architecture_Search_: Performs weight-sharing based supernet training and subnet search for the neural network architecture
2. _Mixed_Precision_Quantization_and_PIM_Search_: Finds the efficient mixed-precision quantization map and PIM configuration parameters for the searched NN architecture

You need to pull the [MNSIM-2.0](https://github.com/thu-nics/MNSIM-2.0) PIM simulation framework in both of the directories before initiating simulations. Please cite the following paper if you use the framework.
_Md Hasibul Amin, Mohammadreza Mohammadi, Jason D. Bakos, and Ramtin Zand. 2025. CrossNAS: A Cross-Layer Neural Architecture Search Framework for PIM Systems. In Proceedings of the Great Lakes Symposium on VLSI 2025 (GLSVLSI '25). Association for Computing Machinery, New York, NY, USA, 334–340. https://doi.org/10.1145/3716368.3735178_

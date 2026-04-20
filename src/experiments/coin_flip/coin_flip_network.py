import torch
import torch.nn as nn
import numpy as np

# implementation following https://github.com/samlobel/CFN
# @inproceedings{Lobel2023Flipping,
# title={Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning},
# author={Sam Lobel and Akhil Bagaria and George Konidaris},
# booktitle={International Conference on Machine Learning},
# year={2023},
# }
# d_dim comes from the paper, it is the dimension of the coin_flip_vector, that is to say,
# the sample size from the Rademacher dist

class CoinFlipNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 d_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.d_dim = d_dim
        self.coin_flip_predictor = self._init_flip_network(state_dim, hidden_dim, d_dim)
        self.frozen_prior_network = self._init_frozen_prior_network(state_dim, hidden_dim, d_dim)
    
    def _init_flip_network(self, state_dim: int, hidden_dim: int, d_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, d_dim)
        )
    
    # we just need some noise thus keeping a random initalised network frozen
    # therefore we can simply create a secound coin flip network and freez it
    def _init_frozen_prior_network(self, state_dim: int, hidden_dim: int, d_dim: int):
        prior_net = self._init_flip_network(state_dim, hidden_dim, d_dim)
        for param in prior_net.parameters():
            nn.init.normal_(param, mean=0, std=0.01)
            param.requires_grad = False
        
        return prior_net


    def forward(self, state: torch.Tensor):
        coin_flip_preds = self.coin_flip_predictor(state)

        with torch.no_grad():
            coin_flip_priors = self.frozen_prior_network(state)

        final_coin_flip_preds = coin_flip_preds #+ coin_flip_priors

        pseudo_count = (1/self.d_dim) * (abs(final_coin_flip_preds).sum() ** 2)

        one_over_counts = pseudo_count ** 0.5 # inverse of pseudo-count

        return coin_flip_preds, coin_flip_priors, final_coin_flip_preds, one_over_counts

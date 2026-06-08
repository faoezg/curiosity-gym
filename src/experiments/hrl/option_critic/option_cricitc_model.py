from .option_critic import OptionCriticNetwork
from .buffer import OptionReplayBuffer, OptionTransition


from torch.distributions import Bernoulli
import torch

# adopted from https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py
class OptionCriticModel():
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 option_dim,
                 hidden_dim,
                 buffer_size = 1000,
                 gamma = 0.1,
                 entropy_scaler = 0.1,
                 termination_scaler = 0.1,
                 ) -> None:
        self.device = device
        self.gamma = gamma
        self.entropy_scaler = entropy_scaler
        self.termination_scaler = termination_scaler

        self.option_critic_network = OptionCriticNetwork(state_dim, action_dim, option_dim, hidden_dim).to(device)
        self.target_option_critic_network = OptionCriticNetwork(state_dim, action_dim, option_dim, hidden_dim).to(device)
        self.target_option_critic_network.load_state_dict(self.option_critic_network.state_dict())

        self.buffer = OptionReplayBuffer(buffer_size)

    def predict_option_termination(self, state):
        hidden = self.option_critic_network.get_hidden(state)
        termination = self.option_critic_network.get_terminations(hidden)
        option_termination = Bernoulli(termination).sample()

        policy = self.option_critic_network.get_policy_over_options(hidden)
        new_option = policy.argmax(dim=-1)

        return bool(option_termination.item()), new_option.item()

    def calc_critic_loss(self, trajecotrie_batches: list[list[OptionTransition]]):

        for batch_idx, batch in enumerate(trajecotrie_batches):
            states = []
            rewards = []
            next_states = []
            dones = []
            option_idxs = []

            for transition in batch:
                states.append(transition.state)
                rewards.append(transition.reward)
                next_states.append(transition.next_state)
                dones.append(transition.done)
                option_idxs.append(transition.option_idx)
            
            states_tensor = torch.tensor(states).to(self.device)
            next_states_tensor = torch.tensor(next_states).to(self.device)
            masks = 1 - torch.FloatTensor(dones).to()

            hidden = self.option_critic_network.get_hidden(states_tensor)
            policies = self.option_critic_network.get_policy_over_options(hidden)

            target_next_hidden = self.target_option_critic_network.get_hidden(next_states_tensor)
            target_next_policies = self.target_option_critic_network.get_policy_over_options(target_next_hidden)

            next_hidden = self.option_critic_network.get_hidden(next_states_tensor)
            next_termination_probs = self.option_critic_network.get_terminations(next_hidden).detach()
            next_options_term_prob = next_termination_probs[batch_idx, option_idxs]
            
            termination_weighted_next_policy_error = (1 - next_options_term_prob) * target_next_policies[batch_idx, option_idxs] \
                                                    + next_options_term_prob * target_next_policies.max(dim=-1)[0]
            g_t = rewards + masks * self.gamma * termination_weighted_next_policy_error

            td_error = (1/2 * ((policies[batch_idx, option_idxs] - g_t.detach()) ** 2)).mean()
            return td_error
    
    def calc_actor_loss(self, transition: OptionTransition):
        state_tensor = torch.tensor(transition.state).to(self.device)
        next_state_tensor = torch.tensor(transition.next_state).to(self.device)
        hidden = self.option_critic_network.get_hidden(state_tensor)
        next_hidden = self.option_critic_network.get_hidden(next_state_tensor)
        target_next_hidden = self.target_option_critic_network.get_hidden(state_tensor)

        option_term_prob = self.option_critic_network(hidden)[:, transition.option_idx]
        next_option_term_prob = self.option_critic_network(next_hidden)[:, transition.option_idx].detach()

        policy = self.option_critic_network.get_policy_over_options(hidden).detach().squeeze()
        target_next_policy = self.target_option_critic_network.get_policy_over_options(next_hidden).detach().squeeze()

        termination_weighted_next_policy_error = (1 - next_option_term_prob) * target_next_policy[transition.option_idx] \
                                                    + next_option_term_prob * target_next_policy.max(dim=-1)[0]
        g_t = transition.reward + (1 - transition.done) * self.gamma * termination_weighted_next_policy_error

        termination_loss = option_term_prob * (policy[transition.option_idx].detach() - policy.max(dim=-1)[0].detach() + self.termination_scaler) \
                                            * (1 - transition.done)

        policy_loss = -logp * (g_t.detach() - policy[transition.option_idx]) - self.entropy_scaler * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss

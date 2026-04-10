import torch
from torch import device, optim
import numpy as np
import random

from curiosity_gym.utils.enums import Action
from .buffer import SequenceReplayBuffer, R2D2Transition
from .r2d2_network import R2D2Network

class R2D2Model():
    def __init__(self,
                 state_dim: int,
                 encoder_hidden_dim: int = 12,
                 action_dim: int = 4,
                 lstm_hidden_dim: int = 512,
                 dueling_hidden_dim: int = 128,
                 device: device | str = "cpu",
                 sequence_length: int = 10,
                 burn_in_transition_count: int = 5,
                 n_step: int = 5,
                 reward_discount: float = 0.997, # from the R2D2 paper
                 eps: float = 0.1,
                 target_update_step_count: int = 2500,
                 buffer_size: int = 64,
                 buffer_alpha: float = 0.9,
                 buffer_priority_eta: float = 0.9,
                 ) -> None:
        self.device = device
        self.r2d2_network = R2D2Network(state_dim, encoder_hidden_dim, action_dim, lstm_hidden_dim, dueling_hidden_dim).to(device)
        self._init_r2d2_target_network(state_dim, encoder_hidden_dim, action_dim, lstm_hidden_dim, dueling_hidden_dim)
        self.buffer = SequenceReplayBuffer(buffer_size, buffer_alpha, sequence_length, sequence_length // 2)
        self.optimizer = optim.Adam(self.r2d2_network.parameters(), lr=0.001)

        self.state_dim = state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.sequence_length = sequence_length
        self.burn_in_transition_count = burn_in_transition_count
        self.n_step = n_step
        self.reward_discount = reward_discount
        self.eps = eps
        self.target_update_step_count = target_update_step_count
        self.buffer_priority_eta = buffer_priority_eta

        self._prev_action = 0
        self._prev_reward = 0.0
        self._step_count = 0
        self._gru_hidden = (
            torch.zeros(1, 1, lstm_hidden_dim, device=device),
            torch.zeros(1, 1, lstm_hidden_dim, device=device)
        )

    def _init_r2d2_target_network(self, state_dim: int, encoder_hidden_dim: int, action_dim:int, lstm_hidden_dim: int, dueling_hidden_dim: int):
        self.r2d2_target_network = R2D2Network(state_dim, encoder_hidden_dim, action_dim, lstm_hidden_dim, dueling_hidden_dim).to(self.device)
        self.r2d2_target_network.load_state_dict(self.r2d2_network.state_dict())
        self.r2d2_target_network.eval()

    def calc_q_value(self, state: np.ndarray, goal_state: np.ndarray, action: Action | int):
        self.r2d2_network.eval()

        self._reset_for_next_episode()
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            goal_state_tensor = self._to_tensor(goal_state).unsqueeze(0)
            action_tensor = self._to_tensor(action, torch.long).unsqueeze(0)
            reward_tensor = self._to_tensor(self._prev_reward).unsqueeze(0)
            zero_gru = self._gru_hidden = (
            torch.zeros(1, 1, self.lstm_hidden_dim, device=self.device),
            torch.zeros(1, 1, self.lstm_hidden_dim, device=self.device)
            )
            value, new_gru = self.r2d2_network.forward(state_tensor, goal_state_tensor, action_tensor, reward_tensor, zero_gru)
            self._gru_hidden = new_gru
            value = value.squeeze(0)

        self.r2d2_network.train()
        return value.max().detach()

    def get_action_from_greedy_policy(self, state: np.ndarray, goal_state: np.ndarray) -> Action | int:
        action_list = [element.value for element in Action] # Why is there no method for this?
        if (random.random() < self.eps):
            action = random.choice(action_list)
            self._prev_action = action
            return action
        
        q_vals, new_gru_hidden = self._pass_state_through_r2d2_network(state, goal_state)
        q_vals = q_vals.squeeze(0)
        action = int(q_vals.argmax(dim=1).detach().item())
        self._gru_hidden = new_gru_hidden

        self._prev_action = action # will become the prev_action once env has stepped
        self._prev_reward = 0.0 # reset reward, as it will be received after env step following the action

        return action
    
    def _pass_state_through_r2d2_network(self, state: np.ndarray, goal_state: np.ndarray):
        state_tensor = self._to_tensor(state).unsqueeze(0) # ensure traj dim exists
        goal_state_tensor = self._to_tensor(goal_state).unsqueeze(0) # ensure traj dim exists
        prev_action_tensor = self._to_tensor(self._prev_action, dtype=torch.long).unsqueeze(0)
        prev_reward_tensor = self._to_tensor(self._prev_reward).unsqueeze(0)

        with torch.no_grad():
            q_vals, new_gru_hidden = self.r2d2_network.forward(
                state_tensor,
                goal_state_tensor,
                prev_action_tensor,
                prev_reward_tensor,
                self._gru_hidden
            )

        return q_vals, new_gru_hidden
 
    def handle_transition(self,
                        state: np.ndarray,
                        action: Action | int,
                        reward: float,
                        next_state: np.ndarray,
                        goal_state: np.ndarray,
                        done: bool,
                        priority: float = 1.0):

        self._store_transition(state,
                        action,
                        reward,
                        next_state,
                        goal_state,
                        done,
                        priority)
        self._prev_reward = reward

        if (not done):
            #_, new_gru_hidden = self._pass_state_through_r2d2_network(next_state, goal_state) 
            #self._gru_hidden = new_gru_hidden # this is different from the prev new_gru_hidden, as this stems from the next_state
            self._prev_reward = reward
            self._prev_action = action
        #else:
            #self._reset_for_next_episode()


    def _store_transition(self,
                          state: np.ndarray,
                          action: Action | int,
                          reward: float,
                          next_state: np.ndarray,
                          goal_state: np.ndarray,
                          done: bool,
                          priority: float = 1.0):
        gru_hidden_copy = (
            self._gru_hidden[0].detach().clone(),
            self._gru_hidden[1].detach().clone()
        )

        transition = R2D2Transition(
            state,
            action,
            reward,
            next_state,
            priority,
            goal_state,
            done,
            gru_hidden_copy
        )
        self.buffer.add(transition)
    
    def _reset_for_next_episode(self):
       # self._gru_hidden = (
       #     torch.zeros_like(self._gru_hidden[0]),
       #     torch.zeros_like(self._gru_hidden[1]),
       # )
       # self._prev_action = 0
       # self._prev_reward = 0.0
       pass
    
    def train_network(self, batch_size: int) -> float:
        if (len(self.buffer) // self.sequence_length < batch_size):
            print("Skipp Training", len(self.buffer) // self.sequence_length)
            return 0.0

        weights_tensor, states_tensor, goal_states_tensor, actions_tensor, rewards_tensor, dones_tensor, prev_actions_tensor, prev_rewards_tensor, inital_hidden, inital_c, batch_starter_indicies = self._get_training_tensors(batch_size)

        inital_hidden, inital_c = self._burn_in_lstm(states_tensor,
                                                     goal_states_tensor,
                                                     prev_actions_tensor,
                                                     prev_rewards_tensor,
                                                     inital_hidden,
                                                     inital_c)

        training_starting_sequence_idx = self.burn_in_transition_count
        q_value = self.r2d2_network.forward_sequence(
            states_tensor[training_starting_sequence_idx:],
            goal_states_tensor[training_starting_sequence_idx:],
            prev_actions_tensor[training_starting_sequence_idx:],
            prev_rewards_tensor[training_starting_sequence_idx:],
            (inital_hidden, inital_c),
        )

        current_seq_dim = q_value.shape[0]

        q_value_all = self.r2d2_network.forward_sequence(
            states_tensor,
            goal_states_tensor,
            prev_actions_tensor,
            prev_rewards_tensor,
            (inital_hidden, inital_c),
        )

        n_step_returns = torch.zeros(current_seq_dim, batch_size, device=self.device)
        bootstrap_q = torch.zeros(current_seq_dim, batch_size, device=self.device)

        for sequenze_idx in range(current_seq_dim):
            absolute_sequence_idx = training_starting_sequence_idx + sequenze_idx
            discount_factor = 1.0
            accumulated_return_value = 0.0
            for k in range(self.n_step):
                if (absolute_sequence_idx + k >= current_seq_dim): #  TODO should we ignore the burn in?
                    break
                accumulated_return_value += discount_factor * rewards_tensor[absolute_sequence_idx + k, :]
                discount_factor *= self.reward_discount

                is_episode_terminated = dones_tensor[absolute_sequence_idx + k, :].any()
                if (is_episode_terminated):
                    break

            n_step_returns[sequenze_idx] = accumulated_return_value

            bootstrap_idx = absolute_sequence_idx + self.n_step
            # the secound predicate is only good to check if the first predicate passes
            if (bootstrap_idx < current_seq_dim and not dones_tensor[bootstrap_idx - 1].any()):
                q_max = q_value_all[bootstrap_idx].argmax(dim=-1)
                with torch.no_grad():
                    target_q = self.r2d2_target_network.forward(
                        states_tensor[bootstrap_idx],
                        goal_states_tensor[bootstrap_idx],
                        q_max,
                        torch.zeros((batch_size,1), device=self.device), # target reward is pointless
                        (inital_hidden, inital_c)
                    )[0].squeeze(0)
                    
                q_max = q_max.unsqueeze(1)
                bootstrap_q[sequenze_idx] = target_q.gather(1, q_max).squeeze(1)
            else:
                bootstrap_q[sequenze_idx] = 0.0
        
        raw_target = n_step_returns + (self.reward_discount ** self.n_step) * bootstrap_q
        target = self._scale_target(raw_target)
        chosen_q_values = q_value.gather(2, actions_tensor[training_starting_sequence_idx:].unsqueeze(-1)).squeeze(-1)

        td_error = chosen_q_values - target
        loss = ((weights_tensor * td_error) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("Loss:", loss.detach().item())

        #full_error = np.zeros((self.sequence_length, batch_size), dtype=np.float32)
        #full_error[training_starting_sequence_idx:] = td_error.detach().cpu().numpy()
        #sequenze_errors = full_error.mean(axis=1)
        new_priorities = np.zeros_like(self.buffer.priorities) +1e-4
        #for idx, sequenze_error in zip(batch_starter_indicies, sequenze_errors):
        #    abs_sequenze_error = np.abs(sequenze_error)
        #    max_sequenze_error = np.max(abs_sequenze_error)
        #    mean_sequenze_error = np.mean(abs_sequenze_error)
        #    new_priority = (self.buffer_priority_eta * max_sequenze_error) + ((1.0 - self.buffer_priority_eta) * mean_sequenze_error) + 1e-4
        #    new_priorities[idx] = new_priority
        self.buffer.update_prioritites(batch_starter_indicies, new_priorities)

        self._step_count += 1
        if (self._step_count % self.target_update_step_count == 0):
            self.r2d2_target_network.load_state_dict(self.r2d2_network.state_dict())
        return loss

    # taken from paper, implemented with AI
    def _scale_target(self, raw_target: torch.Tensor) -> torch.Tensor:
        return torch.sign(raw_target) * (torch.sqrt(torch.abs(raw_target) + 1.0) - 1.0) + 1e-3 * raw_target 

    # taken from paper, implemented with AI
    def _descale_target(self, scaled_target: torch.Tensor) -> torch.Tensor:
        return torch.sign(scaled_target) * ((torch.abs(scaled_target) + 1.0) ** 2 - 1.0) / (1.0 + 1e-3)
    
    def _get_training_tensors(self, batch_size: int):
        batch_sequences, batch_starter_indicies, weights = self.buffer.sample(batch_size)

        max_sequence_len = max(len(sequence) for sequence in batch_sequences)
        batch_dim = len(batch_sequences)

        weights_tensor = self._to_tensor(weights)
        states_tensor = torch.zeros(max_sequence_len, batch_dim, self.state_dim, device=self.device)
        goal_states_tensor = torch.zeros(max_sequence_len, batch_dim, self.state_dim, device=self.device) 
        actions_tensor = torch.zeros(max_sequence_len, batch_dim, device=self.device, dtype=torch.long)
        rewards_tensor = torch.zeros(max_sequence_len, batch_dim, device=self.device)
        dones_tensor = torch.zeros(max_sequence_len, batch_dim, device=self.device)

        
        prev_actions_tensor = torch.zeros(max_sequence_len, batch_dim, device=self.device, dtype=torch.long)
        prev_rewards_tensor = torch.zeros(max_sequence_len, batch_dim, device=self.device)
        inital_hidden = torch.zeros(1, batch_dim, self.lstm_hidden_dim, device=self.device)
        inital_c = torch.zeros(1, batch_dim, self.lstm_hidden_dim, device=self.device)

        for batch_idx, sequence in enumerate(batch_sequences):
            for transition_idx, transition in enumerate(sequence):
                states_tensor[transition_idx, batch_idx] = self._to_tensor(transition.state)
                goal_states_tensor[transition_idx, batch_idx] = self._to_tensor(transition.goal_state)
                if (isinstance(transition.action, Action)):
                    actions_tensor[transition_idx, batch_idx] = transition.action.value
                else:
                    actions_tensor[transition_idx, batch_idx] = transition.action
                rewards_tensor[transition_idx, batch_idx] = transition.reward
                dones_tensor[transition_idx, batch_idx] = transition.done

                is_start_of_episode = transition_idx != 0
                if (not is_start_of_episode):
                    prev_transition = sequence[transition_idx - 1]
                else:
                    prev_transition = transition

                if (isinstance(prev_transition.action, Action)):
                    prev_actions_tensor[transition_idx, batch_idx] = prev_transition.action.value
                else:
                    prev_actions_tensor[transition_idx, batch_idx] = prev_transition.action
                prev_rewards_tensor[transition_idx, batch_idx] = prev_transition.reward
            
            first_transition = sequence[0]
            if first_transition.gru_hidden_states is not None:
                h0, c0 = first_transition.gru_hidden_states
                inital_hidden[:, batch_idx] = h0.squeeze(1)
                inital_c[:, batch_idx] = c0.squeeze(1)
        
        return weights_tensor, states_tensor, goal_states_tensor, actions_tensor, rewards_tensor, dones_tensor, prev_actions_tensor, prev_rewards_tensor, inital_hidden, inital_c, batch_starter_indicies
    
    def _burn_in_lstm(self,
                      states_tensor,
                      goal_states_tensor,
                      prev_actions_tensor,
                      prev_rewards_tensor,
                      inital_hidden,
                      inital_c) -> tuple[torch.Tensor, torch.Tensor]:
        if self.burn_in_transition_count > 0:
            batch_dim = states_tensor.shape[1]

            # burn in the r2d2 network
            _ = self.r2d2_network.forward_sequence(
                states_tensor[:self.burn_in_transition_count],
                goal_states_tensor[:self.burn_in_transition_count],
                prev_actions_tensor[:self.burn_in_transition_count],
                prev_rewards_tensor[:self.burn_in_transition_count],
                (inital_hidden, inital_c)
            )

            encoder_inputs_tensor = torch.cat((states_tensor, goal_states_tensor), dim=-1)
            # get correct hidden states for start of training
            _, (burn_hidden, burn_c) = self.r2d2_network.lstm_network(
                torch.cat(
                    [
                        self.r2d2_network.encoder_network(encoder_inputs_tensor[:self.burn_in_transition_count]).view(-1, self.r2d2_network.encoder_hidden_dim),
                        self.r2d2_network._action_one_hot(prev_actions_tensor[:self.burn_in_transition_count]).view(-1, self.r2d2_network.action_dim),
                        prev_rewards_tensor[:self.burn_in_transition_count].unsqueeze(-1).view(-1,1)
                    ],
                    dim=1
                ).view(self.burn_in_transition_count, batch_dim, -1),
                (inital_hidden, inital_c)
            )
            inital_hidden = burn_hidden
            inital_c = burn_c
        return inital_hidden, inital_c

    def _to_tensor(self, input: np.ndarray | Action | int | float, dtype: torch.dtype = torch.float32):
        if (not isinstance(input, np.ndarray)):
            input = np.array(input, ndmin=1)
        return torch.tensor(input, dtype=dtype, device=self.device)

import torch
import torch.nn as nn

class R2D2Network(nn.Module):
    def __init__(self,
                 state_dim: int,
                 encoder_hidden_dim: int,
                 action_dim: int,
                 lstm_hidden_dim: int,
                 dueling_hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.encoder_input_dim = self.state_dim * 2 # state + goal_state
        self.encoder_hidden_dim = encoder_hidden_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.dueling_hidden_dim = dueling_hidden_dim
        self.output_dim = 1

        self.lstm_input_dim = self.encoder_hidden_dim + self.action_dim + 1 # + 1 for the reward dim
        self.encoder_network = self._init_encoder_network()
        self.lstm_network = self._init_lstm_network()
        self.dueling_network_value_head = self._init_dueling_network_value_head()
        self.dueling_network_advantage_head = self._init_dueling_network_advantage_head()
    
    def _init_encoder_network(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.encoder_input_dim, self.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
            nn.ReLU(),
        )

    def _init_lstm_network(self) -> nn.LSTM:
        return nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=False # R2D2 traines on trajecories, thus expected is (Traj, B, Encodder_hidden)
        )
    
    def _init_dueling_network_value_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.dueling_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.dueling_hidden_dim, 1) # 1 as this should return the value float
        )

    def _init_dueling_network_advantage_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.dueling_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.dueling_hidden_dim, self.action_dim)
        )
    
    def forward(self,
                state_tensor: torch.Tensor,
                goal_tensor: torch.Tensor,
                prev_action_tensor: torch.Tensor,
                prev_reward_tensor: torch.Tensor,
                prev_gru_hidden_value_tensors: tuple[torch.Tensor, torch.Tensor] | None = None):

        #print("goal", goal_tensor)
        
        encoder_input = torch.cat((state_tensor, goal_tensor), dim=-1)
        encoded_state = self.encoder_network(encoder_input)
        one_hot_action = self._action_one_hot(prev_action_tensor)

        lstm_input_tensor = torch.cat(
            (encoded_state, one_hot_action.squeeze(0), prev_reward_tensor),
            dim=-1).unsqueeze(0) # gives us (Traj = 1, batch_dim, lstm_hidden)

        # This is a single step LSTM, we use this to get the value of a single action given current states
        lstm_output, new_hidden_tensors = self.lstm_network(lstm_input_tensor, prev_gru_hidden_value_tensors)

        value = self.dueling_network_value_head(lstm_output)
        advantage = self.dueling_network_advantage_head(lstm_output)
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)

        # print("Value:", value)
        # print("Advantage", advantage)
        # print("Q Value", q_value)

        return q_value, new_hidden_tensors

    # Preconditon that the passed tensors hold trajectores so [Traj, Batch, ...]
    def forward_sequence(self,
                state_tensor: torch.Tensor,
                goal_tensor: torch.Tensor,
                prev_action_tensor: torch.Tensor,
                prev_reward_tensor: torch.Tensor,
                prev_gru_hidden_value_tensors: tuple[torch.Tensor, torch.Tensor] | None = None):

        traj_dim, batch_dim, _ = state_tensor.shape

        encoder_input = torch.cat((state_tensor, goal_tensor), dim=-1)
        encoded_state = self.encoder_network(encoder_input.view(traj_dim * batch_dim, -1))
        encoded_state_flattened = encoded_state.view(traj_dim, batch_dim, -1)
        one_hot_action = self._action_one_hot(prev_action_tensor)
        
        lstm_input_tensor = torch.cat(
            (encoded_state_flattened, one_hot_action, prev_reward_tensor.unsqueeze(-1)),
            dim=-1) # gives us (Traj = 1, batch_dim, ...)

        # This is a multi step LSTM, we use this to get the value of a sequence of actions given the recorded following states
        # This is used for training only
        # therefore hiddens are not needed, as these are not stored in the buffer
        lstm_output, _ = self.lstm_network(lstm_input_tensor, prev_gru_hidden_value_tensors) 
        lstm_output_flattened = lstm_output.view(traj_dim * batch_dim, -1)

        value = self.dueling_network_value_head(lstm_output_flattened)
        advantage = self.dueling_network_advantage_head(lstm_output_flattened)

        q_value = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_value = q_value.view(traj_dim, batch_dim, -1)

        return q_value
    
    def _action_one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        action_shape = actions.shape + (self.action_dim,)
        one_hot = torch.zeros(action_shape, device=actions.device, dtype=torch.float32)
        one_hot.scatter_(-1, actions.unsqueeze(-1), 1.0)
        return one_hot

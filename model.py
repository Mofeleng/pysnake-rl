import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.linear_one = nn.Linear(in_size, hid_size)
        self.linear_two = nn.Linear(hid_size, out_size)


    def forward(self, x):
        x = F.relu(self.linear_one(x))
        x = self.linear_two(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, g):
        self.lr = lr
        self.model = model
        self.g = g
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # loss function

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            #Reshape to (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        # Get predicted Q value
        pred = self.model(state)

        target = pred.clone()

        # Qnew = r + g(max(next_predicted Q Value))
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.g * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad() #Empty the gradient
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleNet(nn.Module):
    
    def __init__(self, lr, n_actions, chkpnt_dir, name):
        super(SimpleNet, self).__init__()
        
        assert isinstance(chkpnt_dir, str)
        assert isinstance(name, str)
        assert isinstance(n_actions, int)
        
        self.num_actions = n_actions

        # layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4) # [batch_size, 3, 74, 240] -> [batch_size, 32, 17, 59]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # [batch_size, 32, 17, 59] -> [batch_size, 64, 7, 28]
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # [batch_size, 64, 7, 28] -> [batch_size, 64, 5, 26]

        self.fc1 = nn.Linear(8320, 512) # [batch_size, 64, 5, 26] -> flatten -> [batch_size, 8320, 512]
        self.fc2 = nn.Linear(512, n_actions) # [batch_size, 128] -> [batch_size, n_actions]
        
        # ativation
        self.relu = nn.ReLU()

        # etc
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.half = self.device.type != 'cpu'

        self.checkpoint_dir = chkpnt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        if not os.path.exists(chkpnt_dir):
            os.makedirs(chkpnt_dir)

    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x
    
    def forward(self, input):
        x = T.tensor(input).to(self.device).float()
        # x = x.half() if self.half else x.float()  # uint8 to fp16/32
        x /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
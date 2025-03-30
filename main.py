import torch
import torch.nn as nn
import chess
from get_dataset import board_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 8 * 8)
        return self.fc_layers(x)
    
model = ChessNet().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

try:
    model.load_state_dict(torch.load('model.pt'))
except:
    pass

def train():
    dataset = torch.load('dataset.pt', weights_only=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    epochs = 3
    for _ in range(epochs):
        for states, results in data_loader:
            states = states.to(device)
            results = results.to(device)

            pred = model(states)

            loss = loss_fn(pred, results)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss)

    torch.save(model.state_dict(), 'model.pt')

board = chess.Board('2b1kbnB/p1Bppp1p/1p6/8/3P4/6P1/PPP1PP1P/RN1QK1NR b KQkq - 0 1')
tensor = board_to_tensor(board)

with torch.inference_mode():
    print(model(tensor.unsqueeze(0)))
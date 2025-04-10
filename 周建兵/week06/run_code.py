from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter



# 模型构建
import torch.nn as nn
class MnistClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, modelType='RNN'):
        super().__init__()
        self.modelType = modelType
        if modelType == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif modelType == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif modelType == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        else:
            raise ValueError("Invalid modelType. Choose from 'RNN', 'LSTM', or 'GRU'.")        
        self.classifier = nn.Linear(in_features=hidden_size, out_features=num_labels)
    
    def forward(self, x):
        out, _ = self.rnn(x)  # out shape: (batch_size, seq_len, hidden_size)
        last_out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        logits = self.classifier(last_out)  # shape: (batch_size, num_labels)
        return logits
    
if __name__ == "__main__":    
    import torch
    # 数据准备
    BATCH_SIZE = 64
    train_dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    writer = SummaryWriter()
    
    #模型训练
    import torch.optim as optim   
    from tqdm import tqdm 

    EPOCHS = 5
    model_rnn = MnistClassifier(input_size=28, hidden_size=50, num_labels=10, modelType='RNN')
    model_lstm = MnistClassifier(input_size=28, hidden_size=50, num_labels=10, modelType='LSTM')
    model_gru = MnistClassifier(input_size=28, hidden_size=50, num_labels=10, modelType='GRU')
    
    model_ditectory = {
        'RNN': model_rnn,
        'LSTM': model_lstm,
        'GRU': model_gru
    }
    

    def runModel(modelType='RNN'):
        model = model_ditectory[modelType]
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
                model.train()
                tqbar = tqdm(train_dataloader)
                for idx,  (img,lbl) in enumerate(tqbar):
                    img = img.squeeze(1)
                    logits = model(img)
                    loss = criterion(logits, lbl)
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    if idx % 100 == 0:
                        tqbar.set_description(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}")
                        writer.add_scalar(f"Train Loss/{modelType}", loss.item(), epoch * len(train_dataloader) + idx)
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for img, lbl in train_dataloader:
                        img = img.squeeze(1)
                        logits = model(img)
                        _, pred = torch.max(logits, 1)
                        correct += (pred == lbl).sum().item()
                        total += lbl.size(0)
                    accuracy = correct / total
                    tqbar.set_description(f"Epoch {epoch+1}/{EPOCHS} Test Accuracy: {accuracy:.4f}")
                    writer.add_scalar(f"Test Accuracy/{modelType}", accuracy, epoch + 1)
                    
    runModel('RNN')
    runModel('LSTM')
    runModel('GRU')
    writer.close()                
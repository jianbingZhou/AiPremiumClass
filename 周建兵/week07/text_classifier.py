import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence  # 长度不同张量填充为相同长度
import jieba


def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])
        
    vocab =  ['PAD','UNK'] + list(vocab)  # PAD: padding, UNK: unknown
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx    
         

class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, lbl_num):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, lbl_num)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out   
    
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import os
    os.chdir(os.path.dirname(__file__))

    with open('./data/comments.pkl', 'rb') as f:
        comments = pickle.load(f)
    
    vocab = build_from_doc(comments)    
    print('词汇表大小:', len(vocab))
    
    embedded = nn.Embedding(len(vocab), 100, padding_idx=0)
    
    def collate_fn(batch_data):
        comments, stars = [],[]
        for comment, star in batch_data:
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in comment]))
            stars.append(star)
        
        commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD']) 
        label = torch.tensor(stars)
        return commt, label
           
    dataloader = DataLoader(comments, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_size = 128
    lbl_num = 2
    model = Comments_Classifier(vocab_size, embedding_dim, hidden_size, lbl_num)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    for epoch in range(epochs):
        for i, (commt, label) in enumerate(dataloader):
            commt = commt.to(device)
            label = label.to(device)
            
            output = model(commt)
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    
    # 保存模型
    torch.save(model.state_dict(), 'comments_classifier.pth')     
    # 保存词典
    torch.save(vocab, 'comments_vocab.pth')           
           

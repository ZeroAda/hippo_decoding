import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class MLP:
    def __init__(self, X, y, input_size=3, hidden_size=64, num_classes=5, batch_size=32, learning_rate=0.001, num_epochs=100):
        self.X = X
        self.y = y
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.model = self._init_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.train_loader, self.val_loader = self._prepare_data()

    def _init_model(self):
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, num_classes)
                self.softmax = nn.Softmax(dim=1)
            
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.softmax(out)
                return out

        return MLP(self.input_size, self.hidden_size, self.num_classes)

    def _prepare_data(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def train(self):
        self.model.train()
        loss_values = []
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            total_loss = 0
            for features, labels in tqdm(self.train_loader, desc='Training', leave=False):
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            loss_values.append(avg_loss)
            # print(f'Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {avg_loss:.4f}')

        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        # plt.show()
        plt.savefig('training_loss.pdf')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in self.val_loader:
                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the validation set: {accuracy} %')
        return accuracy

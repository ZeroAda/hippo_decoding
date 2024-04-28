import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

class MLP:
    def __init__(self, X, y, input_size, hidden_size=64, num_classes=5, batch_size=32, learning_rate=0.001, num_epochs=100, model_save_path='mlp_model.pth'):
        self.X = X
        self.y = y
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path

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
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
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
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            for features, labels in self.train_loader:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                val_accuracy = self.evaluate()
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')

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
        return accuracy

    def test(self, X_test, y_test):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        test_loader = DataLoader(dataset=TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)
        total, correct = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.view(-1).tolist())
                all_labels.extend(labels.view(-1).tolist())

        accuracy = 100 * correct / total
        self.plot_confusion_matrix(all_labels, all_preds)
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
        return accuracy

    def plot_confusion_matrix(self, true_labels, pred_labels):
        print(len(true_labels), len(pred_labels))
        cm = confusion_matrix(true_labels, pred_labels, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True),
        )
        # Decoder - Not used for classification, but necessary for training the Autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dim_feedforward=2048, nhead=8, num_layers=6):
        super(TransformerClassifier, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.final_layer = nn.Linear(input_dim, num_classes)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Transformer expects [seq_len, batch, features]
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Aggregate over sequence to get a single output per input
        output = self.final_layer(output)
        return output

class CombinedFramework:
    def __init__(self, X, y, input_dim, encoding_dim, num_classes, batch_size=32, learning_rate=0.001, num_epochs=100):
        self.X = X
        self.y = y
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.autoencoder = Autoencoder(input_dim, encoding_dim)
        self.classifier = TransformerClassifier(encoding_dim, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(list(self.autoencoder.parameters()) + list(self.classifier.parameters()), lr=self.learning_rate)

        self.train_loader, self.val_loader = self._prepare_data()

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
        loss_values = []
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            total_loss = 0
            for features, labels in tqdm(self.train_loader, desc='Training', leave=False):
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Autoencoder reconstruction
                reconstructed = self.autoencoder(features)
                reconstruction_loss = nn.MSELoss()(reconstructed, features)

                # Encoding and classification
                encoded_features = self.autoencoder.encoder(features)
                encoded_features = encoded_features.unsqueeze(1)  # Add sequence dimension for transformer
                class_outputs = self.classifier(encoded_features)
                classification_loss = self.criterion(class_outputs, labels)

                # Combined loss
                loss = reconstruction_loss + classification_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            loss_values.append(avg_loss)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {avg_loss:.4f}')
            # if loss decreases, store the model
            if len(loss_values) > 1 and loss_values[-1] < loss_values[-2] and epoch == 0:
                torch.save(self.autoencoder.state_dict(), 'autoencoder.pth')
                torch.save(self.classifier.state_dict(), 'classifier.pth')

        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('combined_training_loss.png')

    def evaluate(self):
        self.autoencoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in self.val_loader:
                encoded_features = self.autoencoder.encoder(features)
                encoded_features = encoded_features.unsqueeze(1)  # Add sequence dimension for transformer
                outputs = self.classifier(encoded_features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the validation set: {accuracy} %')
        return accuracy
        
    
class TransformerFramework:
    def __init__(self, X, y, input_dim, num_classes, batch_size=32, learning_rate=0.001, num_epochs=100):
        self.X = X
        self.y = y
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.classifier = TransformerClassifier(input_dim, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

        self.train_loader, self.val_loader = self._prepare_data()

    def _prepare_data(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension for transformer
        X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def train(self):
        loss_values = []
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            total_loss = 0
            for features, labels in tqdm(self.train_loader, desc='Training', leave=False):
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Direct classification
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            loss_values.append(avg_loss)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {avg_loss:.4f}')

        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('training_loss.png')

    def evaluate(self):
        self.classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in self.val_loader:
                outputs = self.classifier(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the validation set: {accuracy} %')
        return accuracy
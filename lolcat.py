import torch
import torch.nn as nn

from torch_scatter import scatter_add
from torch_geometric.utils import softmax
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import *
from torch_geometric.data import DataLoader




####################
# Simple MLP model #
####################
class MLP(nn.Module):
    r"""Multi-layer perceptron model, with optional batchnorm layers.

    Args:
        hidden_layers (list): List of layer dimensions, from input layer to output layer. If first input size is -1,
            will use a lazy layer.
        bias (boolean, optional): If set to :obj:`True`, bias will be used in linear layers. (default: :obj:`True`).
        activation (torch.nn.Module, optional): Activation function. (default: :obj:`nn.ReLU`).
        batchnorm (boolean, optional): If set to :obj:`True`, batchnorm layers are added after each linear layer, before
            the activation (default: :obj:`False`).
        drop_last_nonlin (boolean, optional): If set to :obj:`True`, the last layer won't have activations or
            batchnorm layers. (default: :obj:`True`)

    Examples:
        >>> m = MLP([-1, 16, 64])
        MLP(
          (layers): Sequential(
            (0): LazyLinear(in_features=0, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
          )
        )
    """
    def __init__(self, hidden_layers, *, bias=True, activation=nn.ReLU(True), batchnorm=False, drop_last_nonlin=True, dropout=0.):
        super().__init__()

        # build the layers
        layers = []
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            if in_dim == -1:
                layers.append(nn.LazyLinear(out_dim, bias=bias))
            else:
                layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features=out_dim))
            if activation is not None:
                layers.append(activation)
            if dropout > 0.:
                layers.append(nn.Dropout(dropout))

        # remove activation and/or batchnorm layers from the last block
        if drop_last_nonlin:
            remove_layers = -(int(activation is not None) + int(batchnorm) + int(dropout>0.))
            if remove_layers:
                layers = layers[:remove_layers]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


#####################
# Attention Pooling #
#####################
class MultiHeadGlobalAttention(torch.nn.Module):
    """Multi-Head Global pooling layer."""
    def __init__(self, in_channels, out_channels, heads=1):
        super(MultiHeadGlobalAttention, self).__init__()
        self.heads = heads

        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, heads * in_channels, bias=False),
            nn.PReLU(),
            nn.Linear(heads * in_channels, heads, bias=False),
        )

        self.nn = MLP([in_channels, out_channels * heads, out_channels * heads])


    def forward(self, x, batch, return_attention=False):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1

        gate = self.gate_nn(x)
        # print(x.shape, gate.shape, size, batch.shape)
        # reshape batch tensor into [32,-1]
        batch_index = batch.view(size,-1)

        # x = self.nn(x).view(x.size(0), self.heads, -1)
        x = self.nn(x).view(x.size(0), self.heads, batch_index.size(1), -1)

        # print("after nn", x.shape)
        # print(batch_index)

        score = softmax(gate, batch_index[:,0], num_nodes=size, dim=0)
        score = score.unsqueeze(-1)
        # print("score", score.shape, "x", x.shape)

        out = scatter_add(score * x, batch_index, dim=0, dim_size=size)

        out = out.view(size, -1)
        # print("out", out.shape)

        if not return_attention:
            return out
        else:
            return out, gate, score

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


################
# LOLCAT model #
################
class LOLCAT(nn.Module):
    """LOLCAT model. It consists of an encoder, a pooling layer and a classifier. The pooling layer is a multi-head
    global attention layer, which computes a global embedding for each cell. Both the encoder and the classifier 
    are MLPs."""
    def __init__(self, encoder, classifier, pool):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.pool = pool

    def forward(self, x, batch, return_attention=False):
        """Forward pass. If return_attention is True, the attention scores and gate values are returned as well.
        
        Args:
            x (torch.Tensor): Input tensor of shape (num_nodes, num_features).
            batch (torch.Tensor): Batch tensor of shape (num_nodes,).
            return_attention (bool, optional): If set to :obj:`True`, the attention scores and gate values are returned.
                (default: :obj:`False`).
        """
        emb = self.encoder(x)  # all trial sequences are encoded

        # compute global cell-wise embedding
        if return_attention:
            global_emb, gate, score = self.pool(emb, batch, return_attention=True)
        else:
            global_emb = self.pool(emb, batch)

        # classify
        logits = self.classifier(global_emb)

        if return_attention:
            return logits, {'global_emb': global_emb, 'attention': score, 'gate': gate}
        else:
            return logits, {'global_emb': global_emb}
        
class LOLCARTrainer:
    def __init__(self, processed_data,channel_index_label, label_index, input_size=128, hidden_size=64, num_classes=5, batch_size=32, learning_rate=0.001, num_epochs=200, model_save_path='mlp_model_6.pth'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.processed_data = processed_data
        self.channel_index_label = channel_index_label
        self.label_index = label_index

        self.model = self._init_model()
        self.train_dataset, self.eval_dataset, self.train_loader, self.eval_loader = self._prepare_data()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _init_model(self):
        encoder = MLP([-1,64,32,16])
        classifier = MLP([-1,16,5])
        pool = MultiHeadGlobalAttention(in_channels=16, out_channels=16, heads=1)
        model = LOLCAT(encoder, classifier, pool)
        return model
    
    def _prepare_data(self):

        train_indices, eval_indices = train_test_split_indices(len(self.processed_data), test_size=0.2, random_seed=42)
        temp = np.array(self.processed_data)
        train_dataset = CustomDataset(list(temp[train_indices]), self.channel_index_label, self.label_index)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        eval_dataset = CustomDataset(list(temp[eval_indices]), self.channel_index_label, self.label_index)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)

        return train_dataset, eval_dataset, train_loader, eval_loader
    
    def train(self):
        self.model.train()
        loss_values = []
        best_accuracy = 0
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            total_loss = 0
            for data in tqdm(self.train_loader):
                x, batch, target = torch.Tensor(np.array(data.x)), data.batch, data.y
                # print("batches", x, batch, target)
                logits, globel_emb = self.model(x, batch)
                # print("outputs", logits, "target",target)
                loss = self.criterion(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            loss_values.append(avg_loss)
            
            if epoch % 10 == 0:
                val_accuracy = self.evaluate()
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f'Saved best model with accuracy: {best_accuracy:.2f}%')
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
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
            for data in tqdm(self.eval_loader):
                x, batch, target = torch.Tensor(np.array(data.x)), data.batch, data.y
                logits, globel_emb = self.model(x, batch)
                _, predicted = torch.max(logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the validation set: {accuracy} %')
        return accuracy

    # def test(self, X_test, y_test):
    #     X_test = torch.tensor(X_test, dtype=torch.float32)
    #     y_test = torch.tensor(y_test, dtype=torch.long)
    #     test_loader = DataLoader(dataset=TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)
    #     self.model.eval()
    #     with torch.no_grad():
    #         correct = 0
    #         total = 0
    #         for features, labels in test_loader:
    #             outputs = self.model(features)
    #             _, predicted = torch.max(outputs, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #     accuracy = 100 * correct / total
    #     print(f'Accuracy of the model on the test set: {accuracy} %')
    #     return accuracy


    
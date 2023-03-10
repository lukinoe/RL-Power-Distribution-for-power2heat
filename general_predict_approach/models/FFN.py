
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class FFN():

    def __init__(self, nn_params) -> None:
        self.nn_params = nn_params


    def initModel(self, n_features):

        if self.nn_params["activation1"] == "relu": activation1 = nn.ReLU()
        if self.nn_params["activation1"] == "sigmoid": activation1 = nn.Sigmoid()
            

        if self.nn_params["activation2"] == "relu": activation2 = nn.ReLU()
        if self.nn_params["activation2"] == "sigmoid": activation2 = nn.Sigmoid()


        model = nn.Sequential(
                nn.Linear(n_features, self.nn_params["n_hidden1"]),
                activation1,
                nn.Linear(self.nn_params["n_hidden1"], self.nn_params["n_hidden2"]),
                activation2,
                nn.Linear(self.nn_params["n_hidden2"], 1)
        )

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform(module.weight.data)
                module.bias.data.fill_(0.0)

        return model


            
    def train(self, X_train, y_train, X_test, y_test):


        n_features = X_train.shape[1]
        nn_params = self.nn_params

        model = self.initModel(n_features)
        
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=nn_params["lr"])


        train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        valid_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

        train_loader = DataLoader(train_data, batch_size=self.nn_params["batch_size"])
        valid_loader = DataLoader(valid_data, batch_size=self.nn_params["batch_size"])

            

            # set up training parameters
        n_epochs = nn_params["epochs"]

        train_losses = []
        valid_losses = []

        # train the model
        for epoch in range(n_epochs):
            epoch_train_loss = 0
            epoch_valid_loss = 0

            # train
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                    # forward pass
                outputs = model(inputs)

                    # compute loss
                loss = loss_fn(outputs, labels)

                    # backward pass and update weights
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            # validation
            model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(valid_loader):
                    # forward pass
                    outputs = model(inputs)

                    # compute loss
                    loss = loss_fn(outputs, labels)

                    epoch_valid_loss += loss.item()

            train_losses.append(epoch_train_loss / len(train_loader))
            valid_losses.append(epoch_valid_loss / len(valid_loader))

            # verbose output
            print(f"Epoch {epoch+1}, train loss {train_losses[-1]:.4f}, valid loss {valid_losses[-1]:.4f}")

    

        model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in valid_loader:
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.concatenate(predictions)

        return model, predictions



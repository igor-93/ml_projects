import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyNN():

    def __init__(self, n_classes):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.model = None

    def fit(self, X, y, eval_set):
        n, d = X.shape

        X_train = torch.from_numpy(X).to(device)
        y_train = torch.from_numpy(y).to(device)

        X_val = torch.from_numpy(eval_set[0][0]).to(device)
        y_val = torch.from_numpy(eval_set[0][1]).to(device)

        n_epochs = 5 # 00
        batch_size = 256  # 512

        layer1 = torch.nn.Linear(d, 2048)
        layer2 = torch.nn.Linear(2048, 1024)
        layer3 = torch.nn.Linear(1024, self.n_classes)

        torch.nn.init.xavier_normal_(layer1.weight)
        torch.nn.init.xavier_normal_(layer2.weight)
        torch.nn.init.xavier_normal_(layer2.weight)

        # Use the nn package to define our model and loss function.
        self.model = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            layer2,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            layer3,
        ).to(device)

        learning_rate = 1e-5
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        epoch_of_min_loss = -1
        min_loss = 1e6

        for epoch in range(n_epochs):
            print(f"Epoch: {epoch + 1}/{n_epochs}")

            permutation = torch.randperm(n)
            for i in range(0, n, batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                # in case you wanted a semi-full example
                outputs = self.model.forward(batch_x)
                loss = self.loss_fn(outputs, batch_y)

                loss.backward()
                optimizer.step()

            print(f"Train loss at epoch {epoch}: {loss.item()}")

            y_val_pred = self.model.forward(X_val)
            val_loss = self.loss_fn(y_val_pred, y_val)
            print(f"Validation loss at step {epoch}: {val_loss.item()}")
            print()

            if val_loss < min_loss:
                min_loss = val_loss
                epoch_of_min_loss = epoch

            if epoch >= 10 + epoch_of_min_loss:
                print(f"Early stopping, min {min_loss} reached at epoch {epoch_of_min_loss}")
                break

    def predict_proba(self, X):
        n = X.shape[0]
        batch_size = 2048
        X = torch.from_numpy(X)

        output = []
        for i in range(0, n, batch_size):
            X_batch = X[i:i + batch_size].to(device)
            output_batch = self.model.forward(X_batch).to("cpu").detach().numpy()

            output.append(output_batch)

        output = np.vstack(output)
        return output


def train_nn(X_train, y_train, X_val, y_val, n_classes):
    n, d = X_train.shape

    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)

    X_val = torch.from_numpy(X_val).to(device)
    y_val = torch.from_numpy(y_val).to(device)

    n_epochs = 500
    batch_size = 256 # 512

    layer1 = torch.nn.Linear(d, 2048)
    layer2 = torch.nn.Linear(2048, 1024)
    layer3 = torch.nn.Linear(1024, n_classes)

    torch.nn.init.xavier_normal_(layer1.weight)
    torch.nn.init.xavier_normal_(layer2.weight)
    torch.nn.init.xavier_normal_(layer2.weight)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        layer1,
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.4),
        layer2,
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.4),
        layer3,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_of_min_loss = -1
    min_loss = 1e6

    for epoch in range(n_epochs):
        print(f"Epoch: {epoch+1}/{n_epochs}")

        permutation = torch.randperm(n)
        for i in range(0, n, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # in case you wanted a semi-full example
            outputs = model.forward(batch_x)
            loss = loss_fn(outputs, batch_y)

            loss.backward()
            optimizer.step()

        print(f"Train loss at epoch {epoch}: {loss.item()}")

        y_val_pred = model(X_val)
        val_loss = loss_fn(y_val_pred, y_val)
        print(f"Validation loss at step {epoch}: {val_loss.item()}")
        print()

        if val_loss < min_loss:
            min_loss = val_loss
            epoch_of_min_loss = epoch

        if epoch >= 10 + epoch_of_min_loss:
            print(f"Early stopping, min {min_loss} reached at epoch {epoch_of_min_loss}")
            break

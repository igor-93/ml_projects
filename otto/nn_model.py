import torch
import numpy as np

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    raise AssertionError("CUDA NOT AVAILABLE")
device = torch.device(device)


class MyNN:

    def __init__(self, n_classes):
        self.loss_fn = torch.nn.NLLLoss()
        self.n_classes = n_classes
        self.model = None
        self.n_epochs = 500
        self.batch_size = 32  # 64 # even better 128 # better 256 # worse 512
        self.learning_rate = 1e-5
        self.l2 = 1e-4  # 1e-4
        print("n_epochs: ", self.n_epochs)
        print("batch_size: ", self.batch_size)
        print("learning_rate: ", self.learning_rate)
        print("l2: ", self.l2)

    def fit(self, X, y, eval_set):
        n, d = X.shape

        X_train = torch.from_numpy(X).to(device)
        y_train = torch.from_numpy(y).to(device)

        X_val = torch.from_numpy(eval_set[0][0]).to(device)
        y_val = torch.from_numpy(eval_set[0][1]).to(device)

        layer1 = torch.nn.Linear(d, 2048)
        layer2 = torch.nn.Linear(2048, 512)
        layer3 = torch.nn.Linear(512, self.n_classes)
        # layer1 = torch.nn.Linear(d, 1024)
        # layer2 = torch.nn.Linear(1024, 512)
        # layer3 = torch.nn.Linear(512, 256)
        # layer4 = torch.nn.Linear(256, self.n_classes)

        torch.nn.init.xavier_normal_(layer1.weight)
        torch.nn.init.xavier_normal_(layer2.weight)
        torch.nn.init.xavier_normal_(layer3.weight)
        #torch.nn.init.xavier_normal_(layer4.weight)

        self.model = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            layer2,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            layer3,
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.4),
            # layer4,
            torch.nn.LogSoftmax(dim=1)
        ).to(device)
        # Use the nn package to define our model and loss function.

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2)

        epoch_of_min_loss = -1
        min_loss = 1e6

        for epoch in range(self.n_epochs):
            print_epoch = False
            if epoch % 10 == 0:
                print_epoch = True
                print(f"Epoch: {epoch + 1}/{self.n_epochs}")

            permutation = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                # in case you wanted a semi-full example
                log_probs = self.model.forward(batch_x)
                assert log_probs.shape[0] == batch_x.shape[0], f"Shape mismatch: {log_probs.shape} vs {batch_x.shape}"
                loss = self.loss_fn(log_probs, batch_y)

                loss.backward()
                optimizer.step()

            y_val_pred = self.model.forward(X_val)
            val_loss = self.loss_fn(y_val_pred, y_val)
            if print_epoch:
                print(f"Train loss at epoch {epoch}: {loss.item()}")
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
            log_probs = self.model.forward(X_batch).to("cpu").detach().numpy()
            probs = np.exp(log_probs)
            if len(probs) > 0:
                sum_1 = np.abs(np.sum(probs, axis=1) - 1.0)
                assert (sum_1 < 1e-4).all(), print(sum_1)
                probs = probs / np.sum(probs, axis=1)[:, np.newaxis]

                output.append(probs)

        output = np.vstack(output)
        return output


import torch

class BasicUtils():
    def __init__(self):
        super().__init__()

    def model_saver(self, model, path):
        torch.save(model, path)
        print("Model saved. Name: " + path + "")

    def device_chooser(self):
        if torch.cuda.is_available():
            device = "cuda" # NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = "mps" # Apple GPU
        else:
            device = "cpu" # CPU

        return device
    
    def loss_chooser(self, loss):
        if loss == "crossentropy":
            loss_fn = torch.nn.CrossEntropyLoss()

        else:
            print("Loss function not found! Using Cross Entropy Loss for default loss function.")
            loss_fn = torch.nn.CrossEntropyLoss()

        return loss_fn

    def optim_chooser(self, optim, model, learning_rate):
        if optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        elif optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        elif optim == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        elif optim == "adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

        else:
            print("Optimizer not found! Using SGD for default optimizer.")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
        return optimizer
    
class TrainTestUtils():
    def __init__(self):
        super().__init__()

    def train(self, dataloader, model, loss_fn, optimizer, epoch, num_epochs, batch_size):
        total_loss = 0
        device = BasicUtils().device_chooser()
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().numpy()

            if batch % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, ((batch+1)*len(X)), size, loss.item()))

        return total_loss/(size//batch_size)

    def test(self, dataloader, model, loss_fn):
        device = BasicUtils().device_chooser()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss


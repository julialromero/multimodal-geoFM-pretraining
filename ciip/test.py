import torch 

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.cuda.set_device(1)

def test_model(dataloader, model, loss_fn, val=False):
    print("Running test_model function...")
    model.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        i = 0
        for sample in dataloader:
            print("Running sample number ", i, "...")
            i += 1
            X, y = sample['image'], sample['label']
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    prefix = "Validation" if val else "Test"
    print(f"{prefix} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

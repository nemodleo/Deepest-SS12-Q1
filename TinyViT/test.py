import torch
from tqdm import tqdm

def test(model, device, test_dataloader):
    model.eval()

    # test loop
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct / float(total)} %')
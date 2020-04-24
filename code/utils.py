import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchsampler import ImbalancedDatasetSampler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

def imagesToDataloader(path,
                    batch_size=32,
                    trans=None,
                    balance=False,
                    train=True):

    print('Mode: Training is {}'.format(train))
    # Define transforms
    if trans is None:
        trans = transforms.Compose([
                            transforms.Resize(224),
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
    # Create an ImageFolder
    dataset = ImageFolder(path, transform=trans)
    classToIdx = dataset.class_to_idx


    # Create a dataloader        
    if train and balance:
        sampler = ImbalancedDatasetSampler(dataset)
    elif train and not balance:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    print('Dataloader created')
    return dataloader, classToIdx


def evaluate(model, dataloader, device, pos_label=1):
    model.eval()
    predProbList, predList, trueList = [], [], []
    print('Evaluating...')
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(dataloader)):
            # Load a batch from dataloader
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            out = model(inputs)
            # Save prediction & predicted probability of (+)
            out = F.softmax(out, dim=1)
            pred = out.max(1)[1]
            predProb = out[:, pos_label]
            predList.extend(pred.detach().tolist())
            predProbList.extend(predProb.detach().tolist())
            # Save ground truth
            trueList.extend(labels.detach().tolist())
    # Compute accuracy
    acc = accuracy_score(trueList, predList)
    # Compute other metrics
    precision, recall, fscore, _ = precision_recall_fscore_support(trueList, predList, average='binary', pos_label=pos_label)
    
    return predProbList, predList, trueList, acc, precision, recall, fscore
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import cm


def load_model(label=0):
    """
    loading the saved model
    """
    model = torch.load(f"model/torch_model_class_{label}.pth")
    model = model.cuda()
    model.eval()
    model = model.cuda()
    # print(model)
    return model


def load_data():
    """
    load and prepare data from cifar dataset
    """
    transform = transforms.Compose([transforms.ToTensor()])
    data = CIFAR10(".", train=False, download=True, transform=transform)
    # test_sample = torch.unsqueeze(data.data[:32].clone().cuda(),1).float()
    plt.figure()
    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.imshow(data.data[i], cmap='gray')
        plt.axis('off')
        plt.title(data.targets[i])
    plt.show()
    sample_data, _ = torch.utils.data.random_split(data, [2000, 8000])
    dataloader = DataLoader(dataset=sample_data, batch_size=32)
    return dataloader


def extract_embeddings(model, dataloader):
    """
    Using dataloader to extract the embeddings for a specific layer
    """

    # Define your output variable that will hold the output
    out = None

    # Define a hook function. It sets the global out variable equal to the
    # output of the layer to which this hook is attached to.
    def hook(module, input, output):
        global out
        out = output
        return None

    # Your model layer has a register_forward_hook that does the registering for you
    model.backbone.layer4.register_forward_hook(hook)

    # Then you just loop through your dataloader to extract the embeddings
    embeddings = np.zeros(shape=(0, 512))
    labels = np.zeros(shape=(0))

    for x, y in iter(dataloader):
        # global out
        x = x.cuda()
        model(x)
        labels = np.concatenate((labels, y.numpy().ravel()))
        embeddings = np.concatenate(
            [embeddings, out.detach().cpu().numpy().reshape(-1, 512)], axis=0)
    return embeddings, labels


def dimentions_reduction(embeddings):
    """
    Create a two dimensional t-SNE projection of the embeddings
    """
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)
    return tsne_proj


def plot_features(tsne_proj):
    """
    Plot TSNE Projection
    """
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(18, 18))
    num_categories = 10
    # for lab in range(num_categories):
    # #     print(lab)
    #   indices=labels==lab
    #   ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab+1)).reshape(1,4), label = lab ,alpha=0.8)

    lab = 0
    indices = labels != lab
    lab = 0
    ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(
        cmap(lab)).reshape(1, 4), label="neg", alpha=0.5)

    indices = labels == lab
    lab = 1
    ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(
        cmap(lab+1)).reshape(1, 4), label="pos", alpha=0.5)

    ax.legend(fontsize='large', markerscale=2)
    plt.savefig('visualize_features.png')
    plt.show()


if __name__ == '__main__':

    model = load_model(label=0)
    dataloader = load_data()

    embeddings, labels = extract_embeddings(model, dataloader)

    tsne_proj = dimentions_reduction(embeddings)

    plot_features(tsne_proj)

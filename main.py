import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_preprocessing import load_data, augment_data, split_data
from model_training import train_model

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# EDA: Display sample images
def display_sample_images(train_loader, classes):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# EDA: Visualizing distribution of classes
def visualize_class_distribution(train_set):
    classes = train_set.classes
    class_counts = [0] * len(classes)
    for _, label in train_set:
        class_counts[label] += 1

    plt.figure(figsize=(10, 5))
    sns.barplot(x=classes, y=class_counts)
    plt.title('Distribution of Classes in CIFAR-10 Training Set')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# EDA: Visualizing images from each class
def visualize_images_per_class(train_set):
    classes = train_set.classes
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Sample Images from Each Class')

    for i, ax in enumerate(axes.flatten()):
        idx = next(j for j, label in enumerate(train_set.targets) if label == i)
        img, label = train_set[idx]
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(classes[label])
        ax.axis('off')

    plt.show()

# EDA: Applying PCA
def apply_pca_and_visualize(train_loader, classes):
    all_data = []
    all_labels = []

    for images, labels in train_loader:
        all_data.append(images)
        all_labels.append(labels)

    all_data = torch.cat(all_data).view(-1, 3 * 32 * 32)
    all_labels = torch.cat(all_labels)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_data.numpy())

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.title('PCA of CIFAR-10 Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def main():
    # Load and preprocess data
    train_loader, test_loader = load_data()
    train_set = train_loader.dataset

    # EDA: Display sample images
    display_sample_images(train_loader, train_set.classes)

    # EDA: Visualizing distribution of classes
    visualize_class_distribution(train_set)

    # EDA: Visualizing images from each class
    visualize_images_per_class(train_set)

    # EDA: Applying PCA
    apply_pca_and_visualize(train_loader, train_set.classes)

    # Data augmentation
    augmented_train_set = augment_data(train_set)
    train_data, val_data = split_data(augmented_train_set)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=100, shuffle=False)

    # Train and evaluate model
    model = train_model(train_loader, val_loader)

if __name__ == '__main__':
    main()

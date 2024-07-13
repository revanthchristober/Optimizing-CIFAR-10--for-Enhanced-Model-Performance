# **Optimizing CIFAR-10 for Enhanced Model Performance**

## **Overview**
This project aims to optimize the CIFAR-10 dataset for better model performance through data analysis, preprocessing, augmentation, and model training. The project combines data exploration, augmentation techniques, and a simple CNN model to improve classification accuracy on the CIFAR-10 dataset.

## **Project Structure**
```
project/
│
├── data/data_preprocessing.py   # Functions for data loading, augmentation, and splitting
├── model/model_training.py       # Defines the CNN model and training routine
├── notebooks/EDA.ipynb         # Contains the EDA of the CIFAR-10 dataset.
├── utils/utils.py                # Utility functions
├── main.py                 # Main script to run the entire pipeline
├── requirements.txt        # List of required Python packages
└── README.md               # Project documentation
```

## **Setup Instructions**

### Requirements
- Python 3.6+
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/revanthchristober/Optimizing-CIFAR-10--for-Enhanced-Model-Performance.git
    cd Optimizing-CIFAR-10--for-Enhanced-Model-Performance
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### How to Run
1. Run the main script to execute the entire pipeline:
    ```sh
    python main.py
    ```

This will:
- Load and preprocess the CIFAR-10 dataset.
- Perform exploratory data analysis (EDA).
- Augment the training data.
- Split the data into training and validation sets.
- Train and evaluate a simple CNN model.

## **Detailed Explanation**

### Data Preprocessing

#### `data_preprocessing.py`
- **load_data()**: Loads the CIFAR-10 dataset with basic transformations.
- **augment_data()**: Applies data augmentation techniques like random horizontal flip and random cropping to the training set.
- **split_data()**: Splits the training set into training and validation subsets.

### Model Training

#### `model_training.py`
- **SimpleCNN**: Defines a simple convolutional neural network with two convolutional layers, two fully connected layers, and ReLU activations.
- **train_model()**: Trains the CNN model using the training data and evaluates it on the validation data.

### Utility Functions

#### `utils.py`
- **calculate_mean_std()**: Calculates the mean and standard deviation of the dataset, useful for data normalization.

### Exploratory Data Analysis (EDA)

#### `main.py`
- **imshow(img)**: Helper function to display images.
- **display_sample_images(train_loader, classes)**: Displays a batch of sample images from the training set.
- **visualize_class_distribution(train_set)**: Plots the distribution of classes in the training set.
- **visualize_images_per_class(train_set)**: Shows one sample image from each class.
- **apply_pca_and_visualize(train_loader, classes)**: Applies PCA to reduce the data to 2 dimensions and visualizes it in a scatter plot.

### Running the Pipeline
The `main.py` script integrates all the steps:
1. Loads and preprocesses the CIFAR-10 data.
2. Performs EDA, including displaying sample images, visualizing class distributions, and applying PCA.
3. Augments the data and splits it into training and validation sets.
4. Trains the CNN model using the augmented training data and evaluates it on the validation set.

### Results
The results of the training process, including loss and accuracy for each epoch, are printed to the console. The EDA visualizations help understand the data distribution and the effect of PCA on the high-dimensional data.

## **Conclusion**
This project demonstrates the importance of data preprocessing and augmentation in improving model performance. By combining EDA, augmentation, and a simple CNN model, we aim to achieve better classification accuracy on the CIFAR-10 dataset.

## **References**
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

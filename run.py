from perceptron import Perceptron
from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np

def run_perceptron(dataset_name):
    """
    Accepts a dataset and returns the score of the
    result on testing and training data.
    """
    # Set up hyperparameters
    hyperparameters = {
        'learning_rate': 0.1,
        'epochs': 10
    }

    # Load dataset
    dataset = io.loadmat(dataset_name)
    X_train = dataset['training_data']
    y_train = dataset['training_labels']

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    # Initialize perceptron
    perceptron = Perceptron(hyperparameters)
    perceptron.fit(X_train, y_train)
    y_predicted = perceptron.predict(X_test)
    score = np.sum(y_predicted == y_test) / y_test.shape[0]
    print(f'Score: {score}')


if __name__ == "__main__":
    run_perceptron('data/mnist_data.mat')


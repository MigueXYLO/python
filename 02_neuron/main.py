from percepron import Perceptron  # Importing the Perceptron class from the perceptron module
import numpy as np  # Importing the numpy library and aliasing it as np

if __name__ == '__main__':
    # Defining the training data as a numpy array
    training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Defining the corresponding training labels as a numpy array
    #and
    #training_label = np.array([-1, -1, -1, 1])
    #or
    #training_label = np.array([-1, 1, 1, 1])
    #xor
    training_label = np.array([0, 1, 1, 0])

    # Creating an instance of the Perceptron class
    perceptron = Perceptron()

    # Training the perceptron using the fit() method
    # The learning_rate parameter is set to 0.1 and the seed parameter is set to 1337
    perceptron.fit(training_data, training_label, learning_rate=0.1, seed=1337)

    # Looping through each sample in the training data
    for sample in training_data:
        # Predicting the output for each sample using the predict() method
        output = perceptron.predict(sample)
        
        # Printing the sample and its corresponding output
        print(str(sample) + ' ' + str(output))
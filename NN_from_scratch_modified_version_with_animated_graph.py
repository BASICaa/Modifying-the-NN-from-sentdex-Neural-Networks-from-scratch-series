import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

# class for creatinng neurons
class layer_dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
# activation for A rectified linear unit (ReLU)
class Activate_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)
#sigmoid Activation (still need some modification)
class Activate_sigmoid:
	def forward(self, inputs):
		self.output = 1/(1 + np.exp(-inputs))
#softmax activation
class Activate_Softmax:
	def forward(self, inputs):
		self.exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		self.probabilities = self.exp_values / np.sum(self.exp_values, axis=1, keepdims=True)
		self.output = self.probabilities
#calculating the loss
class Loss:
	def Calculate_loss(self, output, y):
		sample_loss = self.forward(output, y)
		data_loss = np.mean(sample_loss)
		return data_loss
	
class Loss_CategoricaalCrossEntropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
		
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

#this version of NN in the video did not have a acceptable accuracy, However, with changes like
#increasing sample size and increasing hidden layers and neurons the accruacy have been increased significantly

X, y = vertical_data(samples=150, classes=3)

dense_1 = layer_dense(2, 5)
activation_1 = Activate_ReLU()

dense_2 = layer_dense(5, 5)
activation_2 = Activate_ReLU()

dense_3 = layer_dense(5, 3)
activation_3 = Activate_Softmax()

loss_function = Loss_CategoricaalCrossEntropy()

highest_accuracy = 0
lowest_loss = 99999999
best_dense1_weights = dense_1.weights.copy()
best_dense1_biases = dense_1.biases.copy()
best_dense2_weights = dense_2.weights.copy()
best_dense2_biases = dense_2.biases.copy()
best_dense3_weights = dense_3.weights.copy()
best_dense3_biases = dense_3.biases.copy()
loss_history = []
accuracy_history = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
batch_size = 100

#by doing this we have a plot that shows the result frame by frame

def update(frame):
    global lowest_loss, best_dense1_weights, best_dense1_biases, best_dense2_weights, best_dense2_biases, best_dense3_weights, best_dense3_biases, RealTime_accuracy
    #becuase of amount of time the processing consumed by batching it, it took alot less but still it is slow
    for batch_start in range(0, len(X), batch_size):
	    #generating weights and biases randomly
        dense_1.weights += 0.05 + np.random.randn(2, 5)
        dense_1.biases += 0.05 + np.random.randn(1, 5)
        dense_2.weights += 0.05 + np.random.randn(5, 5)
        dense_2.biases += 0.05 + np.random.randn(1, 5)
        dense_3.weights += 0.05 + np.random.randn(5, 3)
        dense_3.biases += 0.05 + np.random.randn(1, 3)
        #getting activation
        dense_1.forward(X)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        activation_2.forward(dense_2.output)
        dense_3.forward(activation_2.output)
        activation_3.forward(dense_3.output)
    
        loss = loss_function.Calculate_loss(activation_3.output, y)
        prediction = np.argmax(activation_3.output, axis=1)
        accuracy = np.mean(prediction == y)
	    #putting iteration as frame so we can know when the loss decreased
        iteration = frame
    
        if loss < lowest_loss:
            print(f"now set of weights found, iteration:{iteration} and loss is {loss} and accuracy is {accuracy}")
            best_dense1_weights = dense_1.weights.copy()
            best_dense1_biases = dense_1.biases.copy()
            best_dense2_weights = dense_2.weights.copy()
            best_dense2_biases = dense_2.biases.copy()
            best_dense3_weights = dense_3.weights.copy()
            best_dense3_biases = dense_3.biases.copy()
            lowest_loss = loss
            RealTime_accuracy = accuracy
        else:
            dense_1.weights = best_dense1_weights.copy()
            dense_1.biases = best_dense1_biases.copy()
            dense_2.weights = best_dense2_weights.copy()
            dense_2.biases = best_dense2_biases.copy()
            dense_3.weights = best_dense3_weights.copy()
            dense_3.biases = best_dense3_biases.copy()
        #appending the accuracy and loss in a list
        accuracy_history.append(RealTime_accuracy)
        loss_history.append(lowest_loss)
        ax1.clear()
        ax1.plot(range(len(accuracy_history)), accuracy_history, label='Accuracy', color='b', marker='o')       
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.clear()
        ax2.plot(range(len(loss_history)), loss_history, label='Loss', color='r', marker='s')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.legend()

ani = FuncAnimation(fig, update, frames=range(100000), repeat=False)
plt.show()
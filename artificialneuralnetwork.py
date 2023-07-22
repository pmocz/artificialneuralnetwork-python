import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from functools import partial

"""
Create Your Own Artificial Neural Network for Multi-class Classification (With Python)
Philip Mocz (2023), @PMocz

Create and train your own artificial neural network to classify images of galaxies from SDSS/the Galaxy Zoo project.

"""

def g(x):
	""" sigmoid function """
	return 1.0 / (1.0 + np.exp(-x))


def grad_g(x):
	""" gradient of sigmoid function """
	gx = g(x)
	return gx * (1.0 - gx)	


def predict(Theta1, Theta2, X):
	""" Predict labels in a trained three layer classification network.
	Input:
	  Theta1       trained weights applied to 1st layer (hidden_layer_size x input_layer_size+1)
	  Theta2       trained weights applied to 2nd layer (num_labels x hidden_layer_size+1)
	  X            matrix of training data      (m x input_layer_size)
	Output:     
	  prediction   label prediction
	"""
	
	m = np.shape(X)[0]                    # number of training values
	num_labels = np.shape(Theta2)[0]
	
	a1 = np.hstack((np.ones((m,1)), X))   # add bias (input layer)
	a2 = g(a1 @ Theta1.T)                 # apply sigmoid: input layer --> hidden layer
	a2 = np.hstack((np.ones((m,1)), a2))  # add bias (hidden layer)
	a3 = g(a2 @ Theta2.T)                 # apply sigmoid: hidden layer --> output layer
	
	prediction = np.argmax(a3,1).reshape((m,1))
	return prediction


def reshape(theta, input_layer_size, hidden_layer_size, num_labels):
	""" reshape theta into Theta1 and Theta2, the weights of our neural network """
	ncut = hidden_layer_size * (input_layer_size+1)
	Theta1 = theta[0:ncut].reshape(hidden_layer_size, input_layer_size+1)
	Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size+1)
	return Theta1, Theta2
	
	
def cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
	""" Neural net cost function for a three layer classification network.
	Input:
	  theta               flattened vector of neural net model parameters
	  input_layer_size    size of input layer
	  hidden_layer_size   size of hidden layer
	  num_labels          number of labels
	  X                   matrix of training data
	  y                   vector of training labels
	  lmbda               regularization term
	Output:
	  J                   cost function
	"""
	
	# unflatten theta
	Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
	
	# number of training values
	m = len(y)
	
	# Feedforward: calculate the cost function J:
	
	a1 = np.hstack((np.ones((m,1)), X))   
	a2 = g(a1 @ Theta1.T)                 
	a2 = np.hstack((np.ones((m,1)), a2))  
	a3 = g(a2 @ Theta2.T)                 

	y_mtx = 1.*(y==0)
	for k in range(1,num_labels):
		y_mtx = np.hstack((y_mtx, 1.*(y==k)))

	# cost function
	J = np.sum( -y_mtx * np.log(a3) - (1.0-y_mtx) * np.log(1.0-a3) ) / m

	# add regularization
	J += lmbda/(2.*m) * (np.sum(Theta1[:,1:]**2)  + np.sum(Theta2[:,1:]**2))
	
	return J


def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
	""" Neural net cost function gradient for a three layer classification network.
	Input:
	  theta               flattened vector of neural net model parameters
	  input_layer_size    size of input layer
	  hidden_layer_size   size of hidden layer
	  num_labels          number of labels
	  X                   matrix of training data
	  y                   vector of training labels
	  lmbda               regularization term
	Output:
	  grad                flattened vector of derivatives of the neural network 
	"""
	
	# unflatten theta
	Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
	
	# number of training values
	m = len(y)
	
	# Backpropagation: calculate the gradients Theta1_grad and Theta2_grad:
	
	Delta1 = np.zeros((hidden_layer_size,input_layer_size+1))
	Delta2 = np.zeros((num_labels,hidden_layer_size+1))

	for t in range(m):
		
		# forward
		a1 = X[t,:].reshape((input_layer_size,1))
		a1 = np.vstack((1, a1))   #  +bias
		z2 = Theta1 @ a1
		a2 = g(z2)
		a2 = np.vstack((1, a2))   #  +bias
		a3 = g(Theta2 @ a2)
		
		# compute error for layer 3
		y_k = np.zeros((num_labels,1))
		y_k[y[t,0].astype(int)] = 1
		delta3 = a3 - y_k
		Delta2 += (delta3 @ a2.T)
		
		# compute error for layer 2
		delta2 = (Theta2[:,1:].T @ delta3) * grad_g(z2)
		Delta1 += (delta2 @ a1.T)	

	Theta1_grad = Delta1 / m
	Theta2_grad = Delta2 / m

	# add regularization
	Theta1_grad[:,1:] += (lmbda/m) * Theta1[:,1:]	
	Theta2_grad[:,1:] += (lmbda/m) * Theta2[:,1:]

	# flatten gradients
	grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

	return grad


N_iter = 1
J_min = np.Inf
theta_best = []
Js_train = np.array([])
Js_test = np.array([])

def callbackF(input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label, theta_k):
	""" Calculate some stats per iteration and update plot """
	global N_iter
	global J_min
	global theta_best
	global Js_train
	global Js_test
	# unflatten theta
	Theta1, Theta2 = reshape(theta_k, input_layer_size, hidden_layer_size, num_labels)
	# training data stats
	J = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
	y_pred = predict(Theta1, Theta2, X)
	accuracy = np.sum(1.*(y_pred==y))/len(y)
	Js_train = np.append(Js_train, J)
	# test data stats
	J_test = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
	test_pred = predict(Theta1, Theta2, test)
	accuracy_test = np.sum(1.*(test_pred==test_label))/len(test_label)
	Js_test= np.append(Js_test, J_test)
	# print stats
	print('iter={:3d}:  Jtrain= {:0.4f} acc= {:0.2f}%  |  Jtest= {:0.4f} acc= {:0.2f}%'.format(N_iter, J, 100*accuracy, J_test, 100*accuracy_test))
	N_iter += 1
	# Update theta_best
	if (J_test < J_min):
		theta_best = theta_k
		J_min = J_test
	# Update Plot
	iters = np.arange(len(Js_train))
	plt.clf()
	plt.subplot(2,1,1)
	im_size = 32
	pad = 4
	galaxies_image = np.zeros((3*im_size,6*im_size+2*pad), dtype=int) + 255
	for i in range(3):
		for j in range(6):
			idx = 3*j + i + 900*(j>1) + 900*(j>3) + (N_iter % 600) # +10
			shift = 0 + pad*(j>1) + pad*(j>3)
			ii = i * im_size
			jj = j * im_size + shift
			galaxies_image[ii:ii+im_size,jj:jj+im_size] = X[idx].reshape(im_size,im_size) * 255
			my_label = 'E' if y_pred[idx]==0 else 'S' if y_pred[idx]==1 else 'I'
			my_color = 'blue' if (y_pred[idx]==y[idx]) else 'red'
			plt.text(jj+2, ii+10, my_label, color=my_color)
			if (y_pred[idx]==y[idx]):
				plt.text(jj+4, ii+25, "✓", color='blue', fontsize=50)
	plt.imshow(galaxies_image, cmap='gray')
	plt.gca().axis('off')
	plt.subplot(2,1,2)
	plt.plot(iters, Js_test, 'r', label='test')
	plt.plot(iters, Js_train, 'b', label='train')
	plt.xlabel("iteration")
	plt.ylabel("cost")
	plt.xlim(0,600)
	plt.ylim(1,2.1)
	plt.gca().legend()
	plt.pause(0.001)


def main():
	""" Artificial Neural Network for classifying galaxies """
	
	# set the random number generator seed
	np.random.seed(917)
	
	# Load the training and test datasets
	train = np.genfromtxt('train.csv', delimiter=',')
	test = np.genfromtxt('test.csv', delimiter=',')
	
	# get labels (0=Elliptical, 1=Spiral, 2=Irregular)
	train_label = train[:,0].reshape(len(train),1)
	test_label = test[:,0].reshape(len(test),1)
	
	# normalize image data to [0,1]
	train = train[:,1:] / 255.
	test = test[:,1:] / 255.
	
	# Construct our data matrix X (2700 x 5000)
	X = train

    # Construct our label vector y (2700 x 1)
	y = train_label
	
	# Two layer Neural Network parameters:
	m = np.shape(X)[0]
	input_layer_size = np.shape(X)[1]
	hidden_layer_size = 8
	num_labels = 3
	lmbda = 1.0    # regularization parameter
	
	# Initialize random weights:
	Theta1 = np.random.rand(hidden_layer_size, input_layer_size+1) * 0.4 - 0.2
	Theta2 = np.random.rand(num_labels, hidden_layer_size+1) * 0.4 - 0.2
	
	# flattened initial guess
	theta0 = np.concatenate((Theta1.flatten(), Theta2.flatten()))
	J = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
	print('initial cost function J =', J)
	train_pred = predict(Theta1, Theta2, train)
	print('initial accuracy on training set =', np.sum(1.*(train_pred==train_label))/len(train_label))
	global Js_train
	global Js_test
	Js_train = np.array([J])
	J_test = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
	Js_test = np.array([J_test])

	# prep figure
	fig = plt.figure(figsize=(6,6), dpi=80)

	# Minimize the cost function using a nonlinear conjugate gradient algorithm
	args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)  # parameter values
	cbf = partial(callbackF, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label)
	theta = optimize.fmin_cg(cost_function, theta0, fprime=gradient, args=args, callback=cbf, maxiter=600)

	# unflatten theta
	Theta1, Theta2 = reshape(theta_best, input_layer_size, hidden_layer_size, num_labels)
	
	# Make predictions for the training and test sets
	train_pred = predict(Theta1, Theta2, train)
	test_pred = predict(Theta1, Theta2, test)
	
	# Print accuracy of predictions
	print('accuracy on training set =', np.sum(1.*(train_pred==train_label))/len(train_label))
	print('accuracy on test set =', np.sum(1.*(test_pred==test_label))/len(test_label))	
			
	# Save figure
	plt.savefig('artificialneuralnetwork.png',dpi=240)
	plt.show()
	    
	return 0



if __name__== "__main__":
  main()


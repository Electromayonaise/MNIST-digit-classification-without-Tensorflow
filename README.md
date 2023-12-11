<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

<div align="center">
  <img src="https://github.com/1999AZZAR/1999AZZAR/blob/main/resources/img/grid-snake.svg" alt="snake" />
</div>
<div id="user-content-toc">

# MNIST-digit-classification
Building a neural network from scratch that will allow MNIST digit classification, without using TensorFlow. The project that does use TensorFlow can be found [here](https://github.com/Electromayonaise/Handwritten-Digit-Recognition-with-TensorFlow).

## Process 

The training images have a dimension of 28x28, meaning they consist of 784 pixels. Each of said pixels has a pixel value that represents it's color between 0 (black) and 255 (White). The first thing to do is to consider the images as a matrix, a matrix that normally would have each row constitute an example (X) :

$$A = \begin{pmatrix}
-X^{[1]}- \\
-X^{[2]}-  \\
-X^{[3]}-  \\
-X^{[4]}-  \\ 
-X^{[n]}-
\end{pmatrix}$$   

However we will work with it's transposed where each column is an example, so each column has 784 rows corresponding to each pixel:

$$A^{T} = \begin{pmatrix}
| & | & | & | & | \\ 
X^{[1]}&  X^{[2]} & X^{[3]} & X^{[4]} & X^{[n]}\\
| & | & | & | & | \\ 
\end{pmatrix}$$   

Now, starting to descibe the nural network, it is quite a simple one, with only 2 layers as follows: 

- The 0th layer: 784 nodes (input layer)
- The 1st layer: 10 units (hidden layer)
- The 2nd layer: 10 units (output layer)

### 1. Forward propagation 

This process basically consists in taking an image, running it through the network and computing the output 

Input layer: $A^{[0]} = X$ where X is the matrix representing the examples with a dimension of 784 x n 

Then, it is necessary to multiply the input by the weight of each connection to the 1st layer and add the bias: 

$$Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}$$

Now we need to apply an activation function, otherwise each node would just be a linear combination of the nodes before it plus a bias. Meaning without an activation function a neural network is simply a linear regression. 

$A^{[1]} = g(Z^{[1]})$ Where the funtion $g$ is our activation function, in this case ReLU (Rectified linear unit), meaning $A^{[1]} = ReLU(Z^{[1]})$ 

To get from layer 1 to layer 2 first it is necessary to add the weights and bias 

$$ Z^{[2]} = W^{[2]} X^{[1]} + b^{[1]} $$ 

This last step is multiplying the 1st layer by the weight of each connection to the 2nd layer and adding the bias. 

Now we need to choose and apply the second activation function, in this case Softmax 

$A^{[2]} = Softmax(Z^{[2]})$ This will give a probability (a double between 0 and 1) which is the probability the prediction is correct 

### 2. Backwards propagation 

Backwards propagation serves as a way to adjust the weights and biases and train the network. It is the opposite process to forward prop., so it starts with a prediction and finds how much the prediction deviated from the actual label (so instead of giving a success probability it gives an epsilon or error) so it is possible to see how much did the previus weights and biases contributed to the actual error, and adjust them accordingly. 




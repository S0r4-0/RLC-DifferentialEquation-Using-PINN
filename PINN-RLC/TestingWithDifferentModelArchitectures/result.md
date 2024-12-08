## Final Inference

- **Number of Hidden Layers:**  Minimal impact on the model's performance indicates that the existing architecture is adequately complex.

- **Number of Nodes in each Hidden Layer:** Similar to the number of hidden layers, there is minimal impact on performance.

- **Activation Function**
  - Relu: Takes longer to predict with reduced accuracy, making it a less suitable choice.
  - Sigmoid: While it fails in the neural network (NN), it can be considered for the physics-informed neural network (PINN); however, Tanh is a better and faster choice.
  
- **Learning Rate**
  - In the NN, increasing the learning rate improves performance and vice versa.
  - However, in the PINN, increasing the learning rate significantly impacts performance negatively, while decreasing the learning rate shows almost no variation in performance.
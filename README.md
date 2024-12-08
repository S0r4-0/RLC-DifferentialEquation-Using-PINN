# 🧠 Solving RLC Differential Equations Using PINN

This repository demonstrates the use of **Physics-Informed Neural Networks (PINN)** to solve differential equations governing RLC (Resistor-Inductor-Capacitor) circuits. The goal is to show how neural networks can be employed to learn the solution of differential equations under the physics-constrained approach.

---

## 📜 Introduction

The **RLC circuit** is a fundamental physical system studied in physics and electrical engineering. The system's dynamics can be expressed by differential equations based on the RLC configuration. This repository leverages **Physics-Informed Neural Networks (PINNs)** to approximate the solution to the source-free RLC differential equation using machine learning techniques.

The aim of this project is to explore how PINNs can be trained with both supervised data loss and physics-based constraints to approximate solutions for the RLC circuit dynamics over time.

---

## 🧮 **Problem Overview**

The system is described by the **source-free underdamped series RLC circuit**, governed by the differential equation based on Kirchhoff's voltage law:

$$
L \frac{d^2 I}{dt^2} + R \frac{dI}{dt} + \frac{1}{C} I = 0,
$$

where L, R, and C are the inductance, resistance, and capacitance of the circuit, respectively. The system's initial conditions are given by:

$$
I(0) = I_0, \quad \frac{dI}{dt}(0) = 0,
$$

and the underdamped state corresponds to:

$$
\alpha < \omega_0, \quad \text{where } \alpha = \frac{R}{2L}, \omega_0 = \frac{1}{\sqrt{LC}}.
$$

The solution in the underdamped regime can be analytically described by:

$$
I(t) = I_0 \cdot e^{-\alpha t} \cdot \sin(\omega_d t + \phi), \quad \text{where } \omega_d = \sqrt{\omega_0^2 - \alpha^2}.
$$

---

## ✨ Features

- Implementation of a PINN architecture to solve the **source-free RLC circuit differential equation**.
- Comparison between NN and PINN in solving the differential equations.
- Testing effect of varying hidden layers and number of nodes in model architecture.
- Comparison between activation functions (ReLU, Sigmoid) for solving these equations.
- Testing of different learning rates and analyzing their effects on convergence and accuracy.
- Visualization of the convergence behavior using differential equation-based physics constraints.

---

## 🔧 Pre-requisites

Python 3.10 or above

## 🛠 Setup & Installation

Clone the repository and set up your environment to get started.

### 1. Clone the Repository

```bash
git clone https://github.com/S0r4-0/RLC-DifferentialEquation-Using-PINN.git
cd PINN-RLC
```
### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Results

The comparison reveals that while NNs are faster and easier to train with large datasets, they lack generalization and physics-based reasoning. PINNs, on the other hand, demonstrate improved generalization and align closely with the system's physics even when the labeled dataset is sparse, though at the cost of higher computational expense.

When solving differential equations related to physical systems such as RLC circuits, PINNs are advantageous due to their physics-based priors. This approach is especially valuable when data availability is limited or extrapolation to unseen inputs is critical.


## 🤝 Contributing

We welcome contributions! If you would like to improve the model, add new experiments, fix bugs, or enhance the documentation, feel free to fork the repository and open a Pull Request.

# 📜 License
This project is licensed under the [MIT License](LICENSE)

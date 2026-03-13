# Task I: Quantum Computing

This directory contains my implementations for Task I of the Q-MAML GSoC project. The two circuits here are designed to demonstrate core quantum computing primitives — state preparation, entanglement, single-qubit rotations, and quantum state comparison.

I chose PennyLane as the framework throughout because of its clean Python-native interface, automatic differentiation support and integration with machine learning libraries which will be essential in later tasks when training variational circuits with meta-learned parameters.

## Contents

| File | Description |
|------|-------------|
| `circuit1.py` | Five-qubit circuit with entanglement, SWAP, and rotation |
| `circuit2.py` | Four-qubit SWAP test comparing two two-qubit states |
| `requirements.txt` | Python dependencies |

## Circuit 1: Five-Qubit Entangled Circuit

**File:** `circuit1.py`

### Motivation

Before training any variational quantum model, it is important to understand how quantum information flows through a circuit — how superposition is created, how entanglement is built up across a register, and how individual qubit states are modified by single-qubit rotations. This circuit was designed to exercise all three of those mechanisms in a controlled, step-by-step way on a five-qubit register.

### Implementation

I applied the following gate sequence, working from physical reasoning at each step:

1. **Hadamard on all qubits** — I started by placing every qubit into an equal superposition $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$. This is the standard initialisation strategy for variational circuits and ensures no computational basis state is initially preferred, giving the optimiser maximum freedom.

2. **CNOT chain on pairs $(0,1)$, $(1,2)$, $(2,3)$, $(3,4)$** — I used a linear entangling layer, which is a common ansatz pattern in variational quantum algorithms. The chain structure propagates correlations sequentially across the register, creating multi-qubit entanglement without requiring all-to-all connectivity. This mirrors hardware-efficient circuit designs relevant to near-term quantum devices at CERN.

3. **SWAP on qubits $0$ and $4$** — After entangling the register, I exchanged the boundary qubits to demonstrate that the SWAP operation works correctly on an already-entangled state, where the notion of "swapping" is non-trivial due to the existing correlations across the register.

4. **RX($\pi/2$) on qubit $2$** — I applied a half-turn rotation about the X-axis to the central qubit, introducing a relative phase of $e^{-i\pi/4}$ between its $|0\rangle$ and $|1\rangle$ components. Single-qubit rotations like this are the primary tunable parameters in variational quantum circuits, so including one here grounds the circuit in the same gate vocabulary used in Q-MAML training.

### Outputs
- Terminal: first 8 statevector amplitudes and ASCII wire diagram
- `circuit1.png`: matplotlib circuit diagram

## Circuit 2: SWAP Test

**File:** `circuit2.py`

### Motivation

A key challenge in quantum machine learning is evaluating how similar two quantum states are. Unlike classical vectors, quantum states cannot trivially be compared — measuring them directly collapses the superposition. The SWAP test solves this using an ancilla qubit and interference, allowing the squared inner product $|\langle\psi|\phi\rangle|^2$ to be estimated from a single ancilla measurement probability without directly accessing the full statevectors.

I chose this as the second circuit because state similarity is a fundamental building block for both loss functions and model evaluation in variational quantum algorithms, and it directly relates to the kind of quantum kernel methods explored in quantum HEP analysis.

### State Preparation

I prepared two distinct two-qubit states across four data qubits:

| Register | Qubits | Preparation | Resulting State |
|----------|--------|-------------|-----------------|
| $|\psi\rangle$ | 0, 1 | $H \otimes R_X(\pi/3)$ | $(|0\rangle + |1\rangle)/\sqrt{2} \otimes (\cos(\pi/6)|0\rangle - i\sin(\pi/6)|1\rangle)$ |
| $|\phi\rangle$ | 2, 3 | $H \otimes H$ | $(|0\rangle + |1\rangle)/\sqrt{2} \otimes (|0\rangle + |1\rangle)/\sqrt{2}$ |
| Ancilla | 4 | $|0\rangle$ | — |

I deliberately made `q1` and `q3` differ — one prepared with $R_X(\pi/3)$ and the other with $H$ — so the two states are neither identical nor orthogonal. This gives a non-trivial overlap that can be verified analytically, making it a meaningful test of the SWAP test protocol.

### Protocol

The SWAP test follows a three-step interference scheme:

$$H_{\text{anc}} \;\rightarrow\; \text{CSWAP}(4,0,2) \;\rightarrow\; \text{CSWAP}(4,1,3) \;\rightarrow\; H_{\text{anc}} \;\rightarrow\; \text{Measure ancilla}$$

The first Hadamard creates a superposition on the ancilla so it can coherently control both the swapped and unswapped branches simultaneously. The two CSWAP gates conditionally exchange matching qubit pairs between the registers. The final Hadamard converts the phase relationship between branches into a measurable probability difference on the ancilla.

The squared overlap is recovered via:

$$|\langle\psi|\phi\rangle|^2 = 2\,P(|0\rangle_{\text{anc}}) - 1$$

### Verification

To validate the simulation, I computed $|\langle\psi|\phi\rangle|^2$ analytically by evaluating the inner product in the computational basis and confirmed the simulated result matches the closed-form value exactly. This cross-check is important: it demonstrates that the circuit is correctly constructed and that PennyLane's statevector simulator behaves as expected before trusting it for more complex experiments.

### Outputs
- Terminal: ancilla probabilities, simulated and analytical overlap values
- `circuit2.png`: matplotlib circuit diagram
- `swap_test_result.png`: bar chart of ancilla measurement probabilities

## Running the Code

Activate the project virtual environment from the root folder, then run either script:

```bash
# From the repository root
.venv\Scripts\python.exe task1\circuit1.py
.venv\Scripts\python.exe task1\circuit2.py
```

All output images are written to the `task1/` directory alongside the scripts.

## Dependencies

```
pennylane  >= 0.36.0
numpy      >= 1.24.0
matplotlib >= 3.7.0
```

Install via:

```bash
.venv\Scripts\python.exe -m pip install -r task1\requirements.txt
```

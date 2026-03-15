# Task V: Quantum Graph Neural Network (QGNN)

This task explores the design and implementation of a Quantum Graph Neural Network (QGNN) circuit using Google Cirq. The QGNN leverages the graph structure of data, encoding node features and edge connections into a quantum circuit.

## QGNN Circuit Concept
- **Node Encoding:** Each node is represented by a qubit. Node features are encoded using rotation gates (e.g., RX, RY).
- **Edge Encoding:** Edges are represented by entangling gates (e.g., CNOT, CZ) between qubits corresponding to connected nodes.
- **Message Passing:** Quantum operations (e.g., controlled rotations) simulate message passing between nodes.
- **Readout:** Measurement of qubits provides node or graph-level predictions.

## Files
- qgnn_circuit.py: Implementation and circuit drawing
- README.md: Instructions and explanation

## Requirements
- Python 3.8+
- cirq
- matplotlib

## Usage
1. Run `python qgnn_circuit.py` to generate and visualize the QGNN circuit.

---

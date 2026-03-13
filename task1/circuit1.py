"""
Task I – Part 1: Five-Qubit Quantum Circuit
============================================
Implements the following sequence of operations:
    b) Hadamard gate on every qubit
    c) CNOT on pairs (0,1), (1,2), (2,3), (3,4)
    d) SWAP between qubits 0 and 4
    e) X-rotation by pi/2 on qubit 2
    f) Circuit diagram saved to circuit1.png
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Simulator backend — statevector simulation on the default PennyLane device
# ---------------------------------------------------------------------------
N_QUBITS = 5
dev = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# Circuit definition
# ---------------------------------------------------------------------------
@qml.qnode(dev)
def circuit1():
    # Place each qubit into an equal superposition of |0⟩ and |1⟩
    for q in range(N_QUBITS):
        qml.Hadamard(wires=q)

    # Entangle adjacent qubits through a linear CNOT chain, propagating
    # correlations from qubit 0 through to qubit 4
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])

    # Exchange the quantum states of the first and last qubits
    qml.SWAP(wires=[0, 4])

    # Apply a rotation of π/2 about the X-axis on qubit 2, introducing a
    # relative phase of e^{-iπ/4} between the |0⟩ and |1⟩ components
    qml.RX(np.pi / 2, wires=2)

    return qml.state()


# ---------------------------------------------------------------------------
# Entry point — executes the circuit and exports visualisations
# ---------------------------------------------------------------------------
def main():
    state = circuit1()

    print("=" * 60)
    print("Circuit 1 – Statevector (first 8 amplitudes shown)")
    print("=" * 60)
    for idx, amp in enumerate(state[:8]):
        print(f"  |{idx:05b}> : {amp:.6f}")
    print("  ...")

    # Render the circuit using matplotlib and persist it as a high-resolution PNG
    fig, ax = qml.draw_mpl(circuit1)()
    fig.suptitle("Circuit 1: 5-Qubit Quantum Circuit", fontsize=13)
    fig.tight_layout()
    fig.savefig("circuit1.png", dpi=150, bbox_inches="tight")
    print("\nCircuit diagram saved → circuit1.png")
    plt.close(fig)

    # Print the ASCII wire diagram for quick terminal inspection
    print("\nText circuit diagram:")
    print(qml.draw(circuit1)())


if __name__ == "__main__":
    main()

import cirq
import matplotlib.pyplot as plt

# Example graph: 3 nodes, edges [(0,1), (1,2)]
def create_qgnn_circuit(node_features, edges):
    qubits = [cirq.GridQubit(0, i) for i in range(len(node_features))]
    circuit = cirq.Circuit()
    # Encode node features
    for i, feature in enumerate(node_features):
        circuit.append(cirq.rx(feature)(qubits[i]))
    # Encode edges with entanglement
    for edge in edges:
        circuit.append(cirq.CNOT(qubits[edge[0]], qubits[edge[1]]))
    # Message passing (example: controlled RY)
    for edge in edges:
        circuit.append(cirq.ry(0.5).controlled_by(qubits[edge[0]])(qubits[edge[1]]))
    # Readout
    for q in qubits:
        circuit.append(cirq.measure(q))
    return circuit, qubits

if __name__ == "__main__":
    # Example: 3 nodes, features [0.2, 0.5, 0.8], edges [(0,1), (1,2)]
    node_features = [0.2, 0.5, 0.8]
    edges = [(0,1), (1,2)]
    circuit, qubits = create_qgnn_circuit(node_features, edges)
    print("QGNN Circuit:")
    print(circuit)
    # Draw circuit
    cirq.vis.plot_circuit(circuit)
    plt.title("Quantum Graph Neural Network Circuit")
    plt.show()

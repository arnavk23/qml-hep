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
        # Correct controlled RY usage
        circuit.append(cirq.ControlledGate(cirq.ry(0.5))(qubits[edge[0]], qubits[edge[1]]))
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
    # Draw circuit as wire diagram using Matplotlib
    import matplotlib.pyplot as plt
    from cirq.contrib.svg import circuit_to_svg
    svg = circuit_to_svg(circuit)
    with open("task5/qgnn_circuit.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    print("Circuit diagram saved as task5/qgnn_circuit.svg")

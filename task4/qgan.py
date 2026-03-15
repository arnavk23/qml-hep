import numpy as np
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Load data from NPZ file
def load_data(npz_path):
    data = np.load(npz_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    return X_train, y_train, X_test, y_test

# Create quantum circuit for generator
def create_generator_circuit(qubits, params):
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))
        circuit.append(cirq.ry(params[i + len(qubits)])(qubit))
    # Add entanglement for richer generation
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit

# Create quantum circuit for discriminator
def create_discriminator_circuit(qubits, params):
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))
        circuit.append(cirq.ry(params[i + len(qubits)])(qubit))
    # Add entanglement for richer discrimination
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit

# Build QGAN model
def build_qgan(num_qubits):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    gen_params = tf.keras.Input(shape=(2 * num_qubits,), dtype=tf.dtypes.float32)
    disc_params = tf.keras.Input(shape=(2 * num_qubits,), dtype=tf.dtypes.float32)

    gen_circuit = tfq.layers.PQC(create_generator_circuit(qubits, gen_params), gen_params)
    disc_circuit = tfq.layers.PQC(create_discriminator_circuit(qubits, disc_params), disc_params)

    return gen_circuit, disc_circuit, qubits

# Training loop (simplified)
    import matplotlib.pyplot as plt
    gen_circuit, disc_circuit, qubits = build_qgan(num_qubits)

    # Prepare quantum data encoding
    def encode_data(X):
        return [cirq.Circuit(cirq.rx(xi)(qubits[i]) for i, xi in enumerate(x)) for x in X]

    X_train_q = encode_data(X_train)
    X_test_q = encode_data(X_test)

    # Generator model
    generator = tf.keras.Sequential([
        tfq.layers.PQC(create_generator_circuit(qubits, tf.keras.Input(shape=(2 * num_qubits,))), tf.keras.Input(shape=(2 * num_qubits,)))
    ])

    # Discriminator model
    discriminator = tf.keras.Sequential([
        tfq.layers.PQC(create_discriminator_circuit(qubits, tf.keras.Input(shape=(2 * num_qubits,))), tf.keras.Input(shape=(2 * num_qubits,)))
        , tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Optimizers
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Losses
    bce = tf.keras.losses.BinaryCrossentropy()

    # Training loop (full batch)
    gen_params = np.random.uniform(0, np.pi, size=(len(X_train), 2 * num_qubits))
    disc_params = np.random.uniform(0, np.pi, size=(len(X_train), 2 * num_qubits))

    for epoch in range(epochs):
        # Generate fake samples
        fake_circuits = [create_generator_circuit(qubits, gen_params[i]) for i in range(len(X_train))]
        fake_labels = np.zeros(len(X_train))

        # Real samples
        real_circuits = X_train_q
        real_labels = np.ones(len(X_train))

        # Combine
        circuits = fake_circuits + real_circuits
        labels = np.concatenate([fake_labels, real_labels])

        # Discriminator training
        with tf.GradientTape() as tape:
            disc_preds = discriminator.predict(np.array([disc_params[i] for i in range(2 * len(X_train))]))
            disc_loss = bce(labels, disc_preds.squeeze())
        grads = tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # Generator training
        with tf.GradientTape() as tape:
            gen_preds = generator.predict(np.array([gen_params[i] for i in range(len(X_train))]))
            # Try to fool discriminator
            gen_loss = bce(np.ones(len(X_train)), gen_preds.squeeze())
        grads = tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: D Loss={disc_loss:.4f}, G Loss={gen_loss:.4f}")

    # Evaluate
    test_disc_params = np.random.uniform(0, np.pi, size=(len(X_test), 2 * num_qubits))
    test_preds = discriminator.predict(np.array([test_disc_params[i] for i in range(len(X_test))]))
    y_pred = (test_preds.squeeze() > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, test_preds.squeeze())
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test AUC: {auc:.3f}")

    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, test_preds.squeeze())
    plt.figure()
    plt.plot(fpr, tpr, label='QGAN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    print("\nTo tune hyperparameters, adjust num_qubits, epochs, batch_size, and learning_rate in train_qgan().")
    print("Try grid search or cross-validation for best results.")

if __name__ == "__main__":
    # Update the path to your NPZ file
    npz_path = "input/QIS_EXAM_200Events.npz"
    X_train, y_train, X_test, y_test = load_data(npz_path)
    train_qgan(X_train, y_train, X_test, y_test)

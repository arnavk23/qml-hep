import numpy as np
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# ====================== DATA LOADING ======================
def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    train_dict = data['training_input'].item()
    test_dict = data['test_input'].item()

    X_train = np.vstack([train_dict['0'], train_dict['1']]).astype(np.float32)
    y_train = np.hstack([np.zeros(len(train_dict['0'])), np.ones(len(train_dict['1']))])

    X_test = np.vstack([test_dict['0'], test_dict['1']]).astype(np.float32)
    y_test = np.hstack([np.zeros(len(test_dict['0'])), np.ones(len(test_dict['1']))])

    return X_train, y_train, X_test, y_test


def normalize_features(X):
    X = np.array(X, dtype=np.float32)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    return X * np.pi


# ====================== QUANTUM GENERATOR ======================
def build_generator_circuit(num_qubits=4, num_layers=3):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    n_params = num_layers * 2 * num_qubits
    param_symbols = sympy.symbols(f'g0:{n_params}')
    
    circuit = cirq.Circuit()
    idx = 0
    for layer in range(num_layers):
        for i, q in enumerate(qubits):
            circuit.append(cirq.rx(param_symbols[idx])(q))
            idx += 1
            circuit.append(cirq.ry(param_symbols[idx])(q))
            idx += 1
        for i in range(num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        if num_qubits > 1:
            circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
    
    readout_op = cirq.Z(qubits[-1])
    return circuit, param_symbols, readout_op


def build_discriminator(input_dim=5):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# ====================== MAIN TRAINING ======================
def train_qgan(X_train, y_train, X_test, y_test,
               num_qubits=4, num_layers=3, epochs=300, batch_size=16, lr=0.005):
    
    X_train_norm = normalize_features(X_train)
    X_test_norm = normalize_features(X_test)
    
    gen_circuit, _, readout_op = build_generator_circuit(num_qubits, num_layers)
    generator_layer = tfq.layers.PQC(gen_circuit, readout_op, 
                                     differentiator=tfq.differentiators.ParameterShift())
    
    discriminator = build_discriminator()
    
    gen_optimizer = tf.keras.optimizers.Adam(lr * 0.7)
    disc_optimizer = tf.keras.optimizers.Adam(lr)
    bce = tf.keras.losses.BinaryCrossentropy()
    
    d_losses = []
    g_losses = []
    
    print("Starting QGAN Training...\n")
    
    for epoch in range(epochs):
        idx = np.random.randint(0, len(X_train_norm), batch_size)
        real_samples = X_train_norm[idx]
        
        # Train Discriminator
        with tf.GradientTape() as tape:
            empty = tfq.convert_to_tensor([cirq.Circuit()] * batch_size)
            gen_expect = generator_layer(empty)
            noise = tf.random.normal((batch_size, 4), stddev=0.5)
            fake_samples = tf.concat([gen_expect, noise], axis=1)
            
            real_pred = discriminator(real_samples)
            fake_pred = discriminator(fake_samples)
            d_loss = 0.5 * (bce(tf.ones_like(real_pred), real_pred) + 
                           bce(tf.zeros_like(fake_pred), fake_pred))
        
        grads_d = tape.gradient(d_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))
        
        # Train Generator
        with tf.GradientTape() as tape:
            empty = tfq.convert_to_tensor([cirq.Circuit()] * batch_size)
            gen_expect = generator_layer(empty)
            noise = tf.random.normal((batch_size, 4), stddev=0.5)
            fake_samples = tf.concat([gen_expect, noise], axis=1)
            
            fake_pred = discriminator(fake_samples)
            g_loss = bce(tf.ones_like(fake_pred), fake_pred)
        
        grads_g = tape.gradient(g_loss, generator_layer.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, generator_layer.trainable_variables))
        
        d_losses.append(float(d_loss.numpy()))
        g_losses.append(float(g_loss.numpy()))
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | D Loss: {d_loss.numpy():.4f} | G Loss: {g_loss.numpy():.4f}")
    
    # Evaluation
    test_preds = discriminator(X_test_norm).numpy().squeeze()
    y_pred = (test_preds > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, test_preds)
    
    print("\n" + "="*70)
    print("TASK IV: QUANTUM GENERATIVE ADVERSARIAL NETWORK (QGAN)")
    print("="*70)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test AUC       : {auc:.4f}")
    print("="*70)
    
    # ====================== SAVE FIGURES ======================
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss', color='blue')
    plt.plot(g_losses, label='Generator Loss', color='red')
    plt.title('QGAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("qgan_training_losses.png", dpi=300, bbox_inches='tight')
    print("→ Saved: qgan_training_losses.png")
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, test_preds)
    plt.plot(fpr, tpr, label=f'QGAN (AUC = {auc:.3f})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Signal vs Background Separation')
    plt.legend()
    plt.grid(True)
    plt.savefig("qgan_roc_curve.png", dpi=300, bbox_inches='tight')
    print("→ Saved: qgan_roc_curve.png")
    
    plt.close('all')
    
    print("\nFigures have been saved in your local task4 folder.")
    print("Check: C:\\Users\\kapoo\\qml-hep\\task4")
    
    return acc, auc


if __name__ == "__main__":
    npz_path = "/app/input/QIS_EXAM_200Events.npz"
    
    X_train, y_train, X_test, y_test = load_data(npz_path)
    
    print(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples, {X_train.shape[1]} features\n")
    
    acc, auc = train_qgan(
        X_train, y_train, X_test, y_test,
        num_qubits=4,
        num_layers=3,
        epochs=300,
        batch_size=16,
        lr=0.005
    )
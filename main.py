import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Activation functions ---
def relu(x):
    assert isinstance(x, np.ndarray), "Input to ReLU must be a numpy array"
    return np.maximum(0, x)

def relu_derivative(x):
    assert isinstance(x, np.ndarray), "Input to ReLU derivative must be a numpy array"
    return (x > 0).astype(float)

def softmax(x):
    assert isinstance(x, np.ndarray), "Input to softmax must be a numpy array"
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return result

# --- Data augmentation ---
def augment_image(img, max_rotation=15, max_translation=5):
    """Apply random rotation and translation to a grayscale image (flattened)."""
    # img shape expected to be (32*32,)
    img = img.reshape(32, 32)
    
    # Rotation
    angle = np.random.uniform(-max_rotation, max_rotation)
    M_rot = cv2.getRotationMatrix2D((16,16), angle, 1)
    
    # Translation
    tx = np.random.uniform(-max_translation, max_translation)
    ty = np.random.uniform(-max_translation, max_translation)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply rotation
    rotated = cv2.warpAffine(img, M_rot, (32, 32), borderMode=cv2.BORDER_REPLICATE)
    # Apply translation
    translated = cv2.warpAffine(rotated, M_trans, (32, 32), borderMode=cv2.BORDER_REPLICATE)
    
    return translated.flatten()

# --- Neural Network class with Adam optimizer & L2 regularization ---
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, lambda_=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes)
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.lambda_ = lambda_  # L2 regularization coefficient
        
        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.weights = []
        self.biases = []
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            # Initialize Adam moment estimates
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))
        
        self.t = 0  # timestep for Adam
        
    def forward(self, X):
        assert isinstance(X, np.ndarray)
        self.activations = [X]
        self.z_values = []
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = relu(z)
            self.activations.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = softmax(z)
        self.activations.append(output)
        return output
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cross_entropy = -np.sum(y_true * np.log(y_pred)) / m
        # Add L2 regularization term
        l2_term = 0.0
        for w in self.weights:
            l2_term += np.sum(w ** 2)
        l2_term = (self.lambda_ / (2 * m)) * l2_term
        return cross_entropy + l2_term
    
    def compute_accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)
    
    def backward(self, X, y, outputs):
        m = X.shape[0]
        self.d_weights = [None] * len(self.weights)
        self.d_biases = [None] * len(self.biases)
        
        # Output layer gradient
        dZ = outputs - y  # shape (m, num_classes)
        self.d_weights[-1] = (self.activations[-2].T @ dZ) / m + (self.lambda_ / m) * self.weights[-1]
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Hidden layers backpropagation
        for i in range(len(self.weights)-2, -1, -1):
            dA = dZ @ self.weights[i+1].T
            dZ = dA * relu_derivative(self.z_values[i])
            self.d_weights[i] = (self.activations[i].T @ dZ) / m + (self.lambda_ / m) * self.weights[i]
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Adam optimizer parameter update
        self.t += 1
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * self.d_weights[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * self.d_biases[i]
            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (self.d_weights[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (self.d_biases[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
    
    def train(self, X, y, X_val, y_val, epochs, batch_size, augment=False):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                if augment:
                    # Augment data on the fly
                    X_batch = np.array([augment_image(img) for img in X_batch])
                
                outputs = self.forward(X_batch)
                batch_loss = self.compute_loss(y_batch, outputs)
                epoch_loss += batch_loss * X_batch.shape[0]
                
                self.backward(X_batch, y_batch, outputs)
            
            train_loss = epoch_loss / X.shape[0]
            train_pred = self.forward(X)
            train_acc = self.compute_accuracy(y, train_pred)
            
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_acc = self.compute_accuracy(y_val, val_pred)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            if epoch % 10 == 0 or epoch == epochs-1:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def predict(self, X):
        outputs = self.forward(X)
        return np.argmax(outputs, axis=1)

# --- Data loading and preprocessing ---
def load_and_preprocess_data(data_dir):
    # Load labels CSV or build dataframe
    labels_csv_path = os.path.join(data_dir, 'amhcd-data-64/labels-map.csv')
    if os.path.exists(labels_csv_path):
        labels_df = pd.read_csv(labels_csv_path)
        assert 'image_path' in labels_df.columns and 'label' in labels_df.columns
    else:
        # Build dataframe by scanning folders
        image_paths = []
        labels = []
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    image_paths.append(os.path.join(label_dir, img_name))
                    labels.append(label_dir)
        labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    
    print(f"Loaded {len(labels_df)} samples with {labels_df['label'].nunique()} classes.")
    
    # Label encode
    label_encoder = LabelEncoder()
    labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
    num_classes = len(label_encoder.classes_)
    
    # Image loader and preprocessor
    def load_and_preprocess_image(path, target_size=(32, 32)):
        img = cv2.imread(os.path.join(data_dir, path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {os.path.join(data_dir, path)}")
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        return img.flatten()
    
    X = np.array([load_and_preprocess_image(p) for p in labels_df['image_path']])
    y = labels_df['label_encoded'].values
    
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 32*32
    
    # Split into train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    one_hot = OneHotEncoder(sparse_output=False)
    y_train_oh = one_hot.fit_transform(y_train.reshape(-1,1))
    y_val_oh = one_hot.transform(y_val.reshape(-1,1))
    y_test_oh = one_hot.transform(y_test.reshape(-1,1))
    
    return X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, num_classes, label_encoder

# --- K-Fold cross-validation ---
def cross_validate(X, y_one_hot, num_classes, label_encoder, k=5, epochs=50, batch_size=32, lambda_=0.001, augment=True):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    val_accuracies = []
    fold = 0
    
    for train_idx, val_idx in skf.split(X, np.argmax(y_one_hot, axis=1)):
        fold += 1
        print(f"\n--- Fold {fold} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]
        
        layer_sizes = [X_train.shape[1], 64, 32, num_classes]
        nn = MultiClassNeuralNetwork(layer_sizes, learning_rate=0.001, lambda_=lambda_)
        
        train_losses, val_losses, train_accs, val_accs = nn.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, augment=augment
        )
        
        val_accuracies.append(val_accs[-1])
        print(f"Fold {fold} validation accuracy: {val_accs[-1]:.4f}")
    
    print(f"\nAverage validation accuracy over {k} folds: {np.mean(val_accuracies):.4f}")
    return val_accuracies

# --- Main ---
if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, label_encoder = load_and_preprocess_data(data_dir)
    
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Combine train + val for cross-validation
    X_full = np.vstack([X_train, X_val])
    y_full = np.vstack([y_train, y_val])
    
    # K-fold cross-validation
    val_accuracies = cross_validate(X_full, y_full, num_classes, label_encoder, k=5, epochs=50, batch_size=32, lambda_=0.001, augment=True)
    
    # After CV, train final model on full train+val set
    print("\nTraining final model on full train+val set...")
    layer_sizes = [X_full.shape[1], 64, 32, num_classes]
    final_nn = MultiClassNeuralNetwork(layer_sizes, learning_rate=0.001, lambda_=0.001)
    
    final_nn.train(X_full, y_full, X_test, y_test, epochs=100, batch_size=32, augment=True)
    
    # Final evaluation on test set
    y_pred_test = final_nn.predict(X_test)
    print("\nClassification Report (Test set):")
    print(classification_report(np.argmax(y_test, axis=1), y_pred_test, target_names=label_encoder.classes_))
    
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix (Test set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot learning curves are not stored here since training prints only — but could be added
    
    print("Training and evaluation complete.")


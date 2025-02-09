import streamlit as st
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle

# -----------------------------
# Agent Function: Cognitive Load Feedback
# -----------------------------
def cognitive_load_feedback(predicted_state):
    feedback_messages = {
        # "stressed": "High cognitive load detected. Consider taking a short break or reducing your workload.",
        # "focused": "Your cognitive load is optimal. Keep up the good work.",
        "Relaxing": "You’re floating, weightless, with no immediate tasks. Your brain enters a calm state like when you’re on Earth with your eyes closed and at peace. This baseline state is crucial—it keeps your mind sharp for the challenges ahead. Mental clarity is just as important as physical readiness in space.✨🌙",
        "Real Left": "Time to move. As you flex your left hand, your right motor cortex activates, showing event-related desynchronization (ERD) in the alpha band. Every movement matters up here. Whether gripping a tool or adjusting equipment, these small actions keep your motor functions sharp in microgravity. 👈🤚",
        "Real Right": "Now, focus on your right hand. Your left hemisphere takes control, fine-tuning precision and coordination. Just as ERD patterns emerge, your brain reinforces its ability to operate in this strange environment. In space, even simple gestures help maintain dexterity and neural balance.👉✋",
        "Real Fists": "Engage both hands at once. Bilateral motor cortex activation kicks in as you clench and release your fists. Whether you’re handling delicate instruments or preparing for a high-intensity task, this coordination strengthens your ability to adapt. It’s also useful for rehabilitation—keeping your body and mind in sync.🤜🤛",
        "Real Feet": "Your feet may not be planted on solid ground, but they still play a crucial role. As you move them, your central motor cortex (near the Cz electrode) fires up. ERD patterns in alpha/beta waves emerge, keeping circulation steady and preventing stiffness. Flex, tap, and stretch—you’re keeping yourself mission-ready. 👞🏃‍♂",
        "Stressed": "Your brain shifts into high alert, releasing cortisol. In space, controlled stress sharpens focus, but too much impairs decisions. Deep breaths and mindfulness help you stay sharp.🧠⚡"
    }
    return feedback_messages.get(predicted_state, "Cognitive state unclear. Please verify your input.")

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    if df.isnull().values.any():
        st.warning("Warning: Data contains missing values. Please check your CSV.")
    
    if 'Subject' in df.columns:
        df = df.drop(columns=['Subject'])
    
    df.columns = [col.replace(" ", "").lower() for col in df.columns]
    target_col = 'classes'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    y = y.astype(str)
    
    num_cols = ['distributionshape', 'mean', 'signalstrength', 'rms']
    cat_cols = [col for col in feature_cols if col not in num_cols]
    
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    if cat_cols:
        X_cat = pd.get_dummies(X[cat_cols], prefix=cat_cols)
        X = pd.concat([X[num_cols], X_cat], axis=1)
    else:
        X = X[num_cols]
    
    X = X.astype(np.float32)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X.values, y_encoded, le

# -----------------------------
# Custom Dataset Definition
# -----------------------------
class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# -----------------------------
# Transformer-Based Model Definition
# -----------------------------
class LightweightEEGTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, num_layers=2, num_heads=4, dropout=0.1):
        super(LightweightEEGTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        logits = self.classifier(x)
        return logits

# -----------------------------
# Training and Evaluation Functions
# -----------------------------
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        st.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    st.write(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# -----------------------------
# Main Execution: Training, Evaluation, and Interactive Inference
# -----------------------------
# Streamlit page config
st.set_page_config(
    page_title="AI Agent in a Spaceship | Revolutionizing Space Health",
    page_icon="QE.jpg",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Page Title and Logo
st.image("QE.jpg", width=200)  # Adjust the width as needed
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Hello, I am ArmonIA</h2>", unsafe_allow_html=True)

def main():
    st.title("EEG Cognitive Load Prediction")
    
    # Set the CSV file path (update as needed)
    csv_file = r"C:\Users\HP\Downloads\HackathonAI/Cleaned_EEG_Dataset1.csv"
    
    # Train the model if not already saved.
    if not os.path.exists("eeg_transformer_model.pth") or not os.path.exists("label_encoder.pkl"):
        st.write("Training model...")
        X, y, label_encoder = load_and_preprocess_data(csv_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        input_dim = X.shape[1]
        num_classes = len(label_encoder.classes_)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = LightweightEEGTransformer(input_dim=input_dim, num_classes=num_classes)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        st.write("Starting training...")
        train_model(model, train_loader, criterion, optimizer, device, num_epochs=20)
        st.write("Evaluating model...")
        evaluate_model(model, test_loader, device)
        
        torch.save(model.state_dict(), "eeg_transformer_model.pth")
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
    else:
        st.write("Loading saved model and label encoder...")
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        input_dim = 9  # Update if necessary.
        num_classes = len(label_encoder.classes_)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LightweightEEGTransformer(input_dim=input_dim, num_classes=num_classes)
        model.load_state_dict(torch.load("eeg_transformer_model.pth", map_location=device))
        model.to(device)
    
    # -----------------------------
    # Interactive Inference Section
    # -----------------------------
    st.subheader("Interactive Inference")
    
    # Create a session state to store input values
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {
            'dist_shape': 0.0,
            'mean_val': 0.0,
            'signal_strength': 0.0,
            'rms_val': 0.0,
            'wave_type': 'Alpha'
        }
    
    # Input fields
    dist_shape = st.number_input("Enter Distribution Shape:", format="%.2f", value=st.session_state.inputs['dist_shape'])
    mean_val = st.number_input("Enter Mean:", format="%.2f", value=st.session_state.inputs['mean_val'])
    signal_strength = st.number_input("Enter Signal Strength:", format="%.2f", value=st.session_state.inputs['signal_strength'])
    rms_val = st.number_input("Enter RMS:", format="%.2f", value=st.session_state.inputs['rms_val'])
    
    categories = ["Alpha", "Beta", "Gamma", "Delta", "Theta"]
    wave_type = st.selectbox("Select Wave Type:", categories, index=categories.index(st.session_state.inputs['wave_type']))
    
    output_value = None  # Variable to store output


    if st.button("Predict"):
        one_hot = [1 if wave_type == cat else 0 for cat in categories]
        new_sample = np.concatenate([np.array([dist_shape, mean_val, signal_strength, rms_val], dtype=np.float32),
                                     np.array(one_hot, dtype=np.float32)], axis=0)
        
        if new_sample.shape[0] != input_dim:
            st.error(f"Input dimension mismatch. Expected {input_dim} features, got {new_sample.shape[0]}.")
        else:
            new_sample_tensor = torch.tensor(new_sample, dtype=torch.float32).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(new_sample_tensor.to(device))
                _, predicted_idx = torch.max(output, 1)
                predicted_class = label_encoder.inverse_transform(predicted_idx.cpu().numpy())
            
            output_value = predicted_class[0]  # Store the predicted class

            emoji_mapping = {
                "Relaxing": "🧘‍♂️",
                "Real Left": "🤚",
                "Real Right": "✋",
                "Real Fists": "✊",
                "Real Feet": "🚶‍♂",
                "Stressed": "😰"
            }
            emoji = emoji_mapping.get(output_value, "")
            st.success(f"Predicted class for the new sample: {output_value} {emoji}")


            feedback = cognitive_load_feedback(output_value)
            
            st.info(f"Agent Feedback: {feedback}")

    # Show reset button only if there is output
    if output_value:
        if st.button("Reset"):
            # Reset the input fields
            st.session_state.inputs['dist_shape'] = 0.0
            st.session_state.inputs['mean_val'] = 0.0
            st.session_state.inputs['signal_strength'] = 0.0
            st.session_state.inputs['rms_val'] = 0.0
            st.session_state.inputs['wave_type'] = 'Alpha'
            output_value = None  # Clear output
            st.experimental_rerun()  # Rerun the app to refresh the inputs

if __name__ == "__main__":
    main()

    
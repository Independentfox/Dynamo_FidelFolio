# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/kaggle/input/fidelfolio-dataset/FidelFolio_Dataset.csv')

other_cols_object = [f"Feature{i}" for i in [4, 5, 6, 7, 9]]
other_cols_object.append(" Target 1 ")
other_cols_object.append(" Target 2 ")
other_cols_object.append(" Target 3 ")

for col in other_cols_object:
    # Convert to string first, then remove commas, then convert to float
    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')


#--------------------------------------------------------------------#


# Sort and prepare data by time
df_sorted = df.sort_values(by=["Company", "Year"])

features=[f'Feature{i}' for i in range (1,29)]
num_cols=features

# Clean targets
targets = [' Target 1 ', ' Target 2 ', ' Target 3 ']
df[targets] = df[targets].apply(pd.to_numeric, errors='coerce')

# Fill NaNs with company-wise mean, then global mean as fallback
for target in targets:
    company_mean = df.groupby('Company')[target].transform(lambda x: x.fillna(x.mean()))
    global_mean = df[target].mean()
    df[target] = company_mean.fillna(global_mean)

# Fill NaNs for each feature by company-wise mean
for feature in features:
    feature_mean = df.groupby('Company')[feature].transform(lambda x: x.fillna(x.mean()))
    global_mean = df[feature].mean()
    df[feature] = feature_mean.fillna(global_mean)


#-------------------------------------------------------------------#
    

# Winsorization: cap values outside IQR bounds
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)

df[num_cols] = df[num_cols].apply(cap_outliers)


#-------------------------------------------------------------------#


from sklearn.preprocessing import StandardScaler

# Exclude target columns from scaling
target_cols = [' Target 1 ', ' Target 2 ', ' Target 3 ']
feature_cols = num_cols

# Scale features
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])


from sklearn.model_selection import train_test_split

X = df[feature_cols]
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


#-------------------------------------------------------------------#
                        # MLP Model 
#-------------------------------------------------------------------#

import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader  
import torch.optim as optim
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)
# Convert to tensors (on CPU)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoaders with batch_size = 64
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(input_size=X_train.shape[1], output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


import matplotlib.pyplot as plt

epochs = 1000
losses = []  # <-- Track losses here

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device) 
        yb = yb.to(device)  

        preds = model(xb)   # preds: (64, 3)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)  

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Plot loss vs epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.metrics import mean_squared_error
import numpy as np

# Set model to evaluation mode
model.eval()

# Make predictions on the test set
with torch.no_grad():
    y_pred_test = model(X_test_tensor.to(device)).cpu().numpy()
    y_true_test = y_test_tensor.cpu().numpy()

# Get the target column names
target_names = y_test.columns.tolist()


# Calculate RMSE for each target
for i, target in enumerate(target_names):
    rmse = mean_squared_error(y_true_test[:, i], y_pred_test[:, i], squared=False)
    print(f"Test RMSE for {target}: {rmse:.4f}")


import matplotlib.pyplot as plt

# Plot predicted vs actual for each target
for i, target in enumerate(target_names):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_test[:, i], y_pred_test[:, i], alpha=0.5)
    plt.plot([y_true_test[:, i].min(), y_true_test[:, i].max()],
             [y_true_test[:, i].min(), y_true_test[:, i].max()],
             'r--')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title(f'Predicted vs Actual - {target}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import shap

# Sample subset (for speed)
sample_idx = np.random.choice(len(X_test), size=200, replace=False)
X_shap = X_test.iloc[sample_idx]

# Convert to torch tensor
X_shap_tensor = torch.tensor(X_shap.values, dtype=torch.float32).to(device)

# Wrap model for SHAP
explainer = shap.DeepExplainer(model, torch.tensor(X_train.values[:1000], dtype=torch.float32).to(device))
shap_values = explainer.shap_values(X_shap_tensor)

# Plot SHAP summary for the first target
shap.summary_plot(shap_values[0], X_shap, feature_names=feature_cols, show=True)


from lime import lime_tabular
def predict_fn(input_np):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_np, dtype=torch.float32).to(device)
        outputs = model(input_tensor)
        return outputs.cpu().numpy()
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_cols,
    mode='regression',  # change to 'classification' if your task is classification
    verbose=True
)
idx = 0  # You can change this
instance = X_test.values[idx]
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    num_features=10  # number of top features to display
)
exp.show_in_notebook(show_table=True)
# or
exp.as_list()  # for a list of feature contributions


torch.save(model.state_dict(), 'fidelfolio_model_MLP.pkt')
print("Model saved as fidelfolio_model.pkt")

#-------------------------------------------------------------------#
                        # LSTM Model 
#-------------------------------------------------------------------#

# Sort and prepare data by time
df_sorted = df.sort_values(by=["Company", "Year"])

# Define input features and targets
features = [col for col in df.columns if 'Feature' in col]
targets = [' Target 1 ', ' Target 2 ', ' Target 3 ']

# Group by Company → Create sequences
sequences = []
target_seq = []


for company, group in df_sorted.groupby('Company'):
    
    group = group.reset_index(drop=True)
    if len(group) >= 3:
        X = group[features].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        y = group[targets].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        sequences.append(torch.tensor(X, dtype=torch.float32))
        target_seq.append(torch.tensor(y, dtype=torch.float32))


# Convert to padded sequences (same length)
from torch.nn.utils.rnn import pad_sequence
import torch

X_seq = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
y_seq = [torch.tensor(tgt, dtype=torch.float32) for tgt in target_seq]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
# Sort and prepare data by time

# Convert to padded sequences (same length)
from torch.nn.utils.rnn import pad_sequence
import torch

# 1. Sort and prepare data
features = [col for col in df.columns if 'Feature' in col]
targets = [' Target 1 ', ' Target 2 ', ' Target 3 ']
df_sorted = df.sort_values(by=["Company", "Year"])

sequences = []
target_seq = []

for company, group in df_sorted.groupby("Company"):
    group = group.reset_index(drop=True)
    if len(group) >= 3:
        X = group[features].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        y = group[targets].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        sequences.append(torch.tensor(X))
        target_seq.append(torch.tensor(y[-1]))  # only the last target

# 2. Custom Dataset
task_dataset = list(zip(sequences, target_seq))

class LSTMDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 3. Create DataLoader with list-based batching
def collate_fn(batch):
    X_list, y_list = zip(*batch)
    return list(X_list), list(y_list)

train_dataset = LSTMDataset(task_dataset)
data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# 4. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, packed_x):
        packed_out, (hn, cn) = self.lstm(packed_x)
        out = self.dropout(hn[-1])  # last hidden state from top layer
        return self.fc(out)

# 5. Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = LSTMModel(input_size=len(features), hidden_size=64, output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)


import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_sequence

epochs = 2000
losses = []

for epoch in range(epochs):
    model1.train()
    total_loss = 0
    for X_list, Y_list in data_loader:
        packed_x = pack_sequence([x.to(device) for x in X_list], enforce_sorted=False)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred = model1(packed_x)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation
model1.eval()
preds1 = []
actuals1 = []

with torch.no_grad():
    for X_list, Y_list in data_loader:
        packed_x = pack_sequence([x.to(device) for x in X_list], enforce_sorted=False)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred1 = model1(packed_x)
        preds1.append(pred1.cpu())

        actuals1.append(y_batch.cpu())

# Stack all batches into single arrays
preds1 = torch.cat(preds1).numpy()
actuals1 = torch.cat(actuals1).numpy()

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Plot predicted vs actual for each target
for i, target in enumerate(target_names):
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals1[:, i], preds1[:, i], alpha=0.5)
    plt.plot([actuals1[:, i].min(), actuals1[:, i].max()],
             [actuals1[:, i].min(), actuals1[:, i].max()],
             'r--')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title(f'Predicted vs Actual - {target}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate RMSE
for i, target in enumerate(target_names):
    rmse = mean_squared_error(actuals1[:, i], preds1[:, i], squared=False)
    print(f"Test RMSE for {target}: {rmse:.4f}")


# LIME in LSTM 
def flatten_last_k(seq_list, k=5):
    flat_X = []
    for seq in seq_list:
        seq_np = seq.numpy()
        if len(seq_np) >= k:
            trimmed = seq_np[-k:]  # last k timesteps
        else:
            # pad with zeros if shorter
            pad_len = k - len(seq_np)
            trimmed = np.concatenate([np.zeros((pad_len, seq_np.shape[1])), seq_np])
        flat_X.append(trimmed.flatten())
    return np.array(flat_X)

# Create flat dataset for LIME
X_flat = flatten_last_k(sequences, k=5)  # shape: [num_samples, k * num_features]
y_flat = np.stack(target_seq)


def reconstruct_sequence_from_flat(flat_array, k=5):
    """Convert flat vector back into a sequence tensor of shape [k, num_features]"""
    return torch.tensor(flat_array.reshape(k, -1), dtype=torch.float32)

def predict_fn_lime(input_np):
    model1.eval()
    input_seqs = []
    for flat_seq in input_np:
        seq_tensor = reconstruct_sequence_from_flat(flat_seq, k=5).to(device)
        input_seqs.append(seq_tensor)
    
    with torch.no_grad():
        packed_batch = pack_sequence(input_seqs, enforce_sorted=False)
        outputs = model1(packed_batch)
        return outputs.cpu().numpy()


from lime import lime_tabular

feature_names = [f"{f}_t{t}" for t in range(5) for f in features]

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_flat,
    feature_names=feature_names,
    mode='regression',
    verbose=True
)


idx = 0  # index of the company-sequence you want to explain
exp = explainer.explain_instance(
    data_row=X_flat[idx],
    predict_fn=predict_fn_lime,
    num_features=10  # top-k features
)

# Show explanation
exp.show_in_notebook(show_table=True)


torch.save(model.state_dict(), 'fidelfolio_model_LSTM.pkt')
print("Model saved as fidelfolio_model.pkt")


#-------------------------------------------------------------------#
                # LSTM with Attention Layer 
#-------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import numpy as np
import pandas as pd

# Define the LSTM model with Attention
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn_linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn_linear(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.dropout(context)
        output = self.fc(out)
        
        if return_attention:
            return output, attn_weights
        return output


# Create dataset and loader
class LSTMDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

task_dataset = list(zip(sequences, target_seq))
train_dataset = LSTMDataset(task_dataset)

def collate_fn(batch):
    X_list, y_list = zip(*batch)
    return list(X_list), list(y_list)

data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = LSTMAttentionModel(input_size=len(features), hidden_size=64, output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr=0.001)



losses = []  # Track average loss per epoch

# Train model
epochs = 3000
for epoch in range(epochs):
    model2.train()
    total_loss = 0
    for X_list, Y_list in data_loader:
        x_batch = torch.nn.utils.rnn.pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred = model2(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    losses.append(avg_loss)  # ✅ Append after epoch ends

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
# Evaluate
model2.eval()
preds2 = []
actuals2 = []

with torch.no_grad():
    for X_list, Y_list in data_loader:
        x_batch = torch.nn.utils.rnn.pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred2 = model2(x_batch)
        preds2.append(pred2.cpu())
        actuals2.append(y_batch.cpu())

preds2 = torch.cat(preds2).numpy()
actuals2 = torch.cat(actuals2).numpy()


plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Plot predicted vs actual for each target
for i, target in enumerate(target_names):
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals2[:, i], preds2[:, i], alpha=0.5)
    plt.plot([actuals2[:, i].min(), actuals2[:, i].max()],
             [actuals2[:, i].min(), actuals2[:, i].max()],
             'r--')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title(f'Predicted vs Actual - {target}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate RMSE
for i, target in enumerate(target_names):
    rmse = mean_squared_error(actuals2[:, i], preds2[:, i], squared=False)
    print(f"Test RMSE for {target}: {rmse:.4f}")


torch.save(model.state_dict(), 'fidelfolio_model_LSTMAttention.pkt')
print("Model saved as fidelfolio_model.pkt")


#-------------------------------------------------------------------#
                        # Transformer 
#-------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd


# Define the Transformer model for time-series forecasting
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, output_size, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))  # max seq_len = 500
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_linear(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        out = self.dropout(x[:, -1, :])  # last time step
        return self.fc_out(out)

# Prepare the data
features = [col for col in df.columns if 'Feature' in col]
targets = [' Target 1 ', ' Target 2 ', ' Target 3 ']
df_sorted = df.sort_values(by=["Company", "Year"])


# Create dataset and loader
class TimeSeriesDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

task_dataset = list(zip(sequences, target_seq))
train_dataset = TimeSeriesDataset(task_dataset)

def collate_fn(batch):
    X_list, y_list = zip(*batch)
    return list(X_list), list(y_list)

data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model3 = TransformerModel(
    input_size=len(features),
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    output_size=3
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model3.parameters(), lr=0.001)


losses = []  # Initialize a list to store epoch-wise loss
epochs = 1200

for epoch in range(epochs):
    model3.train()
    total_loss = 0
    for X_list, Y_list in data_loader:
        x_batch = pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred = model3(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    losses.append(avg_loss)  # Append average loss per epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluate
model3.eval()
preds3 = []
actuals3 = []

with torch.no_grad():
    for X_list, Y_list in data_loader:
        x_batch = pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred3 = model3(x_batch)
        preds3.append(pred3.cpu())
        actuals3.append(y_batch.cpu())

preds3 = torch.cat(preds3).numpy()
actuals3 = torch.cat(actuals3).numpy()


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Plot predicted vs actual for each target
for i, target in enumerate(target_names):
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals3[:, i], preds3[:, i], alpha=0.5)
    plt.plot([actuals3[:, i].min(), actuals3[:, i].max()],
             [actuals3[:, i].min(), actuals3[:, i].max()],
             'r--')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title(f'Predicted vs Actual - {target}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate RMSE
for i, target in enumerate(target_names):
    rmse = mean_squared_error(actuals3[:, i], preds3[:, i], squared=False)
    print(f"Test RMSE for {target}: {rmse:.4f}")


torch.save(model.state_dict(), 'fidelfolio_model_transformer.pkt')
print("Model saved as fidelfolio_model.pkt")



#-------------------------------------------------------------------#
                        # DeepTCN
#-------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --------------------- TCN Components ---------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size-1)*dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --------------------- Model Definition ---------------------
class DeepTCNModel(nn.Module):
    def __init__(self, input_size, num_channels, output_size):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        out = self.tcn(x)
        pooled = self.global_pool(out).squeeze(2)
        return self.fc(pooled)

# --------------------- Dataset and Loader ---------------------
class TCNDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def collate_pad_tcn(batch):
    X_list, y_list = zip(*batch)
    lengths = [x.shape[0] for x in X_list]
    max_len = max(lengths)
    padded = [F.pad(x, (0, 0, 0, max_len - len(x))) for x in X_list]
    return torch.stack(padded), torch.stack(y_list)

# --------------------- Training and Evaluation ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace with your actual sequence and target tensors
# sequences = [...]
# target_seq = [...]
features = [f'Feature{i+1}' for i in range(28)]
task_dataset = list(zip(sequences, target_seq))
train_dataset = TCNDataset(task_dataset)

data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_pad_tcn)

model4 = DeepTCNModel(input_size=len(features), num_channels=[64, 64, 64], output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --------------------- TCN Components ---------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size-1)*dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

# --------------------- Model Definition ---------------------
class DeepTCNModel(nn.Module):
    def __init__(self, input_size, num_channels, output_size):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        out = self.tcn(x)
        pooled = self.global_pool(out).squeeze(2)
        return self.fc(pooled)
    

# --------------------- Dataset and Loader ---------------------
class TCNDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def collate_pad_tcn(batch):
    X_list, y_list = zip(*batch)
    lengths = [x.shape[0] for x in X_list]
    max_len = max(lengths)
    padded = [F.pad(x, (0, 0, 0, max_len - len(x))) for x in X_list]
    return torch.stack(padded), torch.stack(y_list)


# --------------------- Training and Evaluation ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace with your actual sequence and target tensors
# sequences = [...]
# target_seq = [...]
features = [f'Feature{i+1}' for i in range(28)]
task_dataset = list(zip(sequences, target_seq))
train_dataset = TCNDataset(task_dataset)

data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_pad_tcn)

model4 = DeepTCNModel(input_size=len(features), num_channels=[64, 64, 64], output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


step_losses = []  # List to track loss at each batch (step)
epochs = 1000

for epoch in range(epochs):
    model4.train()
    total_loss = 0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model4(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step_losses.append(loss.item())  # Track step-wise (batch-wise) loss

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

# Evaluate
model4.eval()
preds4, actuals4 = [], []

with torch.no_grad():
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred4 = model4(x_batch)
        preds4.append(pred4.cpu())
        actuals4.append(y_batch.cpu())

preds4 = torch.cat(preds4).numpy()
actuals4 = torch.cat(actuals4).numpy()


for i in range(preds.shape[1]):
    rmse_i = np.sqrt(mean_squared_error(actuals4[:, i], preds4[:, i]))
    print(f"RMSE for target {i+1}: {rmse_i:.4f}")


import matplotlib.pyplot as plt
import numpy as np

# Assuming 'actuals' and 'preds' are NumPy arrays
for i in range(preds4.shape[1]):  # For each target variable
    plt.figure(figsize=(8, 6))
    plt.plot(actuals4[:, i], label="Actual", marker='o', linestyle='--', alpha=0.7)
    plt.plot(preds4[:, i], label="Predicted", marker='x', linestyle='-', alpha=0.7)
    plt.title(f"Actual vs Predicted for Target {i+1}")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()



for i in range(preds4.shape[1]):  # For each target variable
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals4[:, i], preds4[:, i], alpha=0.6)
    plt.title(f"Actual vs Predicted Scatter Plot for Target {i+1}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([actuals4[:, i].min(), actuals4[:, i].max()], 
             [actuals4[:, i].min(), actuals4[:, i].max()], 
             color='red', linestyle='--', label="Perfect Fit")
    plt.legend()
    plt.grid()
    plt.show()


#-------------------------------------------------------------------#
                        # NBeats Model
#-------------------------------------------------------------------#
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size, num_hidden_layers):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        
        # Fully connected stack
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_size, theta_size))
        
        self.fc_stack = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.fc_stack(x)
    

class NBeats(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_blocks=3, num_layers_per_block=4, theta_size=8):
        super(NBeats, self).__init__()
        self.input_size = input_size  # Number of features per timestep
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.theta_size = theta_size
        
        # Create stack of NBeats blocks
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, theta_size, hidden_size, num_layers_per_block)
            for _ in range(num_blocks)
        ])
        
        # Final projection layer
        self.projection = nn.Linear(theta_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize prediction
        prediction = torch.zeros(batch_size, self.output_size, device=x.device)
        
        # Process each timestep independently (simplified approach)
        # Alternatively, you could use an RNN/LSTM to process the sequence first
        x_means = x.mean(dim=1)  # Average across time dimension
        
        for block in self.blocks:
            # Pass through block
            theta = block(x_means)  # Use the averaged features
            
            # Project to output dimension
            block_prediction = self.projection(theta)
            
            # Add to overall prediction
            prediction = prediction + block_prediction
        
        return self.dropout(prediction)
    
# Dataset remains the same as your LSTM implementation
class LSTMDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# Assuming you have sequences and target_seq defined as in your original code
# sequences = [torch.randn(seq_len, num_features) for seq_len in [...]]
# target_seq = [torch.randn(output_size) for _ in sequences]
task_dataset = list(zip(sequences, target_seq))
train_dataset = LSTMDataset(task_dataset)

def collate_pad_tcn(batch):
    X_list, y_list = zip(*batch)
    lengths = [x.shape[0] for x in X_list]
    max_len = max(lengths)
    padded = [F.pad(x, (0, 0, 0, max_len - len(x))) for x in X_list]
    return torch.stack(padded), torch.stack(y_list)

data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_pad_tcn)


# Initialize N-BEATS model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model5 = NBeats(
    input_size=len(features),  # number of features per timestep
    output_size=3,             # same as your LSTM output
    hidden_size=64,
    num_blocks=3,
    num_layers_per_block=4,
    theta_size=8               # size of block output before final projection
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

step_losses = []  # List to track loss at each batch (step)
epochs = 1000

for epoch in range(epochs):
    model5.train()
    total_loss = 0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model5(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step_losses.append(loss.item())  # Track step-wise (batch-wise) loss

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

model5.eval()
preds5 = []
actuals5 = []

with torch.no_grad():
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred5 = model5(x_batch)
        preds5.append(pred5.cpu())
        actuals5.append(y_batch.cpu())

preds5 = torch.cat(preds5).numpy()
actuals5 = torch.cat(actuals5).numpy()


for i in range(preds.shape[1]):
    rmse_i = np.sqrt(mean_squared_error(actuals5[:, i], preds5[:, i]))
    print(f"RMSE for target {i+1}: {rmse_i:.4f}")


for i in range(preds.shape[1]):  # For each target variable
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals[:, i], preds[:, i], alpha=0.6)
    plt.title(f"Actual vs Predicted Scatter Plot for Target {i+1}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([actuals[:, i].min(), actuals[:, i].max()], 
             [actuals[:, i].min(), actuals[:, i].max()], 
             color='red', linestyle='--', label="Perfect Fit")
    plt.legend()
    plt.grid()
    plt.show()


#-------------------------------------------------------------------#
                        # DeepAR Model
#-------------------------------------------------------------------#
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error

class DeepAR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DeepAR, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        # Output layers
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_sigma = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get last timestep output
        last_out = lstm_out[:, -1, :]
        
        # Predict distribution parameters
        mu = self.fc_mu(last_out)
        sigma = F.softplus(self.fc_sigma(last_out)) + 1e-6  # Ensure positive
        
        return mu, sigma, hidden
    

# Dataset class remains the same
class LSTMDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Assuming sequences and target_seq are defined
task_dataset = list(zip(sequences, target_seq))
train_dataset = LSTMDataset(task_dataset)

def collate_fn(batch):
    X_list, y_list = zip(*batch)
    return list(X_list), list(y_list)

data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)


# Initialize DeepAR model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepAR(
    input_size=len(features),
    hidden_size=64,
    output_size=3,  # Same as your target size
    num_layers=2
).to(device)

criterion = nn.GaussianNLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_list, Y_list in data_loader:
        # Pad sequences
        x_batch = torch.nn.utils.rnn.pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])
        
        # Forward pass
        mu, sigma, _ = model(x_batch)
        
        # Calculate loss (negative log likelihood)
        loss = criterion(mu, y_batch, sigma**2)  # sigma^2 for variance
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")


# Calculate overall and per-target RMSE
def calculate_rmse(actuals, preds):
    """Calculate RMSE for each target variable separately"""
    # Convert to numpy if they're torch tensors
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    
    # Overall RMSE
    total_rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"\nOverall RMSE: {total_rmse:.4f}")
    
    # Per-target RMSE
    for i in range(preds.shape[1]):
        rmse_i = np.sqrt(mean_squared_error(actuals[:, i], preds[:, i]))
        print(f"RMSE for target {i+1}: {rmse_i:.4f}")
    
    return total_rmse

# Evaluation function for DeepAR
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for X_list, Y_list in data_loader:
            # Prepare batch
            x_batch = torch.nn.utils.rnn.pad_sequence([x.to(device) for x in X_list], batch_first=True)
            y_batch = torch.stack([y.to(device) for y in Y_list])
            
            # Get predictions (using mean of distribution)
            mu, _, _ = model(x_batch)
            
            all_preds.append(mu)
            all_actuals.append(y_batch)
    
    # Concatenate all predictions and actuals
    preds = torch.cat(all_preds, dim=0)
    actuals = torch.cat(all_actuals, dim=0)
    
    # Calculate RMSE metrics
    total_rmse = calculate_rmse(actuals, preds)
    
    return total_rmse, preds, actuals

# Run evaluation after training
test_rmse, test_preds, test_actuals = evaluate_model(model, data_loader, device)


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_and_visualize(model, data_loader, device):
    """Evaluate model and create visualizations for each target variable"""
    model.eval()
    all_preds = []
    all_actuals = []
    
    # Collect predictions
    with torch.no_grad():
        for X_list, Y_list in data_loader:
            x_batch = torch.nn.utils.rnn.pad_sequence([x.to(device) for x in X_list], batch_first=True)
            y_batch = torch.stack([y.to(device) for y in Y_list])
            
            mu, _, _ = model(x_batch)
            all_preds.append(mu.cpu())
            all_actuals.append(y_batch.cpu())
    
    # Concatenate results
    preds = torch.cat(all_preds, dim=0).numpy()
    actuals = torch.cat(all_actuals, dim=0).numpy()
    
    # Calculate and print RMSE for each target
    print("\nEvaluation Metrics:")
    for i in range(preds.shape[1]):
        rmse = np.sqrt(mean_squared_error(actuals[:, i], preds[:, i]))
        print(f"Target {i+1} RMSE: {rmse:.4f}")
        
        # Create scatter plot for this target
        plt.figure(figsize=(8, 6))
        plt.scatter(actuals[:, i], preds[:, i], alpha=0.6, 
                   label=f'Predictions (RMSE={rmse:.4f})')
        plt.title(f"Actual vs Predicted Values - Target {i+1}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")

    # Add perfect fit line
        min_val = min(actuals[:, i].min(), preds[:, i].min())
        max_val = max(actuals[:, i].max(), preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                 color='red', linestyle='--', label="Perfect Fit")
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return preds, actuals

# Usage example:
preds, actuals = evaluate_and_visualize(model, data_loader, device)


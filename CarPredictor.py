import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.api.types import is_numeric_dtype
import os

BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
SEED = 340
CHECKPOINT_PATH = 'best_car_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_worst_predictions(eval_model, loader, original_X_df, original_y, device, top_n=10):
    eval_model.eval()
    all_preds = []
    
    # Get all predictions from the model
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = eval_model(xb)
            all_preds.extend(preds.cpu().numpy())
            
    all_preds = np.array(all_preds)
    original_y = np.array(original_y)
    
    # getting the absolute error for every car
    errors = np.abs(all_preds - original_y)
    
    # get the indices of the highest errors
    worst_indices = np.argsort(errors)[::-1][:top_n]
    
    print(f"\n--- Top {top_n} Worst Predictions ---")
    for i in worst_indices:
        actual_price = original_y[i]
        predicted_price = all_preds[i]
        error_amount = errors[i]
        
        car_details = original_X_df.iloc[i]
        year = car_details.get('year', 'Unknown Year')
        manufacturer = car_details.get('manufacturer', 'Unknown Make')
        model = car_details.get('model', 'Unknown Model')
        
        print(f"Error: ${error_amount:,.0f} | Predicted: ${predicted_price:,.0f} | Actual: ${actual_price:,.0f}")
        print(f"Car: {year} {manufacturer} {model}\n")

def main():
    print("--- Setting up Data Pipeline ---")
    
    df = pd.read_csv("../data/vehicles.csv")
    df = df[(df['price'] > 2_000) & (df['price'] < 65_000)]
    limit = int(0.5 * len(df.columns))
    df = df.dropna(thresh=limit)
    df = df.dropna(subset=['manufacturer', 'model'])    
    
    POSSIBLE_EXCLUDES = ["id", "url", "VIN", "image_url", "description", "county", "region_url", "posting_date", "region"]
    y = df["price"].to_numpy()
    X_df = df.drop(columns=["price"])
    X_df = X_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in X_df.columns])
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_df, y, test_size=0.15, random_state=SEED)
    val_fraction = 0.15 / (1.0 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_fraction, random_state=SEED)
    
    numeric_features = [c for c in X_df.columns if is_numeric_dtype(X_df[c])]
    categorical_features = [c for c in X_df.columns if c not in numeric_features]
    high_cardinality_features = ['model']
    low_cardinality_features = [c for c in categorical_features if c != 'model']

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    target_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target", TargetEncoder(target_type="continuous")),
        ("scaler", StandardScaler()), # Good practice to scale the output of target encoding too!
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat_low", cat_pipe, low_cardinality_features),
            ("cat_high", target_pipe, high_cardinality_features),
        ],
        remainder="drop",
    )

    X_train_processed = pre.fit_transform(X_train, y_train)
    X_val_processed = pre.transform(X_val)
    X_test_processed = pre.transform(X_test)
    
    # converting to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train_processed, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_processed, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_processed, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dimension = X_train_processed.shape[1]
    print(f"Data ready! Input features: {input_dimension}")

    class CarPriceMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims=[128, 64, 32, 16], dropout_rate=0.2):
            super().__init__()
            layers = []
            current = input_dim
            
            for h in hidden_dims:
                # Linear Transformation
                layers.append(nn.Linear(current, h))
                
                # Batch Normalization
                layers.append(nn.BatchNorm1d(h))
                
                # Activation Function
                layers.append(nn.ReLU())
                
                # Dropout 
                if dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                    
                current = h
                
            # Final output layer
            layers.append(nn.Linear(current, 1)) 
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    # Initialize the model with a 20% dropout rate
    model = CarPriceMLP(input_dim=input_dimension, dropout_rate=0.2).to(device)
    
    # Using Mean Squared Error for Regression instead of CrossEntropy
    loss_fn = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    print("--- Starting Training Loop ---")

    def evaluate(eval_model, loader, loss_fn, device):
        eval_model.eval()
        losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = eval_model(xb)
                loss = loss_fn(preds, yb)
                losses.append(loss.item())
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())
                
        # Calculate Regression Metrics
        val_mse = np.mean(losses)
        val_mae = mean_absolute_error(all_targets, all_preds)
        val_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        return float(val_mse), float(val_mae), float(val_rmse)

    history_train_loss = []
    history_val_loss = []
    
    # Track lowest validation MSE for checkpointing
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_train_loss = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss.append(loss.item())
            
        train_loss = np.mean(running_train_loss)
        val_mse, val_mae, val_rmse = evaluate(model, val_loader, loss_fn, device)
        
        history_train_loss.append(train_loss)
        history_val_loss.append(val_mse)
        
        print_str = f"Epoch {epoch:02d} | Train MSE: {train_loss:.0f} | Val MSE: {val_mse:.0f} | Val MAE: ${val_mae:.0f} | Val RMSE: ${val_rmse:.0f}"
        
        # Checkpoint based on Validation MSE
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print_str += "  <-- Checkpoint Saved!"
            
        print(print_str)

    # Checkpoint Reloading Proof
    print("\n--- Verifying Checkpoint Reload ---")
    print(f"Expected Best Validation MSE: {best_val_loss:.0f}")
    
    reloaded_model = CarPriceMLP(input_dim=input_dimension).to(device)
    reloaded_model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    
    chkpt_val_mse, chkpt_val_mae, chkpt_val_rmse = evaluate(reloaded_model, val_loader, loss_fn, device)
    print(f"Reloaded Validation MSE:  {chkpt_val_mse:.0f}")
    print(f"Reloaded Validation MAE:  ${chkpt_val_mae:.0f}")

    print("\n--- Final Test ---")
    test_mse, test_mae, test_rmse = evaluate(reloaded_model, test_loader, loss_fn, device)
    print(f"Final Test MSE: {test_mse:.0f} | Final Test MAE: ${test_mae:.0f} | Final Test RMSE: ${test_rmse:.0f}")


    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), history_train_loss, label='Training MSE')
    plt.plot(range(1, NUM_EPOCHS + 1), history_val_loss, label='Validation MSE')
    plt.title('Car Price Predictor: Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig("LossCurve_Capstone.png")

    print_worst_predictions(reloaded_model, val_loader, X_val, y_val, device, top_n=15)
    
if __name__ == "__main__":
    main()
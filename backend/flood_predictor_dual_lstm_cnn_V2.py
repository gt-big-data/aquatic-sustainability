import os

import glob

import pickle

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import MinMaxScaler

import xarray as xr

import datetime as dt



# =====================================================

# CONFIGURATION

# =====================================================

DATA_DIR = "/storage/home/hcoda1/7/dchen640/scratch/saved_precip_data"

SOIL_DIR = "/storage/home/hcoda1/7/dchen640/scratch/SoilMoistureData"

CSV_PATH = "CompleteFloodDataset.csv"



SAVE_SCALER_PATH = "DualCNNLSTM_scaler_V2.pkl"

MODEL_SAVE_PATH = "DualCNNLSTM_best_model_V2.pt"



SEQ_LEN = 32

SOIL_SEQ_LEN = 14

SOIL_GRID_SIZE = 50

BATCH_SIZE = 4

EPOCHS = 1000

LR = 1e-4



# =====================================================

# DEVICE CHECKER

# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")



# =====================================================

# HELPER FUNCTIONS

# =====================================================

def getSoilMoistureGrid(lat, lon, grid_size, ds, variable_candidates=['sm']):

    if grid_size % 2 != 0:

        raise ValueError("grid_size must be even for CNN input.")

    lat_array = ds['lat'].values

    lon_array = ds['lon'].values

    half = grid_size // 2

    lat_idx = np.abs(lat_array - lat).argmin()

    lon_idx = np.abs(lon_array - lon).argmin()

    lat_start = max(lat_idx - half, 0)

    lat_end = lat_start + grid_size

    if lat_end > len(lat_array):

        lat_end = len(lat_array)

        lat_start = lat_end - grid_size

    lon_start = max(lon_idx - half, 0)

    lon_end = lon_start + grid_size

    if lon_end > len(lon_array):

        lon_end = len(lon_array)

        lon_start = lon_end - grid_size

    for var in variable_candidates:

        if var not in ds:

            continue

        try:

            grid = ds[var].isel(lat=slice(lat_start, lat_end),

                                lon=slice(lon_start, lon_end),

                                time=0)

            return grid

        except Exception as e:

            print(f"[GRID ERROR] Exception extracting grid from '{var}': {e}")

            return False

    print(f"[GRID ERROR] No valid soil moisture variable found. Tried: {variable_candidates}")

    return False



def load_soil_moisture(lat, lon, began_date, grid_size=SOIL_GRID_SIZE):

    start_date = began_date - pd.Timedelta(days=22)

    end_date = began_date - pd.Timedelta(days=8)

    date_list = pd.date_range(start_date, end_date, freq="D")

    grids = []



    for date in date_list:

        year = date.year

        pattern = os.path.join(SOIL_DIR, str(year), "**", f"*{date.strftime('%Y%m%d')}*.nc")

        files = glob.glob(pattern, recursive=True)

        if not files:

            continue

        try:

            ds = xr.open_dataset(files[0])

            grid = getSoilMoistureGrid(lat, lon, grid_size, ds)

            if grid is False:

                continue

            arr = grid.values

            if np.isnan(arr).all():

                continue

            mask = ~np.isnan(arr)

            arr = np.nan_to_num(arr, nan=0.0)

            stacked = np.stack([arr, mask.astype(float)], axis=0)

            grids.append(stacked)

        except Exception:

            continue



    if len(grids) < SOIL_SEQ_LEN:

        return None

    return grids[:SOIL_SEQ_LEN]



# =====================================================

# DATASET

# =====================================================

class FloodDataset(Dataset):

    def __init__(self, df, data_dir, scaler=None, fit_scaler=False, augment=False):

        self.df = df.reset_index(drop=True)

        self.data_dir = data_dir

        self.scaler = scaler

        self.fit_scaler = fit_scaler

        self.augment = augment

        self.samples = []



        for i, row in self.df.iterrows():

            event_id = str(row["ID"])

            country = str(row["Country"]).replace(" ", "_")

            lat = round(row["lat"], 2)

            lon = round(row["long"], 2)

            began = pd.to_datetime(row["Began"])

            began_str = str(began.date())

            year = str(began.year)



            event_folder = os.path.join(

                data_dir, year, f"event_{country}_lat:{lat:.2f}_lon:{lon:.2f}_{began_str}"

            )

            if not os.path.exists(event_folder):

                continue



            precip_grids = sorted(glob.glob(os.path.join(event_folder, "accum_*.npy")))

            if len(precip_grids) != SEQ_LEN:

                continue



            soil_grids = load_soil_moisture(lat, lon, began)

            if soil_grids is None:

                continue



            self.samples.append((precip_grids, soil_grids, row["isFlood"]))



        print(f"Total valid events with both precip & soil moisture: {len(self.samples)}")



        if fit_scaler:

            all_data = []

            for precip_grids, soil_grids, _ in self.samples:

                for g in precip_grids:

                    arr = np.load(g)

                    all_data.append(arr.flatten())

                for arr in soil_grids:

                    all_data.append(arr[0].flatten())

            all_data = np.concatenate(all_data).reshape(-1, 1)

            self.scaler = MinMaxScaler()

            self.scaler.fit(all_data)

            with open(SAVE_SCALER_PATH, "wb") as f:

                pickle.dump(self.scaler, f)

            print("Scaler fitted and saved (V2).")



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        precip_grids, soil_grids, isFlood = self.samples[idx]

        precip_seq, soil_seq = [], []



        for g in precip_grids:

            arr = np.load(g)

            arr = self.scaler.transform(arr.flatten().reshape(-1, 1)).reshape(arr.shape)

            if self.augment and np.random.rand() < 0.3:

                arr += np.random.normal(0, 0.02, arr.shape)

            precip_seq.append(arr[np.newaxis, ...])



        for arr in soil_grids:

            sm = self.scaler.transform(arr[0].flatten().reshape(-1, 1)).reshape(arr[0].shape)

            mask = arr[1]

            if self.augment and np.random.rand() < 0.3:

                sm += np.random.normal(0, 0.02, sm.shape)

            stacked = np.stack([sm, mask], axis=0)

            soil_seq.append(stacked)



        return (

            torch.tensor(np.stack(precip_seq), dtype=torch.float32),

            torch.tensor(np.stack(soil_seq), dtype=torch.float32),

            torch.tensor([isFlood], dtype=torch.float32)

        )



# =====================================================

# MODEL: Dual CNN + LSTM

# =====================================================

class DualCNNLSTM(nn.Module):

    def __init__(self):

        super(DualCNNLSTM, self).__init__()



        self.cnn_precip = nn.Sequential(

            nn.Conv2d(1, 8, 5, 2, 2),

            nn.BatchNorm2d(8),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.5),

            nn.Conv2d(8, 16, 3, 2, 1),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((8, 8)),

            nn.Dropout(0.5)

        )



        self.cnn_soil = nn.Sequential(

            nn.Conv2d(2, 8, 5, 2, 2),

            nn.BatchNorm2d(8),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.5),

            nn.Conv2d(8, 16, 3, 2, 1),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((8, 8)),

            nn.Dropout(0.5)

        )



        self.lstm = nn.LSTM(input_size=2*16*8*8, hidden_size=64, batch_first=True)

        self.dropout_lstm = nn.Dropout(0.5)

        self.fc_flood = nn.Linear(64, 1)



    def forward(self, precip, soil):

        B, Tp, C, H, W = precip.shape

        Ts = soil.shape[1]

        T = min(Tp, Ts)

        lstm_in = []



        for t in range(T):

            p = self.cnn_precip(precip[:, t])

            s = self.cnn_soil(soil[:, t])

            lstm_in.append(torch.cat([p.view(B, -1), s.view(B, -1)], dim=1))



        lstm_in = torch.stack(lstm_in, dim=1)

        lstm_out, _ = self.lstm(lstm_in)

        lstm_last = self.dropout_lstm(lstm_out[:, -1, :])

        flood_pred = torch.sigmoid(self.fc_flood(lstm_last))

        return flood_pred



# =====================================================

# DATA PREPARATION

# =====================================================

df = pd.read_csv(CSV_PATH, parse_dates=["Began"])



if not os.path.exists(SAVE_SCALER_PATH):

    dataset = FloodDataset(df, DATA_DIR, fit_scaler=True, augment=True)

    scaler = dataset.scaler

else:

    with open(SAVE_SCALER_PATH, "rb") as f:

        scaler = pickle.load(f)

    dataset = FloodDataset(df, DATA_DIR, scaler=scaler, fit_scaler=False, augment=True)



train_size = int(0.7 * len(dataset))

val_size = int(0.15 * len(dataset))

test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])



train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)



# =====================================================

# TRAINING LOOP

# =====================================================

model = DualCNNLSTM().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)

bce_loss = nn.BCELoss()



best_val_loss = float("inf")

patience = 20

epochs_no_improve = 0



for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0

    for precip, soil, flood_label in train_loader:

        precip, soil, flood_label = precip.to(device), soil.to(device), flood_label.to(device)

        optimizer.zero_grad()

        flood_pred = model(precip, soil)

        loss = bce_loss(flood_pred, flood_label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)



    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for precip, soil, flood_label in val_loader:

            precip, soil, flood_label = precip.to(device), soil.to(device), flood_label.to(device)

            flood_pred = model(precip, soil)

            total_val_loss += bce_loss(flood_pred, flood_label).item()

    avg_val_loss = total_val_loss / len(val_loader)



    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)



    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print("Validation loss improved, model saved (V2).")

        epochs_no_improve = 0

    else:

        epochs_no_improve += 1

        if epochs_no_improve >= patience:

            print(f"Early stopping at epoch {epoch+1}.")

            break



# =====================================================

# TESTING

# =====================================================

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

model.eval()



correct, total = 0, 0

with torch.no_grad():

    for precip, soil, flood_label in test_loader:

        precip, soil, flood_label = precip.to(device), soil.to(device), flood_label.to(device)

        flood_pred = model(precip, soil)

        predicted = (flood_pred > 0.5).float()

        correct += (predicted == flood_label).sum().item()

        total += flood_label.size(0)



flood_acc = correct / total if total > 0 else 0

print("\n=== TEST SET RESULTS (V2) ===")

print(f"Flood classification accuracy: {flood_acc*100:.2f}%")


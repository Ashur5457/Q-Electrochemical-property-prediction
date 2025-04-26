import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings
import argparse
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# 導入自定義的SMILES編碼器
from smiles_encoder import SMILESEncoder, SMILESEmbeddingModel

# 設置RDKit的日誌級別，減少不必要的警告
RDLogger.DisableLog('rdApp.*')

# 導入sklearn等模型相關庫（包含錯誤處理）
try:
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.multioutput import MultiOutputRegressor
    sklearn_available = True
except ImportError:
    SVR, KNeighborsRegressor, MultiOutputRegressor = None, None, None
    sklearn_available = False
    print("警告: scikit-learn未找到。SVR和KNeighborsRegressor基準模型將被跳過。")

try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    xgb = None
    xgboost_available = False
    print("警告: xgboost未找到。XGBoost基準模型將被跳過。")

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lgb = None
    lightgbm_available = False
    print("警告: lightgbm未找到。LightGBM基準模型將被跳過。")

warnings.filterwarnings('ignore')

# 設置設備
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {DEVICE}")

# --- 量子層 ---
class EnhancedQuantumLayer(nn.Module):
    """
    具有數據重新上傳和可訓練旋轉的改進量子層
    """
    def __init__(self, n_qubits=8, n_layers=3, input_dim=128):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 創建量子設備
        dev = qml.device("default.qubit", wires=n_qubits)
        
        # 定義量子電路
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights_rx, weights_ry, weights_rz, weights_cz, final_rotations):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights_rx[l, i], wires=i)
                    qml.RY(weights_ry[l, i], wires=i)
                    qml.RZ(weights_rz[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, (i + 1) % n_qubits])
                if l % 2 == 1:
                    for i in range(n_qubits):
                        qml.RY(inputs[i] * weights_cz[l, i], wires=i)
            for i in range(n_qubits):
                qml.RX(final_rotations[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # 定義權重形狀
        weight_shapes = {
            "weights_rx": (n_layers, n_qubits),
            "weights_ry": (n_layers, n_qubits),
            "weights_rz": (n_layers, n_qubits),
            "weights_cz": (n_layers, n_qubits),
            "final_rotations": n_qubits
        }
        
        # 創建TorchLayer
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # 輸入投影（將hidden_dim映射到n_qubits）
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, n_qubits),
            nn.LayerNorm(n_qubits)
        )
        
        # 輸出投影（將n_qubits映射回hidden_dim）
        self.output_proj = nn.Sequential(
            nn.Linear(n_qubits, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        # 應用輸入投影
        x_proj = self.input_proj(x)
        
        # 通過量子電路處理
        results = []
        for sample in x_proj:
            sample_cpu = sample.detach().cpu()
            q_result = self.qlayer(sample_cpu)
            results.append(q_result)
        
        # 堆疊結果並移回原始設備
        quantum_output = torch.stack(results).to(x.device)
        
        # 應用輸出投影
        return self.output_proj(quantum_output)

# --- 量子神經網絡迴歸模型（集成SMILES編碼） ---
class QuantumRegressionNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_qubits=8, n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 特徵提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 量子層
        self.quantum_layer = EnhancedQuantumLayer(
            n_qubits=n_qubits, n_layers=n_layers, input_dim=hidden_dim
        )
        
        # 輸出層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # 特徵提取
        features = self.feature_extractor(x)
        
        # 量子處理
        quantum_features = self.quantum_layer(features)
        
        # 結合跳躍連接
        combined_features = features + quantum_features
        
        # 輸出預測
        output = self.output_layer(combined_features)
        return output

# --- SMILES特定的量子迴歸模型 ---
class SMILESQuantumRegressionNet(nn.Module):
    """
    專為SMILES輸入設計的量子回歸模型
    """
    def __init__(self, smiles_encoder, output_dim=1, hidden_dim=256, n_qubits=8, n_layers=3, embed_dim=256):
        super().__init__()
        self.smiles_encoder = smiles_encoder  # SMILESEncoder實例
        self.output_dim = output_dim
        
        # SMILES特徵嵌入層
        encoder_input_dim = smiles_encoder.feature_size
        self.embedding_model = SMILESEmbeddingModel(
            input_dim=encoder_input_dim, 
            embedding_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        
        # 量子回歸網絡
        self.quantum_regression = QuantumRegressionNet(
            input_dim=embed_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_qubits=n_qubits,
            n_layers=n_layers
        )

    def forward(self, smiles_list):
        # 如果輸入已經是張量，則直接使用
        if isinstance(smiles_list, torch.Tensor):
            smiles_features = smiles_list
        else:
            # 將SMILES編碼為特徵張量
            smiles_features = self.smiles_encoder.encode_to_tensor(smiles_list).to(next(self.parameters()).device)
        
        # 嵌入分子特徵
        embedded_features = self.embedding_model(smiles_features)
        
        # 通過量子回歸網絡進行預測
        return self.quantum_regression(embedded_features)
    
    def encode_smiles(self, smiles_list):
        """額外的方法，僅編碼SMILES而不進行預測"""
        smiles_features = self.smiles_encoder.encode_to_tensor(smiles_list).to(next(self.parameters()).device)
        return self.embedding_model(smiles_features)

# --- 基準模型 ---
class MLPBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),  # 添加LayerNorm
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 確保d_model可被nhead整除
        if hidden_dim % nhead != 0:
            # 找到最接近且可被nhead整除的維度
             hidden_dim = (hidden_dim // nhead) * nhead
             if hidden_dim == 0: hidden_dim = nhead  # 確保hidden_dim至少為nhead
             print(f"調整Transformer hidden_dim為{hidden_dim}以能被nhead={nhead}整除")
             self.input_proj = nn.Linear(input_dim, hidden_dim)  # 重新初始化

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dim_feedforward=hidden_dim*2)  # 使用batch_first=True
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Transformer輸入形狀為 [batch_size, seq_len, features]
        x = self.input_proj(x)
        # 如果沒有序列維度則添加（假設每個樣本是長度為1的序列）
        if x.dim() == 2:
             x = x.unsqueeze(1)
        x = self.transformer(x)
        # 在輸出層之前刪除序列維度（選擇第一個/唯一的序列元素）
        if x.dim() == 3:
             x = x[:, 0, :]  # 選擇第一個token的輸出
        return self.output(x)

# --- 從Excel加載數據，現在包含SMILES處理 ---
def load_molecular_data_with_smiles(excel_path, sheet_name, targets, smiles_col='SMILES', use_smiles=True):
    """
    加載並預處理包含SMILES的分子數據
    返回張量用於特徵/目標 (Y_test是numpy格式)
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=1)
        print(f"成功加載Excel文件，包含{len(df)}行")
        print(f"發現列: {len(df.columns)}")
        print(f"數據類型: {df.dtypes.value_counts().to_dict()}")

        # 檢查SMILES列是否存在（如果使用SMILES）
        if use_smiles and smiles_col not in df.columns:
            raise ValueError(f"找不到SMILES列: {smiles_col}")
            
        # 檢查目標列是否存在
        missing_targets = [t for t in targets if t not in df.columns]
        if missing_targets:
            raise ValueError(f"找不到目標列: {missing_targets}")

        print(f"過濾前目標列中的NaN值: {df[targets].isna().sum().sum()}")
        df_clean = df.dropna(subset=targets)
        print(f"丟棄NaN目標後剩餘行數: {len(df_clean)}")

        # 初始化SMILES編碼器
        if use_smiles:
            smiles_list = df_clean[smiles_col].tolist()
            print(f"使用SMILES作為輸入，找到{len(smiles_list)}個SMILES字串")
            smiles_encoder = SMILESEncoder(encoding_method='combined')
            
            # 對SMILES進行編碼
            print("正在編碼SMILES字串...")
            X = smiles_encoder.encode_smiles_list(smiles_list)
            print(f"SMILES編碼特徵形狀: {X.shape}")
            
            # 儲存特徵名稱（用於參考）
            feature_columns = [f"morgan_{i}" for i in range(2048)] + smiles_encoder.descriptor_names
            
        else:
            # 使用數值特徵（原始方法）
            input_df = df_clean.select_dtypes(include=[float, int]).drop(columns=targets, errors='ignore')
            feature_columns = input_df.columns.tolist()

            if len(feature_columns) < 5:
                print("警告: 發現的數值特徵非常少.")
            print(f"輸入特徵 ({len(feature_columns)}): {feature_columns[:5]}...")

            # 特徵工程（簡化）
            if len(feature_columns) > 1:
                 try:
                     col1, col2 = feature_columns[0], feature_columns[1]
                     input_df[f'{col1}_x_{col2}'] = df_clean[col1] * df_clean[col2]
                     print("添加了一個交互項.")
                 except Exception as e:
                     print(f"警告: 無法創建交互項: {e}")

            # 處理缺失值
            if input_df.isna().any().any():
                print("用中位數填充缺失值.")
                for col in input_df.columns:
                    if input_df[col].isna().any():
                         median_val = input_df[col].median()
                         if pd.isna(median_val):
                              median_val = 0  # 如果整列都是NaN，則使用0
                         input_df[col] = input_df[col].fillna(median_val)

            X = input_df.values

        # 提取目標值
        Y = df_clean[targets].values

        # 檢查NaN或Inf
        x_nan_count = np.isnan(X).sum()
        y_nan_count = np.isnan(Y).sum()
        if x_nan_count > 0 or y_nan_count > 0:
            print(f"警告: 填充後在X中發現{x_nan_count}個NaN，在Y中發現{y_nan_count}個NaN")

        # 使用分位數裁剪處理異常值
        try:
            X = np.clip(X, np.nanquantile(X, 0.001), np.nanquantile(X, 0.999))
            Y = np.clip(Y, np.nanquantile(Y, 0.001), np.nanquantile(Y, 0.999))
        except Exception as e:
            print(f"警告: 無法應用分位數裁剪: {e}")

        # 處理剩餘的NaN和Inf值
        X = np.nan_to_num(X, nan=np.nan, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        Y = np.nan_to_num(Y, nan=np.nan, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        print(f"裁剪/nan_to_num後最終數據形狀: X: {X.shape}, Y: {Y.shape}")

        # 設置分層抽樣
        stratify = None
        if len(X) > 10:
            try:
                y_mean = np.mean(Y, axis=1)
                n_bins = min(5, len(X) // 5)
                if n_bins > 1:
                    y_strat = pd.qcut(y_mean, n_bins, labels=False, duplicates='drop')
                    # 檢查分層是否只產生一個唯一值
                    if len(set(y_strat)) > 1:
                         stratify = y_strat
                    else:
                         print("警告: 分層導致單一區間，禁用分層.")
            except Exception as e:
                print(f"警告: 分層失敗: {e}")

        # 分割數據
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=stratify
        )

        # 初始化縮放器
        scaler_x = RobustScaler()
        scaler_y = RobustScaler()

        # 適配並轉換訓練數據，轉換測試數據
        try:
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
        except ValueError as e:
            print(f"警告: RobustScaler失敗，錯誤: {e}。改用StandardScaler.")
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
        except Exception as e: # 捕捉其他潛在錯誤
             print(f"警告: 縮放器發生意外錯誤: {e}。改用StandardScaler.")
             scaler_x = StandardScaler()
             scaler_y = StandardScaler()
             X_train_scaled = scaler_x.fit_transform(X_train)
             Y_train_scaled = scaler_y.fit_transform(Y_train)
             X_test_scaled = scaler_x.transform(X_test)
        
        # 保持Y_test未縮放用於評估
        Y_test_numpy = Y_test.copy()

        # 縮放後檢查NaN/Inf
        for arr, name in [(X_train_scaled, "X_train_scaled"), (Y_train_scaled, "Y_train_scaled"),
                          (X_test_scaled, "X_test_scaled")]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"警告: 縮放後在{name}中發現NaN/Inf! 替換為零.")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # 轉換為張量
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        print("數據處理成功完成.")
        
        if use_smiles:
            # 如果使用SMILES，則還返回原始的SMILES列表和編碼器
            train_smiles = df_clean[smiles_col].iloc[np.where([x in X_train for x in X])[0]].tolist()
            test_smiles = df_clean[smiles_col].iloc[np.where([x in X_test for x in X])[0]].tolist()
            
            return (X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_numpy,
                    scaler_y, feature_columns, targets, train_smiles, test_smiles, smiles_encoder)
        else:
            # 返回張量用於DL和Y_test的numpy格式用於評估
            return (X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_numpy,
                    scaler_y, feature_columns, targets)

    except FileNotFoundError:
         print(f"錯誤: 在{excel_path}找不到Excel文件")
         raise  # 記錄後重新引發錯誤
    except Exception as e:
        print(f"數據加載錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise  # 記錄後重新引發錯誤

# --- 通用訓練函數（DL模型） ---
def train_model(
    model, model_name, # 添加model_name用於保存
    X_train, Y_train, X_val, Y_val,
    epochs=50, batch_size=32, lr=1e-3,
    save_dir_prefix="results", # 使用前綴作為保存目錄
    early_stop_patience=5,
    weight_decay=1e-4, # 添加weight_decay參數
    is_smiles_input=False, # 添加SMILES輸入標誌
    train_smiles=None # 訓練數據的SMILES列表（如果使用SMILES輸入）
):
    """
    DL模型的通用訓練函數。
    使用帶指定權重衰減的AdamW優化器。
    """
    save_dir = f"{save_dir_prefix}_{model_name}" # 創建特定保存目錄
    os.makedirs(save_dir, exist_ok=True)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"開始{model_name}訓練，共{epochs}個週期（lr={lr}, wd={weight_decay}, 保存到{save_dir}）...") # 記錄lr和wd
    
    # 根據輸入類型準備數據集和數據加載器
    if is_smiles_input and train_smiles is not None:
        # 訓練數據就是SMILES列表和目標值
        train_dataset = list(zip(train_smiles, Y_train.to(DEVICE)))
        
        # 創建驗證數據（從訓練數據抽取）
        val_size = min(int(len(train_smiles) * 0.1), 50)  # 最多使用50個樣本作為驗證集
        val_indices = torch.randperm(len(train_smiles))[:val_size].tolist()
        val_smiles = [train_smiles[i] for i in val_indices]
        val_targets = Y_train[val_indices].to(DEVICE)
        val_dataset = list(zip(val_smiles, val_targets))
        
        # 使用自定義批次收集函數處理SMILES數據
        def smiles_collate_fn(batch):
            smiles_list, targets = zip(*batch)
            return list(smiles_list), torch.stack(targets)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=smiles_collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=smiles_collate_fn
        )
    else:
        # 確保數據是在正確設備上的張量
        X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
        X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        with tqdm(train_loader, desc=f"週期 {epoch+1}/{epochs} [{model_name} 訓練]", leave=False) as tepoch:
            for i, batch in enumerate(tepoch):
                # 處理不同類型的輸入
                if is_smiles_input and train_smiles is not None:
                    smiles_batch, y_batch = batch
                    # 模型需要直接接受SMILES輸入
                    predictions = model(smiles_batch)
                else:
                    x_batch, y_batch = batch
                    
                    # 跳過包含NaN的批次
                    if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                        continue

                    # 前向傳播
                    predictions = model(x_batch)
                
                # 跳過產生NaN預測的批次
                if torch.isnan(predictions).any():
                    print(f"警告: 在{model_name}週期{epoch+1}，批次{i}中檢測到NaN預測。跳過。")
                    continue
                
                # 計算損失
                loss = loss_fn(predictions, y_batch)
                
                # 跳過NaN損失
                if torch.isnan(loss):
                    print(f"警告: 在{model_name}週期{epoch+1}，批次{i}中檢測到NaN損失。跳過。")
                    continue

                # 反向傳播和優化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        # 計算平均訓練損失
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else float('inf')
        train_losses.append(avg_train_loss)

        # 驗證
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
             with tqdm(val_loader, desc=f"週期 {epoch+1}/{epochs} [{model_name} 驗證]", leave=False) as vepoch:
                 for batch in vepoch:
                    # 處理不同類型的輸入
                    if is_smiles_input and train_smiles is not None:
                        smiles_batch, y_val_batch = batch
                        val_preds = model(smiles_batch)
                    else:
                        x_val_batch, y_val_batch = batch
                        val_preds = model(x_val_batch)
                    
                    val_loss = loss_fn(val_preds, y_val_batch)
                    if not torch.isnan(val_loss):
                         epoch_val_losses.append(val_loss.item())

        # 計算平均驗證損失
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses) if epoch_val_losses else float('inf')
        val_losses.append(avg_val_loss)
        
        # 更新學習率調度器
        scheduler.step()

        print(f"週期 {epoch+1}/{epochs} [{model_name}] - 訓練損失: {avg_train_loss:.6f}, 驗證損失: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}") # 記錄LR

        # 保存損失曲線
        try:
             plt.figure(figsize=(10, 5))
             plt.plot(train_losses, label='訓練損失')
             plt.plot(val_losses, label='驗證損失')
             plt.xlabel('週期')
             plt.ylabel('MSE損失')
             plt.title(f'{model_name}訓練損失')
             plt.legend()
             plt.grid(True)
             plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
             plt.close()
        except Exception as e:
             print(f"警告: 無法保存{model_name}的損失曲線圖: {e}")

        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            try:
                 torch.save({
                     'model_state_dict': best_model_state,
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(), # 保存調度器狀態
                     'epoch': epoch,
                     'loss': best_val_loss,
                 }, os.path.join(save_dir, 'best_model.pt'))
                 print(f"週期 {epoch+1} [{model_name}] - 保存最佳模型，驗證損失: {best_val_loss:.6f}")
            except Exception as e:
                 print(f"警告: 無法保存{model_name}的最佳模型檢查點: {e}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"{model_name}在{epoch+1}個週期後觸發早停。")
                if best_model_state:
                    try:
                         model.load_state_dict(best_model_state)
                    except Exception as e:
                         print(f"警告: 早停後無法加載{model_name}的最佳模型狀態: {e}")
                else:
                    print(f"警告: {model_name}觸發早停，但未保存最佳模型狀態。")
                break

    # 保存最終模型
    try:
         torch.save({
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict(),
             'epoch': epoch,
             'loss': avg_val_loss,
         }, os.path.join(save_dir, 'final_model.pt'))
    except Exception as e:
         print(f"警告: 無法保存{model_name}的最終模型檢查點: {e}")

    print(f"{model_name}訓練完成。")
    # 如果存在最佳模型狀態，則加載用於最終返回
    if best_model_state:
         try:
             model.load_state_dict(best_model_state)
             print(f"為最終評估加載{model_name}的最佳模型狀態。")
         except Exception as e:
              print(f"警告: 無法加載{model_name}的最終最佳模型狀態: {e}")
              
    return model # 只返回訓練好的模型對象

# --- 評估函數 ---
def evaluate_predictions(predictions_denorm, Y_test_denorm, target_names, model_name, save_dir): # 添加model_name和完整save_dir
    """
    評估預測結果並創建可視化。
    """
    os.makedirs(save_dir, exist_ok=True) # save_dir現在是完整路徑
    metrics = {target: {} for target in target_names}

    for i, target in enumerate(target_names):
        y_true = Y_test_denorm[:, i]
        y_pred = predictions_denorm[:, i] # 直接使用predictions_denorm

        # 排除NaN/Inf值
        valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        if not np.any(valid_idx):
            print(f"警告: {model_name}中的{target}沒有有效預測")
            metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
            continue

        y_true_valid = y_true[valid_idx]
        y_pred_valid = y_pred[valid_idx]

        if len(y_true_valid) < 2: # R2分數至少需要2個點
             print(f"警告: {model_name}中的{target}度量指標沒有足夠的有效點。")
             metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
             continue

        # 計算度量指標
        try:
             mse = mean_squared_error(y_true_valid, y_pred_valid)
             rmse = np.sqrt(mse)
             mae = mean_absolute_error(y_true_valid, y_pred_valid)
             r2 = r2_score(y_true_valid, y_pred_valid)

             metrics[target] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

             # 創建散點圖
             plt.figure(figsize=(8, 8))
             plt.scatter(y_true_valid, y_pred_valid, alpha=0.5)
             min_val = min(y_true_valid.min(), y_pred_valid.min())
             max_val = max(y_true_valid.max(), y_pred_valid.max())
             plt.plot([min_val, max_val], [min_val, max_val], 'r--')
             plt.xlabel(f"真實 {target}")
             plt.ylabel(f"預測 {target}")
             plt.title(f"{target} - {model_name} (R² = {r2:.4f})") # 在標題中使用model_name
             plt.grid(True)
             plt.savefig(os.path.join(save_dir, f"scatter_{target}.png"))
             plt.close()
        except Exception as e:
            print(f"計算/繪製{model_name}中的{target}度量指標時出錯: {e}")
            metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

    # 將度量指標保存為CSV
    metrics_df = pd.DataFrame(metrics).T # 轉置以提高可讀性
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"))
    print(f"--- {model_name}度量指標 --- ") # 識別模型度量指標
    print(metrics_df)

    # 保存預測結果（已經去標準化）
    pd.DataFrame(predictions_denorm, columns=target_names).to_csv(
        os.path.join(save_dir, "predictions.csv"), index=False
    )
    print(f"{model_name}的預測結果保存到{os.path.join(save_dir, 'predictions.csv')}")

    return metrics

# --- 主執行函數 ---
def main():
    print("=" * 80)
    print("使用SMILES輸入的量子迴歸模型")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="SMILES輸入的量子回歸模型")
    
    # 數據參數
    parser.add_argument("--excel-path", type=str, default="pnas.2214357120.sd01.xlsx", help="Excel數據文件的路徑")
    parser.add_argument("--sheet-name", type=str, default="Dataset", help="Excel文件中的工作表名稱")
    parser.add_argument("--smiles-col", type=str, default="SMILES", help="包含SMILES字串的列名")
    parser.add_argument("--target-col", type=str, default="LCE", help="目標列名（例如，電化學庫侖效率）")
    
    # 模型參數
    parser.add_argument("--hidden-dim", type=int, default=256, help="隱藏層的維度")
    parser.add_argument("--qnn-n-qubits", type=int, default=8, help="量子層中的量子比特數")
    parser.add_argument("--qnn-n-layers", type=int, default=3, help="量子層中的層數")
    parser.add_argument("--embed-dim", type=int, default=256, help="SMILES嵌入的維度")
    
    # 訓練參數
    parser.add_argument("--epochs", type=int, default=50, help="訓練週期數")
    parser.add_argument("--batch-size", type=int, default=32, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="學習率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="權重衰減")
    parser.add_argument("--patience", type=int, default=10, help="早停的耐心值")
    parser.add_argument("--debug", action="store_true", help="以調試模式運行（較小模型，更少週期）")
    parser.add_argument("--save-dir", type=str, default="smiles_qnn_results", help="保存結果的目錄")
    
    # 模型選擇
    parser.add_argument("--model", type=str, default="quantum", choices=["quantum", "mlp", "transformer"], 
                        help="選擇使用的模型類型")
                        
    args = parser.parse_args()
    
    # 調試模式處理
    if args.debug:
        print("以調試模式運行 - 使用較小的模型和較少的訓練週期")
        args.epochs = 5
        args.batch_size = 16
        args.hidden_dim = 128
        args.qnn_n_qubits = 4
        args.qnn_n_layers = 2
        args.embed_dim = 128
        
    # 嘗試加載數據
    try:
        print(f"\n加載數據從 {args.excel_path}, 工作表 '{args.sheet_name}'...")
        
        # 使用新的數據加載函數
        targets = [args.target_col]  # 只使用指定的目標列
        results = load_molecular_data_with_smiles(
            args.excel_path, 
            args.sheet_name, 
            targets, 
            smiles_col=args.smiles_col,
            use_smiles=True
        )
        
        # 解包結果
        (X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_numpy,
         scaler_y, feature_columns, target_names, train_smiles, test_smiles, smiles_encoder) = results
        
        input_dim = smiles_encoder.feature_size
        output_dim = len(target_names)
        
        print(f"\n加載了 {len(train_smiles)} 個訓練樣本和 {len(test_smiles)} 個測試樣本")
        print(f"SMILES特徵維度: {input_dim}")
        print(f"目標維度: {output_dim} ({', '.join(target_names)})")
        
        # 根據選擇創建模型
        if args.model == "quantum":
            print("\n初始化SMILES量子回歸模型...")
            model = SMILESQuantumRegressionNet(
                smiles_encoder=smiles_encoder,
                output_dim=output_dim,
                hidden_dim=args.hidden_dim,
                n_qubits=args.qnn_n_qubits,
                n_layers=args.qnn_n_layers,
                embed_dim=args.embed_dim
            )
            model_name = "SMILESQuantumReg"
        elif args.model == "mlp":
            print("\n初始化SMILES MLP基準模型...")
            # 創建SMILES編碼+MLP模型
            embedding_model = SMILESEmbeddingModel(
                input_dim=input_dim, 
                embedding_dim=args.embed_dim,
                hidden_dim=args.hidden_dim
            )
            mlp = MLPBaseline(
                input_dim=args.embed_dim,
                output_dim=output_dim,
                hidden_dim=args.hidden_dim
            )
            # 組合模型
            model = nn.Sequential(embedding_model, mlp)
            model_name = "SMILES_MLP"
        elif args.model == "transformer":
            print("\n初始化SMILES Transformer基準模型...")
            # 創建SMILES編碼+Transformer模型
            embedding_model = SMILESEmbeddingModel(
                input_dim=input_dim, 
                embedding_dim=args.embed_dim,
                hidden_dim=args.hidden_dim
            )
            transformer = TransformerBaseline(
                input_dim=args.embed_dim,
                output_dim=output_dim,
                hidden_dim=args.hidden_dim
            )
            # 組合模型
            model = nn.Sequential(embedding_model, transformer)
            model_name = "SMILES_Transformer"
        else:
            raise ValueError(f"不支援的模型類型: {args.model}")
            
        # 列印模型架構和參數
        print(f"\n{model_name}模型架構:")
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可訓練參數數量: {num_params:,}")
        
        # 訓練模型
        print(f"\n開始訓練{model_name}...")
        trained_model = train_model(
            model=model,
            model_name=model_name,
            X_train=X_train_tensor,  # 雖然我們直接使用SMILES，但保留這些參數以保持函數簽名一致
            Y_train=Y_train_tensor,
            X_val=X_train_tensor[:min(100, len(X_train_tensor))],  # 簡單驗證集
            Y_val=Y_train_tensor[:min(100, len(Y_train_tensor))],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir_prefix=args.save_dir,
            early_stop_patience=args.patience,
            weight_decay=args.weight_decay,
            is_smiles_input=True,
            train_smiles=train_smiles
        )
        
        # 評估模型
        print(f"\n評估{model_name}...")
        trained_model.eval()
        with torch.no_grad():
            predictions = trained_model(test_smiles).cpu().numpy()
            
        # 將預測轉換回原始刻度
        predictions_denorm = scaler_y.inverse_transform(predictions)
        
        # 計算和顯示度量指標
        metrics = evaluate_predictions(
            predictions_denorm=predictions_denorm,
            Y_test_denorm=Y_test_numpy,
            target_names=target_names,
            model_name=model_name,
            save_dir=f"{args.save_dir}_{model_name}"
        )
        
        # 列印最終結果
        print("\n最終結果:")
        for target in target_names:
            print(f"\n{target}:")
            for metric, value in metrics[target].items():
                print(f"  {metric}: {value:.4f}")
                
        print(f"\n{model_name}訓練和評估成功完成!")
        
    except Exception as e:
        print(f"\n執行期間發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

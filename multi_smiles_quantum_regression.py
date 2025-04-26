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

# 導入自定義的多SMILES編碼器
from multi_smiles_encoder import MultiSMILESEncoder, SMILESEmbeddingModel

# 設置RDKit的日誌級別，減少不必要的警告
RDLogger.DisableLog('rdApp.*')

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
        
        # 嘗試創建lightning.qubit設備，若失敗則回退到default.qubit
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            print(f"成功初始化 lightning.qubit 量子設備，使用 {n_qubits} 個量子比特")
        except Exception as e:
            print(f"初始化 lightning.qubit 失敗: {e}，回退到 default.qubit")
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

# --- 量子神經網絡迴歸模型（集成多SMILES編碼） ---
class QuantumRegressionNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_qubits=8, n_layers=3, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 特徵提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 量子層
        self.quantum_layer = EnhancedQuantumLayer(
            n_qubits=n_qubits, n_layers=n_layers, input_dim=hidden_dim
        )
        
        # 輸出層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
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

# --- 多SMILES特定的量子迴歸模型 ---
class MultiSMILESQuantumRegressionNet(nn.Module):
    """
    專為多SMILES輸入設計的量子回歸模型
    """
    def __init__(self, smiles_encoder, output_dim=1, hidden_dim=256, n_qubits=8, 
                 n_layers=3, embed_dim=256, dropout_rate=0.1):
        super().__init__()
        self.smiles_encoder = smiles_encoder  # MultiSMILESEncoder實例
        self.output_dim = output_dim
        
        # SMILES特徵嵌入層
        encoder_input_dim = smiles_encoder.feature_size
        self.embedding_model = SMILESEmbeddingModel(
            input_dim=encoder_input_dim, 
            embedding_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # 量子回歸網絡
        self.quantum_regression = QuantumRegressionNet(
            input_dim=embed_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )

    def forward(self, smiles_batch, concentrations_batch):
        """
        處理一批SMILES及其濃度數據
        
        參數:
            smiles_batch: 形狀為 (batch_size, max_smiles) 的SMILES列表列表
            concentrations_batch: 形狀為 (batch_size, max_smiles) 的濃度列表列表
        """
        # 將SMILES和濃度編碼為特徵張量
        encoded_features = self.smiles_encoder.encode_to_tensor(
            smiles_batch, concentrations_batch
        ).to(next(self.parameters()).device)
        
        # 嵌入分子特徵
        embedded_features = self.embedding_model(encoded_features)
        
        # 通過量子回歸網絡進行預測
        return self.quantum_regression(embedded_features)

# --- 基準模型 ---
class MLPBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, nhead=4, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 確保d_model可被nhead整除
        if hidden_dim % nhead != 0:
            hidden_dim = (hidden_dim // nhead) * nhead
            if hidden_dim == 0: hidden_dim = nhead  # 確保hidden_dim至少為nhead
            print(f"調整Transformer hidden_dim為{hidden_dim}以能被nhead={nhead}整除")
            self.input_proj = nn.Linear(input_dim, hidden_dim)  # 重新初始化

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            batch_first=True, 
            dim_feedforward=hidden_dim*2,
            dropout=dropout_rate
        )
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

# --- 從Excel加載數據，處理多個SMILES和濃度 ---
def load_data_with_multi_smiles(excel_path, target_col='LCE', max_smiles=6):
    """
    加載並預處理包含多個SMILES和濃度的數據
    
    參數:
        excel_path: Excel文件路徑
        target_col: 目標列名稱
        max_smiles: 最大SMILES數量
        
    返回:
        訓練和測試數據，以及對應的縮放器
    """
    try:
        # 讀取Excel文件
        df = pd.read_excel(excel_path)
        print(f"成功加載Excel文件，包含{len(df)}行")
        
        # 檢查所需列是否存在
        if target_col not in df.columns:
            raise ValueError(f"找不到目標列: {target_col}")
            
        # 檢查SMILES和濃度列
        smiles_cols = []
        conc_cols = []
        
        # 尋找SMILES和濃度列
        for i in range(1, max_smiles + 1):
            smiles_col = f"SMILES{i}"
            conc_col = f"Concentration{i}"
            
            if smiles_col in df.columns and conc_col in df.columns:
                smiles_cols.append(smiles_col)
                conc_cols.append(conc_col)
                
        if not smiles_cols or not conc_cols:
            # 嘗試其他可能的列名格式
            for col in df.columns:
                if 'smiles' in col.lower() and not any(col == sc for sc in smiles_cols):
                    smiles_cols.append(col)
                elif 'conc' in col.lower() and not any(col == cc for cc in conc_cols):
                    conc_cols.append(col)
            
        if not smiles_cols:
            raise ValueError("找不到SMILES列，請確保列名格式為'SMILES1', 'SMILES2'等")
        if not conc_cols:
            raise ValueError("找不到濃度列，請確保列名格式為'Concentration1', 'Concentration2'等")
            
        print(f"找到{len(smiles_cols)}個SMILES列: {smiles_cols}")
        print(f"找到{len(conc_cols)}個濃度列: {conc_cols}")
        
        # 提取目標值
        y = df[target_col].values
        
        # 排除缺失目標值的行
        valid_idx = ~np.isnan(y)
        if not np.all(valid_idx):
            print(f"排除{np.sum(~valid_idx)}行，因為它們的目標值缺失")
            df = df[valid_idx].reset_index(drop=True)
            y = y[valid_idx]
        
        # 從DataFrame中提取SMILES和濃度數據
        smiles_data = []
        concentration_data = []
        
        for idx, row in df.iterrows():
            smiles_list = []
            conc_list = []
            
            # 收集每行的有效SMILES和濃度
            for smiles_col, conc_col in zip(smiles_cols, conc_cols):
                smiles = row[smiles_col]
                conc = row[conc_col]
                
                # 只添加有效的SMILES和濃度對
                if isinstance(smiles, str) and pd.notna(smiles) and pd.notna(conc):
                    try:
                        # 確認SMILES是有效的
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            smiles_list.append(smiles)
                            conc_list.append(float(conc))
                        else:
                            print(f"警告: 行 {idx+1} 中的 SMILES '{smiles}' 無效，已跳過")
                    except Exception as e:
                        print(f"警告: 處理行 {idx+1} 時出錯: {e}")
            
            # 確保至少有一個有效的SMILES-濃度對
            if smiles_list and conc_list:
                smiles_data.append(smiles_list)
                concentration_data.append(conc_list)
            else:
                print(f"警告: 行 {idx+1} 沒有有效的SMILES-濃度對，已跳過")
        
        # 檢查有效數據量
        if len(smiles_data) == 0:
            raise ValueError("沒有找到有效的SMILES-濃度數據")
            
        print(f"處理後的有效樣本數: {len(smiles_data)}")
        
        # 初始化SMILES編碼器
        smiles_encoder = MultiSMILESEncoder(max_smiles=max_smiles)
        
        # 分割數據
        y = df[target_col].values[:len(smiles_data)]  # 確保目標值與有效樣本數量匹配
        
        # 分割為訓練集和測試集
        indices = np.arange(len(smiles_data))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_smiles = [smiles_data[i] for i in train_idx]
        train_concentrations = [concentration_data[i] for i in train_idx]
        train_y = y[train_idx]
        
        test_smiles = [smiles_data[i] for i in test_idx]
        test_concentrations = [concentration_data[i] for i in test_idx]
        test_y = y[test_idx]
        
        print(f"訓練集: {len(train_smiles)} 樣本")
        print(f"測試集: {len(test_smiles)} 樣本")
        
        # 目標值縮放
        scaler_y = RobustScaler()
        train_y_scaled = scaler_y.fit_transform(train_y.reshape(-1, 1)).flatten()
        
        # 轉換為張量
        train_y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32)
        
        return (train_smiles, train_concentrations, train_y_tensor, 
                test_smiles, test_concentrations, test_y, 
                scaler_y, smiles_encoder)
    
    except Exception as e:
        print(f"加載數據時出錯: {e}")
        import traceback
        traceback.print_exc()
        raise

# --- 訓練函數 ---
def train_model(model, model_name, train_smiles, train_concentrations, train_y,
                epochs=50, batch_size=32, lr=1e-4, weight_decay=1e-4,
                save_dir="multi_smiles_qnn_results", early_stop_patience=10):
    """
    訓練多SMILES量子回歸模型
    
    參數:
        model: 模型實例
        model_name: 模型名稱（用於保存）
        train_smiles: 訓練集SMILES列表的列表
        train_concentrations: 訓練集濃度列表的列表
        train_y: 訓練集目標值張量
        epochs: 訓練週期數
        batch_size: 批次大小
        lr: 學習率
        weight_decay: 權重衰減係數
        save_dir: 保存目錄
        early_stop_patience: 早停耐心值
    
    返回:
        訓練好的模型
    """
    # 建立保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 將模型移至設備
    model.to(DEVICE)
    
    # 定義優化器、損失函數和學習率調度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 準備訓練資料
    dataset_size = len(train_smiles)
    
    # 初始化訓練記錄
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 建立驗證集（從訓練集中分割）
    val_size = min(int(dataset_size * 0.1), 20)  # 使用10%數據作為驗證集，最多20個樣本
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # 訓練循環
    print(f"開始訓練{model_name}，共{epochs}個週期...")
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        # 創建批次
        num_batches = (len(train_indices) + batch_size - 1) // batch_size
        
        # 使用tqdm顯示進度條
        with tqdm(total=num_batches, desc=f"週期 {epoch+1}/{epochs} [訓練]") as pbar:
            for batch_idx in range(num_batches):
                # 獲取當前批次的索引
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_indices))
                batch_indices = train_indices[start_idx:end_idx]
                
                # 準備批次數據
                batch_smiles = [train_smiles[i] for i in batch_indices]
                batch_concentrations = [train_concentrations[i] for i in batch_indices]
                batch_y = train_y[batch_indices].to(DEVICE)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向傳播
                outputs = model(batch_smiles, batch_concentrations)
                
                # 計算損失
                loss = loss_fn(outputs.squeeze(), batch_y)
                
                # 如果損失是NaN，則跳過此批次
                if torch.isnan(loss):
                    print(f"警告: 批次 {batch_idx+1} 產生NaN損失，跳過")
                    continue
                
                # 反向傳播和優化
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 記錄損失
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        
        # 計算平均訓練損失
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        train_losses.append(avg_train_loss)
        
        # 驗證
        model.eval()
        val_losses_epoch = []
        
        with torch.no_grad():
            # 使用驗證集
            val_smiles = [train_smiles[i] for i in val_indices]
            val_concentrations = [train_concentrations[i] for i in val_indices]
            val_y = train_y[val_indices].to(DEVICE)
            
            # 分批次評估（如果驗證集很大）
            val_batch_size = min(batch_size, len(val_indices))
            num_val_batches = (len(val_indices) + val_batch_size - 1) // val_batch_size
            
            for val_batch_idx in range(num_val_batches):
                start_idx = val_batch_idx * val_batch_size
                end_idx = min(start_idx + val_batch_size, len(val_indices))
                batch_indices = list(range(start_idx, end_idx))
                
                batch_val_smiles = [val_smiles[i] for i in batch_indices]
                batch_val_concentrations = [val_concentrations[i] for i in batch_indices]
                batch_val_y = val_y[batch_indices]
                
                # 前向傳播
                val_outputs = model(batch_val_smiles, batch_val_concentrations)
                
                # 計算損失
                val_loss = loss_fn(val_outputs.squeeze(), batch_val_y)
                if not torch.isnan(val_loss):
                    val_losses_epoch.append(val_loss.item())
        
        # 計算平均驗證損失
        avg_val_loss = sum(val_losses_epoch) / len(val_losses_epoch) if val_losses_epoch else float('inf')
        val_losses.append(avg_val_loss)
        
        # 更新學習率調度器
        scheduler.step(avg_val_loss)
        
        # 輸出當前週期損失
        print(f"週期 {epoch+1}/{epochs} - 訓練損失: {avg_train_loss:.6f}, 驗證損失: {avg_val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停檢查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            
            # 保存最佳模型
            torch.save({
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, os.path.join(save_dir, f"{model_name}_best_model.pt"))
            print(f"保存最佳模型，驗證損失: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"早停觸發，最佳驗證損失: {best_val_loss:.6f}")
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break
    
    # 繪製損失曲線
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='訓練損失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='驗證損失')
    plt.xlabel('週期')
    plt.ylabel('損失')
    plt.title(f'{model_name} 訓練曲線')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss_curve.png"))
    plt.close()
    
    # 確保使用最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

# --- 評估函數 ---
def evaluate_model(model, test_smiles, test_concentrations, test_y, scaler_y, model_name, save_dir):
    """
    評估模型性能
    
    參數:
        model: 訓練好的模型
        test_smiles: 測試集SMILES列表的列表
        test_concentrations: 測試集濃度列表的列表
        test_y: 測試集目標值（未縮放）
        scaler_y: 目標值縮放器
        model_name: 模型名稱
        save_dir: 保存目錄
    
    返回:
        評估指標字典
    """
    # 確保保存目錄存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 設置模型為評估模式
    model.eval()
    
    # 執行預測
    with torch.no_grad():
        # 對測試數據進行預測（可能需要分批次處理）
        batch_size = 32
        num_samples = len(test_smiles)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        all_predictions = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_smiles = test_smiles[start_idx:end_idx]
            batch_concentrations = test_concentrations[start_idx:end_idx]
            
            # 預測
            batch_predictions = model(batch_smiles, batch_concentrations).cpu().numpy()
            all_predictions.append(batch_predictions)
        
        # 合併所有預測結果
        predictions_scaled = np.vstack(all_predictions).squeeze()
        
        # 將預測值從縮放空間轉換回原始空間
        predictions = scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
    
    # 計算評估指標
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    # 顯示評估結果
    print(f"\n{model_name} 評估結果:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")
    
    # 繪製散點圖
    plt.figure(figsize=(10, 6))
    plt.scatter(test_y, predictions, alpha=0.6)
    
    # 添加對角線（理想預測線）
    min_val = min(min(test_y), min(predictions))
    max_val = max(max(test_y), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('實際值')
    plt.ylabel('預測值')
    plt.title(f'{model_name} 預測 vs 實際值 (R² = {r2:.4f})')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_scatter_plot.png"))
    plt.close()
    
    # 保存預測結果
    results_df = pd.DataFrame({
        'Actual': test_y,
        'Predicted': predictions,
        'Error': np.abs(test_y - predictions)
    })
    results_df.to_csv(os.path.join(save_dir, f"{model_name}_predictions.csv"), index=False)
    
    # 保存評估指標
    pd.DataFrame([metrics]).T.to_csv(
        os.path.join(save_dir, f"{model_name}_metrics.csv"), 
        header=['Value']
    )
    
    return metrics

# --- 主程序 ---
def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="多SMILES輸入量子回歸模型")
    
    # 資料相關參數
    parser.add_argument("--excel-path", type=str, required=True, help="Excel數據文件路徑")
    parser.add_argument("--target-col", type=str, default="LCE", help="目標列名 (預設: LCE)")
    parser.add_argument("--max-smiles", type=int, default=6, help="每個樣本的最大SMILES數量 (預設: 6)")
    
    # 模型相關參數
    parser.add_argument("--model", type=str, default="quantum", choices=["quantum", "mlp", "transformer"],
                        help="模型類型: quantum, mlp, transformer (預設: quantum)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="隱藏層維度 (預設: 256)")
    parser.add_argument("--embed-dim", type=int, default=256, help="嵌入層維度 (預設: 256)")
    parser.add_argument("--qnn-n-qubits", type=int, default=8, help="量子層中的量子比特數 (預設: 8)")
    parser.add_argument("--qnn-n-layers", type=int, default=3, help="量子層中的層數 (預設: 3)")
    
    # 訓練相關參數
    parser.add_argument("--epochs", type=int, default=50, help="訓練週期數 (預設: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小 (預設: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="學習率 (預設: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="權重衰減 (預設: 1e-4)")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值 (預設: 10)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率 (預設: 0.1)")
    
    # 其他參數
    parser.add_argument("--save-dir", type=str, default="multi_smiles_qnn_results", 
                        help="保存結果的目錄 (預設: multi_smiles_qnn_results)")
    parser.add_argument("--debug", action="store_true", help="開啟調試模式（較小模型、較少週期）")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設: 42)")
    
    args = parser.parse_args()
    
    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 調試模式設置
    if args.debug:
        print("以調試模式運行 - 使用較小的模型和較少的訓練週期")
        args.epochs = 5
        args.batch_size = 16
        args.hidden_dim = 128
        args.embed_dim = 128
        args.qnn_n_qubits = 4
        args.qnn_n_layers = 2
    
    try:
        # 載入數據
        print(f"\n載入數據從 {args.excel_path}...")
        data = load_data_with_multi_smiles(
            args.excel_path, 
            target_col=args.target_col, 
            max_smiles=args.max_smiles
        )
        
        if data is None:
            raise ValueError("數據載入失敗")
            
        # 解包數據
        (train_smiles, train_concentrations, train_y_tensor, 
         test_smiles, test_concentrations, test_y, 
         scaler_y, smiles_encoder) = data
        
        # 確定模型目錄
        model_name = f"{args.model}_model"
        model_dir = os.path.join(args.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 建立模型
        if args.model == "quantum":
            print("\n初始化量子回歸模型...")
            model = MultiSMILESQuantumRegressionNet(
                smiles_encoder=smiles_encoder,
                output_dim=1,  # LCE預測為單一目標
                hidden_dim=args.hidden_dim,
                n_qubits=args.qnn_n_qubits,
                n_layers=args.qnn_n_layers,
                embed_dim=args.embed_dim,
                dropout_rate=args.dropout
            )
        elif args.model == "mlp":
            print("\n初始化MLP基準模型...")
            # 創建SMILES嵌入+MLP模型
            embedding_model = SMILESEmbeddingModel(
                input_dim=smiles_encoder.feature_size, 
                embedding_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                dropout_rate=args.dropout
            )
            
            class MultiSMILESMLPModel(nn.Module):
                def __init__(self, embedding_model, mlp_model):
                    super().__init__()
                    self.embedding_model = embedding_model
                    self.mlp_model = mlp_model
                    self.smiles_encoder = smiles_encoder
                    
                def forward(self, smiles_batch, concentrations_batch):
                    encoded_features = self.smiles_encoder.encode_to_tensor(
                        smiles_batch, concentrations_batch
                    ).to(next(self.parameters()).device)
                    embedded_features = self.embedding_model(encoded_features)
                    return self.mlp_model(embedded_features)
            
            mlp = MLPBaseline(
                input_dim=args.embed_dim,
                output_dim=1,
                hidden_dim=args.hidden_dim,
                dropout_rate=args.dropout
            )
            
            model = MultiSMILESMLPModel(embedding_model, mlp)
            
        elif args.model == "transformer":
            print("\n初始化Transformer基準模型...")
            # 創建SMILES嵌入+Transformer模型
            embedding_model = SMILESEmbeddingModel(
                input_dim=smiles_encoder.feature_size, 
                embedding_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                dropout_rate=args.dropout
            )
            
            class MultiSMILESTransformerModel(nn.Module):
                def __init__(self, embedding_model, transformer_model):
                    super().__init__()
                    self.embedding_model = embedding_model
                    self.transformer_model = transformer_model
                    self.smiles_encoder = smiles_encoder
                    
                def forward(self, smiles_batch, concentrations_batch):
                    encoded_features = self.smiles_encoder.encode_to_tensor(
                        smiles_batch, concentrations_batch
                    ).to(next(self.parameters()).device)
                    embedded_features = self.embedding_model(encoded_features)
                    return self.transformer_model(embedded_features)
            
            transformer = TransformerBaseline(
                input_dim=args.embed_dim,
                output_dim=1,
                hidden_dim=args.hidden_dim,
                nhead=4,  # 可以作為參數暴露
                num_layers=2,  # 可以作為參數暴露
                dropout_rate=args.dropout
            )
            
            model = MultiSMILESTransformerModel(embedding_model, transformer)
            
        else:
            raise ValueError(f"不支援的模型類型: {args.model}")
        
        # 輸出模型信息
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型參數數量: {num_params:,}")
        
        # 訓練模型
        print(f"\n開始訓練{model_name}...")
        trained_model = train_model(
            model=model,
            model_name=model_name,
            train_smiles=train_smiles,
            train_concentrations=train_concentrations,
            train_y=train_y_tensor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_dir=model_dir,
            early_stop_patience=args.patience
        )
        
        # 評估模型
        print(f"\n評估{model_name}...")
        metrics = evaluate_model(
            model=trained_model,
            test_smiles=test_smiles,
            test_concentrations=test_concentrations,
            test_y=test_y,
            scaler_y=scaler_y,
            model_name=model_name,
            save_dir=model_dir
        )
        
        print(f"\n{model_name} 訓練和評估完成!")
        print(f"結果已保存在 {model_dir} 目錄")
        
        # 最終結果摘要
        print("\n最終評估指標:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")
            
    except Exception as e:
        print(f"\n運行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

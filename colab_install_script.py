# 安裝所需的庫
!pip install -q torch numpy pandas matplotlib scikit-learn rdkit pennylane pennylane-lightning
!pip install -q tqdm xgboost lightgbm

# 克隆代碼庫（如果需要）
# !git clone https://github.com/yourusername/your-repo.git
# !cd your-repo

# 下載數據集（如果需要）
# !wget -q https://example.com/pnas.2214357120.sd01.xlsx

# 顯示環境信息
import sys
import torch
import numpy as np
import pandas as pd
import pennylane as qml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# 設置RDKit的日誌級別
RDLogger.DisableLog('rdApp.*')

print("Python版本:", sys.version)
print("PyTorch版本:", torch.__version__)
print("NumPy版本:", np.__version__)
print("Pandas版本:", pd.__version__)
print("PennyLane版本:", qml.version())
print("RDKit版本:", Chem.__version__)

# 檢查是否可以使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

if torch.cuda.is_available():
    print(f"可用的GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 測試PennyLane量子設備是否正常工作
try:
    dev = qml.device("default.qubit", wires=2)
    print("PennyLane默認量子設備測試成功")
    
    # 嘗試lightning.qubit設備
    try:
        dev_lightning = qml.device("lightning.qubit", wires=2)
        print("PennyLane lightning.qubit設備測試成功")
    except Exception as e:
        print(f"PennyLane lightning.qubit設備測試失敗: {e}")
        print("這不會影響模型運行，因為我們會自動回退到default.qubit設備")
        
except Exception as e:
    print(f"PennyLane設備測試失敗: {e}")
    print("請確保PennyLane已正確安裝")

# 測試RDKit是否能處理SMILES
try:
    # 測試SMILES處理
    sample_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林的SMILES
    mol = Chem.MolFromSmiles(sample_smiles)
    if mol:
        print(f"RDKit成功處理SMILES字串。分子名稱: 阿司匹林，原子數: {mol.GetNumAtoms()}")
        
        # 測試Morgan指紋生成
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        print(f"成功生成Morgan指紋，長度: {len(fp)}")
    else:
        print("RDKit無法處理測試SMILES字串")
except Exception as e:
    print(f"RDKit測試失敗: {e}")
    print("請確保RDKit已正確安裝")

print("\n環境測試完成，可以開始運行SMILES量子回歸模型。")

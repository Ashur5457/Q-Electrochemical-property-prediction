# Q-Electrochemical-property-prediction

# SMILES量子回歸模型使用說明

本文檔說明如何在 Google Colab 上運行 SMILES 輸入的量子回歸模型，用於預測分子的電化學庫倫效率 (LCE)。

## 文件說明

本實現包含以下主要文件：

1. **smiles_encoder.py** - SMILES字符串編碼器，將分子結構轉換為數值特徵
2. **smiles_quantum_regression.py** - 主模型實現和訓練代碼
3. **colab_install_script.py** - 在Colab上設置環境的腳本

## 在 Google Colab 上運行步驟

1. 首先，在新的 Colab 筆記本中運行以下命令來設置環境：

```python
# 創建必要的文件
%%writefile smiles_encoder.py
# 這裡貼上 smiles_encoder.py 的完整代碼

%%writefile smiles_quantum_regression.py
# 這裡貼上 smiles_quantum_regression.py 和 smiles_quantum_regression_rest.py 的合併代碼
```

2. 然後運行安裝腳本：

```python
# 安裝所需的庫
!pip install -q torch numpy pandas matplotlib scikit-learn rdkit pennylane pennylane-lightning
!pip install -q tqdm xgboost lightgbm

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

# 測試環境
# ...其他測試代碼從colab_install_script.py複製...
```

3. 上傳數據集文件：
   - 點擊 Colab 左側的文件圖標
   - 點擊上傳按鈕上傳 `pnas.2214357120.sd01.xlsx` 文件

4. 運行模型：

```python
# 使用量子模型 (基本參數)
!python smiles_quantum_regression.py --model quantum --target-col LCE --epochs 50 --batch-size 32

# 或使用調試模式 (更快訓練)
!python smiles_quantum_regression.py --model quantum --target-col LCE --debug
```

## 命令行參數說明

以下是可用的命令行參數：

```
數據參數:
  --excel-path       Excel數據文件的路徑 (默認: pnas.2214357120.sd01.xlsx)
  --sheet-name       Excel工作表名稱 (默認: Dataset)
  --smiles-col       包含SMILES字串的列名 (默認: SMILES)
  --target-col       目標列名，如LCE (默認: LCE)

模型參數:
  --model            模型類型: quantum, mlp, transformer (默認: quantum)
  --hidden-dim       隱藏層維度 (默認: 256)
  --qnn-n-qubits     量子比特數 (默認: 8)
  --qnn-n-layers     量子層數 (默認: 3)
  --embed-dim        SMILES嵌入維度 (默認: 256)

訓練參數:
  --epochs           訓練週期數 (默認: 50)
  --batch-size       批量大小 (默認: 32)
  --lr               學習率 (默認: 1e-4)
  --weight-decay     權重衰減 (默認: 1e-4)
  --patience         早停耐心值 (默認: 10)
  --debug            以調試模式運行
  --save-dir         保存結果的目錄 (默認: smiles_qnn_results)
```

## 測試不同模型

您可以嘗試比較不同的模型架構：

```python
# 量子模型
!python smiles_quantum_regression.py --model quantum --target-col LCE

# MLP基準模型
!python smiles_quantum_regression.py --model mlp --target-col LCE

# Transformer基準模型
!python smiles_quantum_regression.py --model transformer --target-col LCE
```

## 結果解讀

運行完成後，在 `smiles_qnn_results_*` 目錄中可以找到以下結果文件：

- `metrics.csv` - 包含模型性能的度量指標 (RMSE, MAE, R2等)
- `predictions.csv` - 模型在測試集上的預測值
- `scatter_*.png` - 真實值與預測值的散點圖
- `loss_curves.png` - 訓練和驗證損失曲線
- `best_model.pt` - 保存的最佳模型權重

## 常見問題解決

1. **記憶體不足錯誤**：
   - 減少 `batch-size` 參數
   - 使用 `--debug` 模式運行
   - 在 Colab 中啟用高內存運行時 (Runtime > Change runtime type)

2. **運行緩慢**：
   - 使用較小的量子層：`--qnn-n-qubits 4 --qnn-n-layers 2`
   - 減少訓練週期：`--epochs 20`
   - 使用較小的隱藏維度：`--hidden-dim 128 --embed-dim 128`

3. **SMILES解析錯誤**：
   - 檢查Excel文件中SMILES列的格式
   - 確保正確指定了SMILES列名：`--smiles-col 列名`

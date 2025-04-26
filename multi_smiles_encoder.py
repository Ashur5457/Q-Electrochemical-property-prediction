import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, MolSurf
import torch
from torch import nn

class MultiSMILESEncoder:
    """
    將多個SMILES字串及其濃度轉換為數值向量表示的編碼器。
    支援多種分子特徵提取方法，包括：Morgan指紋、RDKit分子描述符等。
    """
    def __init__(self, max_smiles=6, encoding_method='combined', fingerprint_size=2048, 
                 radius=2, use_features=False, device='cpu'):
        """
        初始化多SMILES編碼器。
        
        參數:
            max_smiles (int): 最大SMILES輸入數量
            encoding_method (str): 編碼方法，可選 'morgan', 'descriptors', 'combined'
            fingerprint_size (int): Morgan指紋的長度
            radius (int): Morgan指紋的半徑
            use_features (bool): 是否使用特徵Morgan指紋
            device (str): 設備，'cpu'或'cuda'
        """
        self.max_smiles = max_smiles
        self.encoding_method = encoding_method
        self.fingerprint_size = fingerprint_size
        self.radius = radius
        self.use_features = use_features
        self.device = device
        self.descriptor_names = self._get_descriptor_names()
        self.single_smiles_feature_size = self._get_single_feature_size()
        self.feature_size = self.single_smiles_feature_size * max_smiles + 1  # +1 for concentration
        print(f"已初始化多SMILES編碼器: 方法={encoding_method}, 最大SMILES數={max_smiles}")
        print(f"單個SMILES特徵維度={self.single_smiles_feature_size}, 總特徵維度={self.feature_size}")
    
    def _get_descriptor_names(self):
        """取得RDKit分子描述符的名稱列表"""
        descriptors = [
            # 基本物理化學特性
            'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
            'FractionCSP3', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors',
            'NumHeteroatoms', 'NumAmideBonds', 'NumRings', 'NumAromaticRings',
            'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles',
            'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles',
            # LogP相關
            'MolLogP', 'MolMR',
            # 拓撲特性
            'BalabanJ', 'BertzCT', 'Chi0', 'Chi1',
            # 表面特性
            'LabuteASA', 'TPSA'
        ]
        return descriptors
    
    def _get_single_feature_size(self):
        """計算單個SMILES特徵向量的長度"""
        if self.encoding_method == 'morgan':
            return self.fingerprint_size
        elif self.encoding_method == 'descriptors':
            return len(self.descriptor_names)
        elif self.encoding_method == 'combined':
            return self.fingerprint_size + len(self.descriptor_names)
        else:
            raise ValueError(f"不支援的編碼方法: {self.encoding_method}")
    
    def _get_morgan_fingerprint(self, mol):
        """計算分子的Morgan指紋"""
        if not mol:
            return np.zeros(self.fingerprint_size)
        if self.use_features:
            fp = AllChem.GetMorganFeaturesFingerprint(mol, self.radius)
        else:
            fp = AllChem.GetMorganFingerprint(mol, self.radius)
        fp_array = np.zeros(self.fingerprint_size)
        for idx, val in fp.GetNonzeroElements().items():
            idx_in_array = idx % self.fingerprint_size
            fp_array[idx_in_array] += val
        return fp_array
    
    def _get_descriptors(self, mol):
        """計算分子的RDKit描述符"""
        if not mol:
            return np.zeros(len(self.descriptor_names))
        
        desc_values = []
        for desc_name in self.descriptor_names:
            try:
                desc_func = getattr(Descriptors, desc_name)
                value = desc_func(mol)
            except:
                # 若遇到錯誤，設為0
                value = 0
            desc_values.append(value)
        
        return np.array(desc_values)
    
    def encode_smiles(self, smiles):
        """
        將單個SMILES字串編碼為數值向量。
        
        參數:
            smiles (str): 要編碼的SMILES字串
            
        返回:
            numpy.ndarray: 編碼後的特徵向量
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                print(f"警告: 無法解析SMILES字串: {smiles}")
                return np.zeros(self.single_smiles_feature_size)
            
            if self.encoding_method == 'morgan':
                return self._get_morgan_fingerprint(mol)
            elif self.encoding_method == 'descriptors':
                return self._get_descriptors(mol)
            elif self.encoding_method == 'combined':
                morgan_fp = self._get_morgan_fingerprint(mol)
                descriptors = self._get_descriptors(mol)
                return np.concatenate([morgan_fp, descriptors])
        except Exception as e:
            print(f"處理SMILES時出錯 ({smiles}): {e}")
            return np.zeros(self.single_smiles_feature_size)
    
    def encode_multiple_smiles(self, smiles_list, concentrations):
        """
        將多個SMILES字串及其對應濃度編碼為單一特徵向量。
        
        參數:
            smiles_list (list): 要編碼的SMILES字串列表
            concentrations (list): 對應的濃度列表
            
        返回:
            numpy.ndarray: 編碼後的特徵向量，包含所有SMILES特徵和總濃度
        """
        # 確保不超過最大SMILES數量
        if len(smiles_list) > self.max_smiles:
            print(f"警告: 提供的SMILES數量({len(smiles_list)})超過最大數量({self.max_smiles})，將截斷")
            smiles_list = smiles_list[:self.max_smiles]
            concentrations = concentrations[:self.max_smiles]
        
        # 編碼每個SMILES
        encoded_smiles = []
        for i, smiles in enumerate(smiles_list):
            encoded = self.encode_smiles(smiles)
            encoded_smiles.append(encoded)
        
        # 填充到最大SMILES數量
        while len(encoded_smiles) < self.max_smiles:
            encoded_smiles.append(np.zeros(self.single_smiles_feature_size))
        
        # 計算總濃度
        total_concentration = sum(concentrations)
        
        # 組合所有特徵
        all_features = np.concatenate(encoded_smiles + [np.array([total_concentration])])
        return all_features
    
    def encode_batch(self, batch_smiles_list, batch_concentrations):
        """
        編碼一批SMILES及濃度數據。
        
        參數:
            batch_smiles_list (list of lists): 每個樣本的SMILES列表的列表
            batch_concentrations (list of lists): 每個樣本的濃度列表的列表
            
        返回:
            numpy.ndarray: 編碼後的特徵矩陣，形狀為 (batch_size, feature_size)
        """
        features = []
        for smiles_list, concentrations in zip(batch_smiles_list, batch_concentrations):
            features.append(self.encode_multiple_smiles(smiles_list, concentrations))
        return np.array(features)
    
    def encode_to_tensor(self, batch_smiles_list, batch_concentrations):
        """
        將一批SMILES及濃度數據編碼為PyTorch張量。
        
        參數:
            batch_smiles_list (list of lists): 每個樣本的SMILES列表的列表
            batch_concentrations (list of lists): 每個樣本的濃度列表的列表
            
        返回:
            torch.Tensor: 編碼後的特徵張量，形狀為 (batch_size, feature_size)
        """
        features = self.encode_batch(batch_smiles_list, batch_concentrations)
        return torch.tensor(features, dtype=torch.float32, device=self.device)

class SMILESEmbeddingModel(nn.Module):
    """將編碼的特徵嵌入到較低維度空間的編碼器，可以進行反向傳播訓練"""
    def __init__(self, input_dim, embedding_dim=256, hidden_dim=512, dropout_rate=0.1):
        super(SMILESEmbeddingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

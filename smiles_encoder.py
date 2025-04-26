import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, MolSurf
import torch
from torch import nn

class SMILESEncoder:
    """
    將SMILES字串轉換為數值向量表示的編碼器。
    支援多種分子特徵提取方法，包括：Morgan指紋、RDKit分子描述符、One-hot編碼等。
    """
    def __init__(self, encoding_method='combined', fingerprint_size=2048, 
                 radius=2, use_features=False, device='cpu'):
        """
        初始化SMILES編碼器。
        
        參數:
            encoding_method (str): 編碼方法，可選 'morgan', 'descriptors', 'combined'
            fingerprint_size (int): Morgan指紋的長度
            radius (int): Morgan指紋的半徑
            use_features (bool): 是否使用特徵Morgan指紋
            device (str): 設備，'cpu'或'cuda'
        """
        self.encoding_method = encoding_method
        self.fingerprint_size = fingerprint_size
        self.radius = radius
        self.use_features = use_features
        self.device = device
        self.descriptor_names = self._get_descriptor_names()
        self.feature_size = self._get_feature_size()
        print(f"已初始化SMILES編碼器: 方法={encoding_method}, 特徵維度={self.feature_size}")
    
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
    
    def _get_feature_size(self):
        """計算特徵向量的總長度"""
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
                return np.zeros(self.feature_size)
            
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
            return np.zeros(self.feature_size)
    
    def encode_smiles_list(self, smiles_list):
        """
        將多個SMILES字串編碼為數值向量。
        
        參數:
            smiles_list (list): 要編碼的SMILES字串列表
            
        返回:
            numpy.ndarray: 編碼後的特徵矩陣，形狀為 (len(smiles_list), feature_size)
        """
        features = []
        for smiles in smiles_list:
            features.append(self.encode_smiles(smiles))
        return np.array(features)
    
    def encode_to_tensor(self, smiles_list):
        """
        將SMILES字串列表編碼為PyTorch張量。
        
        參數:
            smiles_list (list): 要編碼的SMILES字串列表
            
        返回:
            torch.Tensor: 編碼後的特徵張量，形狀為 (len(smiles_list), feature_size)
        """
        features = self.encode_smiles_list(smiles_list)
        return torch.tensor(features, dtype=torch.float32, device=self.device)

class SMILESEmbeddingModel(nn.Module):
    """將Morgan指紋嵌入到較低維度空間的編碼器，可以進行反向傳播訓練"""
    def __init__(self, input_dim, embedding_dim=256, hidden_dim=512):
        super(SMILESEmbeddingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

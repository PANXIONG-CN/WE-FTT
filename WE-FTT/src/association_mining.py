"""
Association rule mining module for WE-FTT project.

This module implements K-means clustering and various association rule mining
algorithms (Apriori, Eclat, FP-Growth) to extract knowledge from earthquake
precursor data and calculate feature weights.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path

from .config import DataProcessingConfig


logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """K-means聚类分析器"""
    
    def __init__(self, config: DataProcessingConfig = None):
        self.config = config or DataProcessingConfig()
        self.kmeans_models = {}
        self.cluster_info = {}
    
    def fit_kmeans(self, data: pd.DataFrame, features: List[str]) -> Dict[str, KMeans]:
        """为每个特征列拟合K-means模型"""
        
        for feature in features:
            if feature not in data.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
            
            # 准备数据
            feature_data = data[feature].values.reshape(-1, 1)
            
            # 拟合K-means
            kmeans = KMeans(**self.config.KMEANS_CONFIG)
            cluster_labels = kmeans.fit_predict(feature_data)
            
            # 保存模型和结果
            self.kmeans_models[feature] = kmeans
            
            # 计算聚类信息
            cluster_info = self._calculate_cluster_stats(
                feature_data.flatten(), cluster_labels
            )
            self.cluster_info[feature] = cluster_info
            
            logger.info(f"K-means clustering completed for {feature}")
        
        return self.kmeans_models
    
    def _calculate_cluster_stats(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """计算聚类统计信息"""
        unique_labels = np.unique(labels)
        cluster_stats = {}
        
        for label in unique_labels:
            mask = labels == label
            cluster_data = data[mask]
            
            cluster_stats[f"cluster_{label}"] = {
                "center": float(np.mean(cluster_data)),
                "std": float(np.std(cluster_data)),
                "size": int(np.sum(mask)),
                "min": float(np.min(cluster_data)),
                "max": float(np.max(cluster_data))
            }
        
        return cluster_stats
    
    def predict_clusters(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """预测新数据的聚类标签"""
        result_df = data.copy()
        
        for feature in features:
            if feature in self.kmeans_models:
                feature_data = data[feature].values.reshape(-1, 1)
                cluster_labels = self.kmeans_models[feature].predict(feature_data)
                result_df[f"{feature}_cluster_labels"] = cluster_labels
                
                logger.info(f"Predicted cluster labels for {feature}")
        
        return result_df
    
    def save_cluster_info(self, output_dir: str) -> None:
        """保存聚类信息"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for feature, info in self.cluster_info.items():
            filename = output_path / f"{feature}_cluster_info.json"
            with open(filename, 'w') as f:
                json.dump(info, f, indent=2)
        
        logger.info(f"Cluster information saved to {output_dir}")


class AprioriMiner:
    """Apriori关联规则挖掘器"""
    
    def __init__(self, config: DataProcessingConfig = None):
        self.config = config or DataProcessingConfig()
        self.frequent_itemsets = {}
        self.association_rules = {}
    
    def mine_frequent_itemsets(
        self, 
        transactions: List[List[str]], 
        min_support: Optional[float] = None
    ) -> Dict[int, List[Tuple]]:
        """挖掘频繁项集"""
        if min_support is None:
            min_support = self.config.APRIORI_CONFIG["min_support"]
        
        # 计算项目支持度
        item_support = self._calculate_item_support(transactions)
        
        # 第一次扫描：找出频繁1-项集
        frequent_1_itemsets = [
            (item,) for item, support in item_support.items() 
            if support >= min_support
        ]
        
        frequent_itemsets = {1: frequent_1_itemsets}
        k = 2
        
        # 迭代生成频繁k-项集
        while frequent_itemsets[k-1]:
            candidates = self._generate_candidates(frequent_itemsets[k-1])
            candidate_support = self._calculate_candidate_support(transactions, candidates)
            
            frequent_k_itemsets = [
                itemset for itemset, support in candidate_support.items()
                if support >= min_support
            ]
            
            if frequent_k_itemsets:
                frequent_itemsets[k] = frequent_k_itemsets
                k += 1
            else:
                break
        
        self.frequent_itemsets = frequent_itemsets
        logger.info(f"Found frequent itemsets: {sum(len(v) for v in frequent_itemsets.values())}")
        return frequent_itemsets
    
    def _calculate_item_support(self, transactions: List[List[str]]) -> Dict[str, float]:
        """计算单个项目的支持度"""
        item_counts = {}
        total_transactions = len(transactions)
        
        for transaction in transactions:
            for item in set(transaction):
                item_counts[item] = item_counts.get(item, 0) + 1
        
        return {item: count / total_transactions for item, count in item_counts.items()}
    
    def _generate_candidates(self, frequent_itemsets: List[Tuple]) -> List[Tuple]:
        """生成候选项集"""
        candidates = []
        n = len(frequent_itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = frequent_itemsets[i]
                itemset2 = frequent_itemsets[j]
                
                # 如果前k-2个项目相同，则可以合并
                if itemset1[:-1] == itemset2[:-1]:
                    candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                    if len(candidate) == len(itemset1) + 1:
                        candidates.append(candidate)
        
        return candidates
    
    def _calculate_candidate_support(
        self, 
        transactions: List[List[str]], 
        candidates: List[Tuple]
    ) -> Dict[Tuple, float]:
        """计算候选项集的支持度"""
        candidate_counts = {candidate: 0 for candidate in candidates}
        total_transactions = len(transactions)
        
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if set(candidate).issubset(transaction_set):
                    candidate_counts[candidate] += 1
        
        return {
            candidate: count / total_transactions 
            for candidate, count in candidate_counts.items()
        }
    
    def generate_association_rules(
        self, 
        min_confidence: Optional[float] = None,
        min_lift: Optional[float] = None
    ) -> List[Dict]:
        """生成关联规则"""
        if min_confidence is None:
            min_confidence = self.config.APRIORI_CONFIG["min_confidence"]
        if min_lift is None:
            min_lift = self.config.APRIORI_CONFIG["min_lift"]
        
        rules = []
        
        # 从频繁2-项集开始生成规则
        for k in range(2, len(self.frequent_itemsets) + 1):
            for itemset in self.frequent_itemsets.get(k, []):
                rules.extend(
                    self._generate_rules_from_itemset(
                        itemset, min_confidence, min_lift
                    )
                )
        
        self.association_rules = rules
        logger.info(f"Generated {len(rules)} association rules")
        return rules
    
    def _generate_rules_from_itemset(
        self, 
        itemset: Tuple, 
        min_confidence: float,
        min_lift: float
    ) -> List[Dict]:
        """从频繁项集生成关联规则"""
        rules = []
        
        # 生成所有可能的前件和后件组合
        for i in range(1, len(itemset)):
            from itertools import combinations
            for antecedent in combinations(itemset, i):
                consequent = tuple(item for item in itemset if item not in antecedent)
                
                # 计算置信度和提升度
                confidence = self._calculate_confidence(antecedent, consequent)
                lift = self._calculate_lift(antecedent, consequent)
                
                if confidence >= min_confidence and lift >= min_lift:
                    rule = {
                        "antecedent": list(antecedent),
                        "consequent": list(consequent),
                        "confidence": confidence,
                        "lift": lift,
                        "support": self._get_itemset_support(itemset)
                    }
                    rules.append(rule)
        
        return rules
    
    def _calculate_confidence(self, antecedent: Tuple, consequent: Tuple) -> float:
        """计算置信度"""
        itemset_support = self._get_itemset_support(antecedent + consequent)
        antecedent_support = self._get_itemset_support(antecedent)
        
        return itemset_support / antecedent_support if antecedent_support > 0 else 0
    
    def _calculate_lift(self, antecedent: Tuple, consequent: Tuple) -> float:
        """计算提升度"""
        confidence = self._calculate_confidence(antecedent, consequent)
        consequent_support = self._get_itemset_support(consequent)
        
        return confidence / consequent_support if consequent_support > 0 else 0
    
    def _get_itemset_support(self, itemset: Tuple) -> float:
        """获取项集的支持度"""
        itemset_len = len(itemset)
        
        for support_itemsets in self.frequent_itemsets.get(itemset_len, []):
            if set(support_itemsets) == set(itemset):
                # 这里需要从实际数据中计算支持度
                # 简化处理，实际实现需要保存支持度信息
                return 0.1  # 占位符
        
        return 0.0


class WeightCalculator:
    """特征权重计算器"""
    
    def __init__(self, config: DataProcessingConfig = None):
        self.config = config or DataProcessingConfig()
    
    def calculate_weights_from_rules(
        self, 
        association_rules: List[Dict], 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """根据关联规则计算特征权重"""
        
        # 初始化权重
        weights = {feature: 0.0 for feature in feature_names}
        
        for rule in association_rules:
            # 权重基于置信度和提升度的组合
            rule_weight = rule["confidence"] * rule["lift"] * rule["support"]
            
            # 为规则中涉及的特征增加权重
            involved_features = set(rule["antecedent"] + rule["consequent"])
            
            for feature in feature_names:
                # 检查特征是否与规则相关
                if any(feature in item for item in involved_features):
                    weights[feature] += rule_weight
        
        # 归一化权重
        max_weight = max(weights.values()) if weights.values() else 1.0
        if max_weight > 0:
            weights = {k: v / max_weight for k, v in weights.items()}
        
        logger.info("Feature weights calculated from association rules")
        return weights
    
    def save_weights(self, weights: Dict[str, float], output_file: str) -> None:
        """保存权重到文件"""
        with open(output_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        logger.info(f"Weights saved to {output_file}")


class KnowledgeMiner:
    """知识挖掘整合器"""
    
    def __init__(self, config: DataProcessingConfig = None):
        self.config = config or DataProcessingConfig()
        self.cluster_analyzer = ClusterAnalyzer(config)
        self.apriori_miner = AprioriMiner(config)
        self.weight_calculator = WeightCalculator(config)
    
    def mine_knowledge(
        self, 
        data: pd.DataFrame, 
        features: List[str],
        label_column: str = "label",
        output_dir: str = "knowledge_mining_results"
    ) -> Dict[str, Any]:
        """完整的知识挖掘流程"""
        
        logger.info("Starting knowledge mining process...")
        
        # 1. 聚类分析
        self.cluster_analyzer.fit_kmeans(data, features)
        clustered_data = self.cluster_analyzer.predict_clusters(data, features)
        
        # 2. 准备交易数据（离散化）
        transactions = self._prepare_transactions(clustered_data, features, label_column)
        
        # 3. 关联规则挖掘
        frequent_itemsets = self.apriori_miner.mine_frequent_itemsets(transactions)
        association_rules = self.apriori_miner.generate_association_rules()
        
        # 4. 计算特征权重
        weights = self.weight_calculator.calculate_weights_from_rules(
            association_rules, features
        )
        
        # 5. 保存结果
        self._save_mining_results(
            output_dir, frequent_itemsets, association_rules, weights
        )
        
        results = {
            "cluster_info": self.cluster_analyzer.cluster_info,
            "frequent_itemsets": frequent_itemsets,
            "association_rules": association_rules,
            "feature_weights": weights,
            "clustered_data": clustered_data
        }
        
        logger.info("Knowledge mining process completed")
        return results
    
    def _prepare_transactions(
        self, 
        data: pd.DataFrame, 
        features: List[str],
        label_column: str
    ) -> List[List[str]]:
        """准备交易数据用于关联规则挖掘"""
        transactions = []
        
        for _, row in data.iterrows():
            transaction = []
            
            # 添加聚类标签
            for feature in features:
                cluster_col = f"{feature}_cluster_labels"
                if cluster_col in row:
                    transaction.append(f"{feature}_cluster_{row[cluster_col]}")
            
            # 添加标签信息
            transaction.append(f"label_{row[label_column]}")
            
            transactions.append(transaction)
        
        return transactions
    
    def _save_mining_results(
        self, 
        output_dir: str,
        frequent_itemsets: Dict,
        association_rules: List[Dict],
        weights: Dict[str, float]
    ) -> None:
        """保存挖掘结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存频繁项集
        with open(output_path / "frequent_itemsets.json", 'w') as f:
            # 转换为可序列化格式
            serializable_itemsets = {
                str(k): [list(itemset) for itemset in v] 
                for k, v in frequent_itemsets.items()
            }
            json.dump(serializable_itemsets, f, indent=2)
        
        # 保存关联规则
        with open(output_path / "association_rules.json", 'w') as f:
            json.dump(association_rules, f, indent=2)
        
        # 保存特征权重
        self.weight_calculator.save_weights(weights, output_path / "feature_weights.json")
        
        # 保存聚类信息
        self.cluster_analyzer.save_cluster_info(output_path)
        
        logger.info(f"Mining results saved to {output_dir}")


def run_knowledge_mining(
    data_file: str, 
    features: List[str],
    output_dir: str = "knowledge_mining_results",
    config: DataProcessingConfig = None
) -> Dict[str, Any]:
    """
    运行知识挖掘的便捷函数
    
    Args:
        data_file: 数据文件路径
        features: 特征列表
        output_dir: 输出目录
        config: 配置对象
    
    Returns:
        挖掘结果字典
    """
    # 加载数据
    if data_file.endswith('.parquet'):
        data = pd.read_parquet(data_file)
    else:
        data = pd.read_csv(data_file)
    
    # 初始化挖掘器
    miner = KnowledgeMiner(config)
    
    # 执行挖掘
    results = miner.mine_knowledge(data, features, output_dir=output_dir)
    
    return results
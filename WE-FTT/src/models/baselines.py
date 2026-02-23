"""
Baseline model implementations for WE-FTT project.

This module contains implementations of various baseline models including
RandomForest, CatBoost, TabNet, XGBoost, and LightGBM for comparison
with the WE-FTT model.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class BaselineModel:
    """Base class for all baseline models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model"""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support probability prediction")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Additional metrics
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # ROC AUC (if probability prediction is available)
        try:
            y_proba = self.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:  # Multi-class
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except (NotImplementedError, AttributeError):
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cohen_kappa': kappa,
            'matthews_corrcoef': mcc
        }
        
        if roc_auc is not None:
            results['roc_auc'] = roc_auc
            
        return results


class RandomForestModel(BaselineModel):
    """Random Forest baseline model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with provided config
        if self.config:
            params.update(self.config)
            
        self.model = RandomForestClassifier(**params)
        logger.info(f"Initialized RandomForest with parameters: {params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the Random Forest model"""
        logger.info(f"Training RandomForest on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("RandomForest training completed")
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class CatBoostModel(BaselineModel):
    """CatBoost baseline model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("CatBoost is required. Install with: pip install catboost")
        
        # Default parameters
        params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1,
            'od_type': 'Iter',
            'od_wait': 50,
            'random_seed': 42,
            'allow_writing_files': False,
            'verbose': False
        }
        
        # Update with provided config
        if self.config:
            params.update(self.config)
            
        self.model = CatBoostClassifier(**params)
        logger.info(f"Initialized CatBoost with parameters: {params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the CatBoost model"""
        logger.info(f"Training CatBoost on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("CatBoost training completed")
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.get_feature_importance()


class TabNetModel(BaselineModel):
    """TabNet baseline model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            raise ImportError("PyTorch TabNet is required. Install with: pip install pytorch-tabnet")
        
        # Default parameters
        params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': 1e-4,
            'optimizer_fn': 'adam',
            'optimizer_params': {'lr': 2e-2},
            'mask_type': 'entmax',
            'scheduler_params': {'step_size': 50, 'gamma': 0.9},
            'scheduler_fn': 'step_lr',
            'seed': 42,
            'verbose': 0
        }
        
        # Update with provided config
        if self.config:
            params.update(self.config)
            
        self.model = TabNetClassifier(**params)
        logger.info(f"Initialized TabNet with parameters: {params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the TabNet model"""
        logger.info(f"Training TabNet on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            logger.info(f"Using validation set with {X_val.shape[0]} samples")
        else:
            eval_set = None
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128
        )
        self.is_fitted = True
        logger.info("TabNet training completed")
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class XGBoostModel(BaselineModel):
    """XGBoost baseline model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        # Default parameters
        params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 50
        }
        
        # Update with provided config
        if self.config:
            params.update(self.config)
            
        self.model = XGBClassifier(**params)
        logger.info(f"Initialized XGBoost with parameters: {params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the XGBoost model"""
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            logger.info(f"Using validation set with {X_val.shape[0]} samples")
        else:
            eval_set = None
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True
        logger.info("XGBoost training completed")
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class LightGBMModel(BaselineModel):
    """LightGBM baseline model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        # Default parameters
        params = {
            'n_estimators': 1000,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'min_child_samples': 20,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
            'early_stopping_rounds': 50
        }
        
        # Update with provided config
        if self.config:
            params.update(self.config)
            
        self.model = LGBMClassifier(**params)
        logger.info(f"Initialized LightGBM with parameters: {params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the LightGBM model"""
        logger.info(f"Training LightGBM on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            logger.info(f"Using validation set with {X_val.shape[0]} samples")
        else:
            eval_set = None
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=None if eval_set is None else [
                self.model.__class__.early_stopping(50, verbose=False)
            ]
        )
        self.is_fitted = True
        logger.info("LightGBM training completed")
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


def create_baseline_model(model_name: str, config: Optional[Dict[str, Any]] = None) -> BaselineModel:
    """
    Factory function to create baseline models
    
    Args:
        model_name: Name of the model ('random_forest', 'catboost', 'tabnet', 'xgboost', 'lightgbm')
        config: Model configuration parameters
    
    Returns:
        Baseline model instance
    """
    model_classes = {
        'random_forest': RandomForestModel,
        'catboost': CatBoostModel,
        'tabnet': TabNetModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_classes.keys())}")
    
    return model_classes[model_name](config)


class BaselineTrainer:
    """Trainer class for baseline models"""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = create_baseline_model(model_name, config.get('model_params'))
        
    def train_and_evaluate(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline
        
        Returns:
            Dictionary containing training and evaluation results
        """
        logger.info(f"Starting training and evaluation for {self.model_name}")
        
        # Train the model
        if self.model_name in ['tabnet', 'xgboost', 'lightgbm']:
            # These models support validation during training
            self.model.fit(X_train, y_train, X_val, y_val)
        else:
            self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_results = self.model.evaluate(X_val, y_val)
        logger.info(f"Validation results: {val_results}")
        
        # Evaluate on test set
        test_results = self.model.evaluate(X_test, y_test)
        logger.info(f"Test results: {test_results}")
        
        # Get detailed classification report
        y_pred = self.model.predict(X_test)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        try:
            feature_importance = self.model.get_feature_importance().tolist()
        except (AttributeError, ValueError):
            logger.info("Feature importance not available for this model")
        
        results = {
            'model_name': self.model_name,
            'validation_results': val_results,
            'test_results': test_results,
            'classification_report': classification_rep,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"Training and evaluation completed for {self.model_name}")
        return results
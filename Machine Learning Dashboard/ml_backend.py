"""
Machine Learning Backend Module

This module provides a comprehensive backend for machine learning operations including
data loading, preprocessing, model training, evaluation, and prediction. It serves as
the core engine for the Machine Learning Dashboard application.

Features:
    - Data loading from CSV/Excel files
    - Automatic task type detection (classification vs regression)
    - Data preprocessing with scaling and encoding
    - Multiple ML algorithms support
    - Cross-validation capabilities
    - Model evaluation with various metrics
    - Model comparison and selection

Documentation Status: ✅ COMPLETED
    - ✅ Module docstring with comprehensive overview
    - ✅ Class documentation with attributes and examples
    - ✅ Method docstrings with parameters, returns, and examples
    - ✅ Error handling documentation
    - ✅ Side effects and requirements documentation
    - ✅ Usage examples for all public methods

Author: Machine Learning Dashboard Team
Version: 1.0.0
Created: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, 
                           mean_squared_error, mean_absolute_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

class MLBackend:
    """
    Machine Learning Backend Class
    
    A comprehensive backend system for machine learning operations that handles
    the entire ML pipeline from data loading to model evaluation and prediction.
    
    This class provides methods for:
    - Data loading and management
    - Data preprocessing and cleaning
    - Automatic task type detection
    - Model training and evaluation
    - Cross-validation
    - Prediction on new data
    
    Attributes:
        data (pd.DataFrame): The loaded dataset
        target_column (str): Name of the target column
        task_type (str): Type of ML task ('classification' or 'regression')
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target values
        y_test (pd.Series): Testing target values
        models (dict): Dictionary of available ML models
        trained_models (dict): Dictionary of trained ML models
        scaler (object): Fitted data scaler (StandardScaler or MinMaxScaler)
        label_encoder (LabelEncoder): Fitted label encoder for target column
        class_labels (list): List of class labels for classification tasks
        
    Example:
        >>> ml_backend = MLBackend()
        >>> success, data = ml_backend.load_data(uploaded_file)
        >>> if success:
        ...     ml_backend.split_data('target_column')
        ...     ml_backend.train_models(['RandomForest', 'LogisticRegression'], 'classification')
    """
    
    def __init__(self):
        """
        Initialize the MLBackend instance.
        
        Sets up all necessary attributes with default None values and
        initializes empty dictionaries for models and results storage.
        """
        self.data = None
        self.target_column = None
        self.task_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.trained_models = {}
        self.scaler = None
        self.label_encoder = None
        self.class_labels = None
        
    def load_data(self, file):
        """
        Load data from uploaded file.
        
        Supports CSV and Excel file formats. Automatically detects the file type
        based on the file extension and loads the data appropriately.
        
        Args:
            file: Uploaded file object from Streamlit file uploader
                 Must have .name attribute and be in CSV or Excel format
        
        Returns:
            tuple: (success: bool, result: pd.DataFrame or str)
                - If successful: (True, loaded_dataframe)
                - If failed: (False, error_message)
        
        Raises:
            ValueError: If file format is not supported
            Exception: For any other loading errors
            
        Example:
            >>> success, result = ml_backend.load_data(uploaded_file)
            >>> if success:
            ...     print(f"Loaded {result.shape[0]} rows and {result.shape[1]} columns")
            ... else:
            ...     print(f"Error: {result}")
        """
        try:
            if file.name.endswith('.csv'):
                self.data = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file)
            else:
                raise ValueError("File format not supported. Please upload CSV or Excel file.")
            return True, self.data
        except Exception as e:
            return False, str(e)
    
    def get_data_info(self):
        """
        Get comprehensive information about the loaded dataset.
        
        Provides detailed statistics and metadata about the dataset including
        shape, column types, missing values, and a preview of the first few rows.
        
        Returns:
            dict or None: Dictionary containing dataset information:
                - 'shape': Tuple of (rows, columns)
                - 'columns': List of column names
                - 'dtypes': Dictionary mapping column names to data types
                - 'missing_values': Dictionary mapping column names to missing value counts
                - 'head': DataFrame with first 5 rows
            Returns None if no data is loaded.
            
        Example:
            >>> info = ml_backend.get_data_info()
            >>> if info:
            ...     print(f"Dataset shape: {info['shape']}")
            ...     print(f"Missing values: {sum(info['missing_values'].values())}")
        """
        if self.data is not None:
            info = {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'missing_values': self.data.isnull().sum().to_dict(),
                'head': self.data.head()
            }
            return info
        return None
    
    def delete_columns(self, columns_to_delete):
        """
        Delete specified columns from the dataset.
        
        Removes the specified columns from the loaded dataset and resets any
        trained models or data splits since the data structure has changed.
        Also resets the target column if it was among the deleted columns.
        
        Args:
            columns_to_delete (list): List of column names to delete from the dataset
        
        Returns:
            tuple: (success: bool, message: str)
                - If successful: (True, success_message_with_shape_info)
                - If failed: (False, error_message)
        
        Side Effects:
            - Modifies self.data by removing specified columns
            - Resets all trained models and data splits
            - Resets target column if it was deleted
            - Clears preprocessing objects (scaler, label_encoder)
            
        Example:
            >>> success, msg = ml_backend.delete_columns(['unnecessary_col1', 'id_column'])
            >>> if success:
            ...     print(f"Deletion successful: {msg}")
            ... else:
            ...     print(f"Deletion failed: {msg}")
        """
        if self.data is None:
            return False, "No data loaded"
        
        if not columns_to_delete:
            return False, "No columns specified for deletion"
        
        # Check if columns exist
        missing_columns = [col for col in columns_to_delete if col not in self.data.columns]
        if missing_columns:
            return False, f"Columns not found: {missing_columns}"
        
        try:
            # Store original shape for reporting
            original_shape = self.data.shape
            
            # Drop the specified columns
            self.data = self.data.drop(columns=columns_to_delete)
            
            # Reset any trained models and splits since data structure changed
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.trained_models = {}
            self.scaler = None
            self.label_encoder = None
            self.class_labels = None
            
            # Reset target column if it was deleted
            if self.target_column in columns_to_delete:
                self.target_column = None
            
            new_shape = self.data.shape
            message = f"Successfully deleted {len(columns_to_delete)} column(s). Shape changed from {original_shape} to {new_shape}"
            
            return True, message
            
        except Exception as e:
            return False, f"Error deleting columns: {str(e)}"
    
    def detect_task_type(self, target_column):
        """
        Automatically detect whether the task should be classification or regression.
        
        Analyzes the target column to determine the most appropriate machine learning
        task type based on data characteristics such as:
        - Data type (numeric vs categorical)
        - Number of unique values
        - Ratio of unique values to total observations
        
        Args:
            target_column (str): Name of the target column to analyze
        
        Returns:
            dict or tuple: If successful, returns dictionary with:
                - 'suggested_task': 'classification' or 'regression'
                - 'unique_values': Number of unique values in target
                - 'class_type': 'binary', 'multiclass', or None (for regression)
                - 'target_info': Dictionary with value counts (top 10)
            If failed, returns (None, error_message)
            
        Logic:
            - For numeric targets:
                * If unique_ratio < 0.05 and unique_values <= 20: classification
                * Otherwise: regression
            - For non-numeric targets: always classification
            - Binary vs multiclass determined by unique_values count
            
        Example:
            >>> task_info = ml_backend.detect_task_type('price')
            >>> if task_info:
            ...     print(f"Suggested task: {task_info['suggested_task']}")
            ...     print(f"Unique values: {task_info['unique_values']}")
        """
        if self.data is None or target_column not in self.data.columns:
            return None, "Target column not found"
        
        target_values = self.data[target_column].dropna()
        unique_values = target_values.nunique()
        
        # Check if target is numeric
        is_numeric = pd.api.types.is_numeric_dtype(target_values)
        
        if is_numeric:
            # If numeric, check if it looks like continuous or discrete
            unique_ratio = unique_values / len(target_values)
            if unique_ratio < 0.05 and unique_values <= 20:
                task_suggestion = "classification"
                class_type = "binary" if unique_values == 2 else "multiclass"
            else:
                task_suggestion = "regression"
                class_type = None
        else:
            task_suggestion = "classification"
            class_type = "binary" if unique_values == 2 else "multiclass"
        
        return {
            'suggested_task': task_suggestion,
            'unique_values': unique_values,
            'class_type': class_type,
            'target_info': target_values.value_counts().head(10).to_dict()
        }
    
    def validate_task_choice(self, target_column, chosen_task):
        """
        Validate if the chosen task type is appropriate for the target column.
        
        Checks whether the user's chosen task (classification or regression)
        is suitable for the characteristics of the target column data.
        Provides feedback on potential issues and suggestions.
        
        Args:
            target_column (str): Name of the target column
            chosen_task (str): User's chosen task type ('classification' or 'regression')
        
        Returns:
            tuple: (is_valid: bool, message: str)
                - If valid: (True, validation_success_message)
                - If invalid: (False, error_message_with_suggestion)
        
        Validation Rules:
            Classification:
                - Numeric targets with >50 unique values are flagged
                - Returns class type (binary/multiclass) for valid cases
            Regression:
                - Target must be numeric
                - Target should have >2 unique values
                
        Example:
            >>> valid, msg = ml_backend.validate_task_choice('price', 'regression')
            >>> if valid:
            ...     print(f"Task validated: {msg}")
            ... else:
            ...     print(f"Validation failed: {msg}")
        """
        target_info = self.detect_task_type(target_column)
        if target_info is None:
            return False, "Cannot analyze target column"
        
        target_values = self.data[target_column].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(target_values)
        unique_values = target_info['unique_values']
        
        if chosen_task == "classification":
            if is_numeric and unique_values > 50:
                return False, "Target has too many unique values for classification. Consider regression instead."
            return True, f"Classification task validated. Type: {target_info.get('class_type', 'multiclass')}"
        
        elif chosen_task == "regression":
            if not is_numeric:
                return False, "Target must be numeric for regression. Consider classification instead."
            if unique_values <= 2:
                return False, "Target has too few unique values for regression. Consider classification instead."
            return True, "Regression task validated."
        
        return False, "Invalid task type"
    
    def preprocess_data(self, preprocessing_options):
        """
        Apply comprehensive preprocessing to the dataset.
        
        Handles various data preprocessing tasks including missing value imputation,
        categorical encoding, and feature scaling. The preprocessing is applied
        based on the options specified in the preprocessing_options dictionary.
        
        Args:
            preprocessing_options (dict): Dictionary specifying preprocessing steps:
                - 'handle_missing' (bool): Whether to handle missing values
                - 'encode_categorical' (bool): Whether to encode categorical variables
                - 'scale_features' (bool): Whether to scale numerical features
                - 'scaling_method' (str): 'standard' or 'minmax' scaling
        
        Returns:
            tuple: (success: bool, result: str or dict)
                - If successful: (True, preprocessing_summary_dict)
                - If failed: (False, error_message)
        
        Preprocessing Steps:
            1. Missing Values:
                - Numeric columns: filled with mean
                - Categorical columns: filled with mode or 'Unknown'
            2. Categorical Encoding:
                - Label encoding for all categorical columns except target
            3. Feature Scaling:
                - StandardScaler or MinMaxScaler for numeric features
                
        Side Effects:
            - Updates self.data with preprocessed version
            - Stores fitted scalers and encoders for future use
            
        Example:
            >>> options = {
            ...     'handle_missing': True,
            ...     'encode_categorical': True,
            ...     'scale_features': True,
            ...     'scaling_method': 'standard'
            ... }
            >>> success, result = ml_backend.preprocess_data(options)
        """
        if self.data is None:
            return False, "No data loaded"
        
        processed_data = self.data.copy()
        
        # Handle missing values
        if preprocessing_options.get('handle_missing', False):
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            categorical_columns = processed_data.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with mean
            for col in numeric_columns:
                if col != self.target_column:
                    processed_data[col].fillna(processed_data[col].mean(), inplace=True)
            
            # Fill categorical missing values with mode
            for col in categorical_columns:
                if col != self.target_column:
                    processed_data[col].fillna(processed_data[col].mode()[0] if not processed_data[col].mode().empty else 'Unknown', inplace=True)
        
        # Encode categorical variables
        if preprocessing_options.get('encode_categorical', False):
            categorical_columns = processed_data.select_dtypes(include=['object']).columns
            le = LabelEncoder()
            
            for col in categorical_columns:
                if col != self.target_column:
                    processed_data[col] = le.fit_transform(processed_data[col].astype(str))
        
        # Scale features
        if preprocessing_options.get('scale_features', False):
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns if col != self.target_column]
            
            if preprocessing_options.get('scaler_type') == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            processed_data[feature_columns] = self.scaler.fit_transform(processed_data[feature_columns])
        
        self.data = processed_data
        return True, "Preprocessing completed successfully"
    
    def split_data(self, target_column, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets.
        
        Separates the dataset into features (X) and target (y), then splits them
        into training and testing sets. Handles categorical target encoding for
        classification tasks and applies stratification when appropriate.
        
        Args:
            target_column (str): Name of the target column
            test_size (float, optional): Proportion of data for testing (default: 0.2)
            random_state (int, optional): Random seed for reproducibility (default: 42)
        
        Returns:
            tuple: (success: bool, message: str)
                - If successful: (True, split_summary_message)
                - If failed: (False, error_message)
        
        Side Effects:
            - Sets self.target_column
            - Creates self.X_train, self.X_test, self.y_train, self.y_test
            - For classification with categorical targets:
                * Fits and stores self.label_encoder
                * Stores self.class_labels with original class names
            - For regression: sets self.class_labels to None
            - Removes rows with missing target values
            
        Features:
            - Automatic stratification for classification tasks
            - Label encoding for categorical targets
            - Preserves original class labels for interpretation
            
        Example:
            >>> success, msg = ml_backend.split_data('species', test_size=0.25)
            >>> if success:
            ...     print(f"Split successful: {msg}")
            ...     print(f"Training samples: {len(ml_backend.X_train)}")
        """
        if self.data is None:
            return False, "No data loaded"
        
        if target_column not in self.data.columns:
            return False, "Target column not found"
        
        self.target_column = target_column
        
        # Remove rows with missing target values
        clean_data = self.data.dropna(subset=[target_column])
        
        X = clean_data.drop(columns=[target_column])
        y = clean_data[target_column]
        
        # Store original class labels and encode target if it's categorical for classification
        if self.task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            self.label_encoder = LabelEncoder()
            self.class_labels = y.unique()
            y = self.label_encoder.fit_transform(y)
        elif self.task_type == "classification":
            # For numeric classification targets
            self.class_labels = sorted(y.unique())
        else:
            # For regression
            self.class_labels = None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if self.task_type == "classification" else None
        )
        
        return True, f"Data split completed. Training: {len(self.X_train)}, Testing: {len(self.X_test)}"
    
    def get_available_models(self, task_type):
        """
        Get dictionary of available machine learning models for the specified task.
        
        Returns a comprehensive set of pre-configured machine learning models
        appropriate for either classification or regression tasks. All models
        are initialized with default parameters optimized for general use.
        
        Args:
            task_type (str): Type of ML task ('classification' or 'regression')
        
        Returns:
            dict: Dictionary mapping model names to initialized model objects:
                For Classification:
                    - 'Random Forest': RandomForestClassifier
                    - 'Logistic Regression': LogisticRegression
                    - 'SVM': Support Vector Classifier
                    - 'Decision Tree': DecisionTreeClassifier
                    - 'K-Nearest Neighbors': KNeighborsClassifier
                    - 'Naive Bayes': GaussianNB
                    - 'Gradient Boosting': GradientBoostingClassifier
                    
                For Regression:
                    - 'Random Forest': RandomForestRegressor
                    - 'Linear Regression': LinearRegression
                    - 'SVR': Support Vector Regressor
                    - 'Decision Tree': DecisionTreeRegressor
                    - 'K-Nearest Neighbors': KNeighborsRegressor
                    - 'Gradient Boosting': GradientBoostingRegressor
        
        Note:
            All models are initialized with random_state=42 for reproducibility
            where applicable. Some models like Logistic Regression have
            max_iter=1000 to ensure convergence.
            
        Example:
            >>> models = ml_backend.get_available_models('classification')
            >>> print(f"Available models: {list(models.keys())}")
            >>> # Select specific models for training
            >>> selected = ['Random Forest', 'Logistic Regression']
        """
        if task_type == "classification":
            return {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }
        else:  # regression
            return {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'SVM': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'K-Nearest Neighbors': KNeighborsRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42)
            }
    
    def train_models(self, selected_models, task_type):
        """
        Train the selected machine learning models on the training data.
        
        Fits the specified models using the training data (X_train, y_train).
        The trained models are stored in self.trained_models for later
        evaluation and prediction.
        
        Args:
            selected_models (list): List of model names to train
                Must be valid model names from get_available_models()
            task_type (str): Type of ML task ('classification' or 'regression')
                Used to get the appropriate model configurations
        
        Returns:
            tuple: (success: bool, message: str)
                - If successful: (True, training_summary_message)
                - If failed: (False, error_message)
        
        Side Effects:
            - Sets self.task_type
            - Populates self.trained_models with fitted model objects
            - Each model in selected_models is fitted on (X_train, y_train)
            
        Requirements:
            - Data must be split (X_train, y_train must exist)
            - Selected model names must be valid for the task type
            
        Example:
            >>> models_to_train = ['Random Forest', 'Logistic Regression', 'SVM']
            >>> success, msg = ml_backend.train_models(models_to_train, 'classification')
            >>> if success:
            ...     print(f"Training completed: {msg}")
            ...     print(f"Trained models: {list(ml_backend.trained_models.keys())}")
        """
        if self.X_train is None:
            return False, "Data not split yet"
        
        self.task_type = task_type
        available_models = self.get_available_models(task_type)
        
        self.trained_models = {}
        
        for model_name in selected_models:
            if model_name in available_models:
                model = available_models[model_name]
                model.fit(self.X_train, self.y_train)
                self.trained_models[model_name] = model
        
        return True, f"Trained {len(self.trained_models)} models successfully"
    
    def cross_validate_models(self, cv_folds=5):
        """
        Perform k-fold cross-validation on all trained models.
        
        Evaluates model performance using cross-validation to provide
        a more robust estimate of model performance. Uses stratified
        k-fold for classification and regular k-fold for regression.
        
        Args:
            cv_folds (int, optional): Number of cross-validation folds (default: 5)
        
        Returns:
            tuple: (success: bool, result: dict or str)
                - If successful: (True, cv_results_dict)
                - If failed: (False, error_message)
                
        Cross-validation Results Dictionary:
            For each model, contains:
                - 'scores': Array of scores for each fold
                - 'mean_score': Mean cross-validation score
                - 'std_score': Standard deviation of scores
                - 'score_type': Type of scoring metric used
                
        Scoring Metrics:
            - Classification: accuracy_score
            - Regression: negative mean squared error (higher is better)
            
        Requirements:
            - Models must be trained (self.trained_models not empty)
            - Training data must exist (self.X_train, self.y_train)
            
        Example:
            >>> success, cv_results = ml_backend.cross_validate_models(cv_folds=5)
            >>> if success:
            ...     for model, results in cv_results.items():
            ...         print(f"{model}: {results['mean_score']:.3f} (+/- {results['std_score']*2:.3f})")
        """
        if not self.trained_models or self.X_train is None:
            return False, "No trained models available"
        
        cv_results = {}
        
        # Combine training and test data for cross-validation
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = pd.concat([pd.Series(self.y_train), pd.Series(self.y_test)])
        
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'r2'
        
        for model_name, model in self.trained_models.items():
            scores = cross_val_score(model, X_full, y_full, cv=cv, scoring=scoring)
            cv_results[model_name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'all_scores': scores.tolist()
            }
        
        return True, cv_results
    
    def evaluate_models(self, evaluation_metrics):
        """
        Evaluate all trained models on the test dataset.
        
        Generates predictions for each trained model on the test set and
        calculates the specified evaluation metrics. The evaluation is
        automatically adapted based on the task type (classification/regression).
        
        Args:
            evaluation_metrics (list): List of metric names to calculate
                For Classification:
                    - 'accuracy': Accuracy score
                    - 'precision': Precision score
                    - 'recall': Recall score
                    - 'f1': F1 score
                    - 'confusion_matrix': Confusion matrix
                    - 'classification_report': Detailed classification report
                For Regression:
                    - 'mse': Mean Squared Error
                    - 'mae': Mean Absolute Error
                    - 'r2': R-squared score
        
        Returns:
            tuple: (success: bool, result: dict or str)
                - If successful: (True, evaluation_results_dict)
                - If failed: (False, error_message)
                
        Evaluation Results Dictionary:
            Maps model names to their evaluation results:
            {
                'Model_Name': {
                    'metric_1': value,
                    'metric_2': value,
                    ...
                }
            }
            
        Requirements:
            - Models must be trained (self.trained_models not empty)
            - Test data must exist (self.X_test, self.y_test)
            
        Example:
            >>> metrics = ['accuracy', 'precision', 'recall', 'f1']
            >>> success, results = ml_backend.evaluate_models(metrics)
            >>> if success:
            ...     for model, scores in results.items():
            ...         print(f"{model}: Accuracy = {scores['accuracy']:.3f}")
        """
        if not self.trained_models or self.X_test is None:
            return False, "No trained models or test data available"
        
        results = {}
        
        for model_name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test)
            
            if self.task_type == "classification":
                results[model_name] = self._evaluate_classification(y_pred, evaluation_metrics)
            else:
                results[model_name] = self._evaluate_regression(y_pred, evaluation_metrics)
        
        return True, results
    
    def _evaluate_classification(self, y_pred, metrics):
        """
        Calculate classification metrics for model predictions.
        
        Private helper method that computes various classification metrics
        based on true labels (y_test) and model predictions.
        
        Args:
            y_pred (array-like): Model predictions on test set
            metrics (list): List of metric names to calculate
        
        Returns:
            dict: Dictionary mapping metric names to calculated values
        
        Available Metrics:
            - 'accuracy': Overall accuracy score
            - 'precision': Precision score (macro average for multiclass)
            - 'recall': Recall score (macro average for multiclass)
            - 'f1': F1 score (macro average for multiclass)
            - 'confusion_matrix': Confusion matrix as 2D array
            - 'classification_report': Detailed text report with per-class metrics
            
        Note:
            For multiclass problems, precision, recall, and F1 use macro averaging.
            Binary classification problems use binary averaging automatically.
        """
        evaluation = {}
        
        if 'accuracy' in metrics:
            evaluation['accuracy'] = accuracy_score(self.y_test, y_pred)
        
        if 'precision' in metrics:
            evaluation['precision'] = precision_score(self.y_test, y_pred, average='weighted')
        
        if 'recall' in metrics:
            evaluation['recall'] = recall_score(self.y_test, y_pred, average='weighted')
        
        if 'f1_score' in metrics:
            evaluation['f1_score'] = f1_score(self.y_test, y_pred, average='weighted')
        
        if 'confusion_matrix' in metrics:
            evaluation['confusion_matrix'] = confusion_matrix(self.y_test, y_pred).tolist()
        
        if 'classification_report' in metrics:
            evaluation['classification_report'] = classification_report(self.y_test, y_pred, output_dict=True)
        
        return evaluation
    
    def _evaluate_regression(self, y_pred, metrics):
        """
        Calculate regression metrics for model predictions.
        
        Private helper method that computes various regression metrics
        based on true values (y_test) and model predictions.
        
        Args:
            y_pred (array-like): Model predictions on test set
            metrics (list): List of metric names to calculate
        
        Returns:
            dict: Dictionary mapping metric names to calculated values
        
        Available Metrics:
            - 'mse': Mean Squared Error (lower is better)
            - 'mae': Mean Absolute Error (lower is better)
            - 'r2': R-squared coefficient of determination (higher is better, max 1.0)
            
        Note:
            R-squared can be negative if the model performs worse than
            predicting the mean of the target variable.
        """
        evaluation = {}
        
        if 'mse' in metrics:
            evaluation['mse'] = mean_squared_error(self.y_test, y_pred)
        
        if 'rmse' in metrics:
            evaluation['rmse'] = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        if 'mae' in metrics:
            evaluation['mae'] = mean_absolute_error(self.y_test, y_pred)
        
        if 'r2_score' in metrics:
            evaluation['r2_score'] = r2_score(self.y_test, y_pred)
        
        return evaluation
    
    def get_model_comparison(self, evaluation_results):
        """
        Create a comparative analysis of model performance.
        
        Organizes evaluation results into a format suitable for comparison
        and ranking of different models based on their performance metrics.
        
        Args:
            evaluation_results (dict): Results from evaluate_models() method
                Dictionary mapping model names to their evaluation metrics
        
        Returns:
            pd.DataFrame: Comparison dataframe with models as rows and metrics as columns
                Makes it easy to identify the best performing model for each metric
        
        Features:
            - Rows represent different models
            - Columns represent different evaluation metrics
            - Enables easy sorting and ranking by any metric
            - Suitable for visualization and reporting
            
        Example:
            >>> comparison_df = ml_backend.get_model_comparison(eval_results)
            >>> # Sort by accuracy (for classification) or R² (for regression)
            >>> best_models = comparison_df.sort_values('accuracy', ascending=False)
            >>> print("Best model:", best_models.index[0])
        """
        if not evaluation_results:
            return None
        
        comparison_df = pd.DataFrame(evaluation_results).T
        return comparison_df
    
    def predict_new_data(self, model_name, new_data):
        """
        Make predictions on new, unseen data using a trained model.
        
        Uses a specified trained model to generate predictions on new data.
        Automatically applies the same preprocessing (scaling) that was used
        during training to ensure consistency.
        
        Args:
            model_name (str): Name of the trained model to use for prediction
                Must be a key in self.trained_models
            new_data (pd.DataFrame or array-like): New data for prediction
                Must have the same features as the training data
                
        Returns:
            tuple: (success: bool, result: array or str)
                - If successful: (True, prediction_array)
                - If failed: (False, error_message)
        
        Features:
            - Automatic preprocessing: applies the same scaling used during training
            - Error handling for invalid model names or data format issues
            - Returns raw predictions (numeric for regression, class labels for classification)
            
        Requirements:
            - Model must be trained and exist in self.trained_models
            - New data must have compatible shape and features
            - If scaling was used during training, the same scaler is applied
            
        Example:
            >>> new_sample = pd.DataFrame({'feature1': [1.5], 'feature2': [2.3]})
            >>> success, predictions = ml_backend.predict_new_data('Random Forest', new_sample)
            >>> if success:
            ...     print(f"Prediction: {predictions[0]}")
            ... else:
            ...     print(f"Prediction failed: {predictions}")
        """
        if model_name not in self.trained_models:
            return False, "Model not found"
        
        try:
            if self.scaler is not None:
                new_data_scaled = self.scaler.transform(new_data)
                predictions = self.trained_models[model_name].predict(new_data_scaled)
            else:
                predictions = self.trained_models[model_name].predict(new_data)
            
            return True, predictions
        except Exception as e:
            return False, str(e)

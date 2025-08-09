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
    def __init__(self):
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
        """Load data from uploaded file"""
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
        """Get basic information about the data"""
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
        """Delete specified columns from the dataset"""
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
        """Detect if the task should be classification or regression"""
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
        """Validate if chosen task is appropriate for the target"""
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
        """Apply preprocessing to the data"""
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
        """Split data into training and testing sets"""
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
        """Get available models for the specified task"""
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
        """Train selected models"""
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
        """Perform cross-validation on trained models"""
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
        """Evaluate trained models on test set"""
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
        """Evaluate classification model"""
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
        """Evaluate regression model"""
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
        """Create comparison between models"""
        if not evaluation_results:
            return None
        
        comparison_df = pd.DataFrame(evaluation_results).T
        return comparison_df
    
    def predict_new_data(self, model_name, new_data):
        """Make predictions on new data"""
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

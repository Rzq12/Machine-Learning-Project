"""
Machine Learning Dashboard - Main Streamlit Application

This module contains the main Streamlit web application for the Machine Learning Dashboard.
It provides an interactive interface for end-to-end machine learning workflows including
data upload, preprocessing, model training, evaluation, and results visualization.

The application guides users through a step-by-step process:
1. Upload Data - Load CSV/Excel datasets
2. Delete Columns - Remove unnecessary columns
3. Task Selection - Choose classification or regression
4. Target Column - Select the target variable
5. Preprocessing - Handle missing values, encoding, scaling
6. Data Split - Split into training and testing sets
7. Cross Validation - Perform k-fold cross-validation
8. Model Selection - Choose and train ML models
9. Evaluation - Assess model performance
10. Results - View comprehensive results and comparisons

Features:
    - Interactive data exploration and visualization
    - Automatic task type detection
    - Multiple preprocessing options
    - Support for various ML algorithms
    - Comprehensive model evaluation
    - Interactive plots and charts
    - Model performance comparison
    - Download results functionality

Architecture:
    - Frontend: Streamlit with responsive layout
    - Backend: Custom MLBackend class for ML operations
    - State Management: Streamlit session state for workflow persistence
    - Visualization: Plotly for interactive charts, Matplotlib/Seaborn for static plots
    - ML Algorithms: scikit-learn integration

Dependencies:
    - streamlit: Web application framework
    - pandas: Data manipulation and analysis
    - plotly: Interactive visualizations
    - seaborn/matplotlib: Statistical plotting
    - ml_backend: Custom ML backend module

Documentation Status: ✅ COMPLETED
    - ✅ Module docstring with comprehensive overview
    - ✅ Section headers for all workflow steps
    - ✅ Inline comments for UI components
    - ✅ User guidance and help text
    - ✅ Error handling documentation
    - ✅ Feature descriptions and usage examples

Author: Riezqi
Version: 1.0.0
Created: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from ml_backend import MLBackend

# ================================
# PAGE CONFIGURATION AND SETUP
# ================================

# Configure Streamlit page settings for optimal user experience
st.set_page_config(
    page_title="Machine Learning Dashboard",  # Browser tab title
    page_icon="🤖",                          # Browser tab icon
    layout="wide",                           # Use full width of browser
    initial_sidebar_state="expanded"          # Start with sidebar open
)

# ================================
# SESSION STATE INITIALIZATION
# ================================

# Initialize session state variables to maintain application state across reruns
# This ensures data persists when users interact with the interface

if 'ml_backend' not in st.session_state:
    """Initialize the ML backend instance for handling all ML operations"""
    st.session_state.ml_backend = MLBackend()

if 'data_loaded' not in st.session_state:
    """Track whether dataset has been successfully loaded"""
    st.session_state.data_loaded = False

if 'target_selected' not in st.session_state:
    """Track whether target column has been selected"""
    st.session_state.target_selected = False

if 'models_trained' not in st.session_state:
    """Track whether ML models have been trained"""
    st.session_state.models_trained = False

# ================================
# MAIN INTERFACE LAYOUT
# ================================

# Main title and branding
st.title("🤖 Machine Learning Dashboard")
st.markdown("---")

# ================================
# NAVIGATION SIDEBAR
# ================================

# Create sidebar navigation for step-by-step workflow
st.sidebar.title("📋 Navigation")

# Define the complete ML workflow steps
steps = [
    "1. Upload Data",           # Load dataset from file
    "2. Delete Columns",        # Remove unnecessary columns
    "3. Task Selection",        # Choose classification vs regression
    "4. Target Column",         # Select target variable
    "5. Preprocessing",         # Data cleaning and preparation
    "6. Data Split",           # Train/test split
    "7. Cross Validation",     # Model validation
    "8. Model Selection",      # Choose and train models
    "9. Evaluation",          # Model performance assessment
    "10. Results"             # Final results and comparison
]

# Radio button for step selection - user can navigate between steps
current_step = st.sidebar.radio("Select Step:", steps)

# ================================
# STEP 1: DATA UPLOAD AND EXPLORATION
# ================================

if current_step == "1. Upload Data":
    st.header("📁 Upload Your Dataset")
    
    # File uploader widget with support for CSV and Excel formats
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],  # Supported file formats
        help="Upload your dataset in CSV or Excel format"
    )
    
    if uploaded_file is not None:
        # Show loading spinner while processing the file
        with st.spinner("Loading data..."):
            success, result = st.session_state.ml_backend.load_data(uploaded_file)
        
        if success:
            # ---- SUCCESS: Display data information ----
            st.success("✅ Data loaded successfully!")
            st.session_state.data_loaded = True
            
            # Get comprehensive data information
            data_info = st.session_state.ml_backend.get_data_info()
            
            # ---- METRICS DISPLAY ----
            # Show key dataset statistics in a clean layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", data_info['shape'][0])
                st.metric("Columns", data_info['shape'][1])
            
            with col2:
                missing_count = sum(data_info['missing_values'].values())
                st.metric("Missing Values", missing_count)
                # Calculate and display memory usage
                memory_mb = result.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory Usage", f"{memory_mb:.2f} MB")
            
            # ---- DATA PREVIEW ----
            # Show first few rows of the dataset
            st.subheader("📊 Data Preview")
            st.dataframe(data_info['head'], use_container_width=True)
            
            # ---- COLUMN INFORMATION ----
            # Display detailed information about each column
            st.subheader("🔢 Column Information")
            dtype_df = pd.DataFrame({
                'Column': list(data_info['dtypes'].keys()),
                'Data Type': list(data_info['dtypes'].values()),
                'Missing Values': [data_info['missing_values'][col] for col in data_info['dtypes'].keys()]
            })
            st.dataframe(dtype_df, use_container_width=True)
            
        else:
            # ---- ERROR HANDLING ----
            st.error(f"❌ Error loading data: {result}")

# ================================
# STEP 2: COLUMN DELETION (OPTIONAL)
# ================================

elif current_step == "2. Delete Columns":
    st.header("🗑️ Delete Columns (Optional)")
    
    # Check if data has been loaded
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        # Get current dataset information
        data_info = st.session_state.ml_backend.get_data_info()
        
        # Informational message about column deletion benefits
        st.info("💡 You can delete unnecessary columns to improve model performance and reduce complexity.")
        
        # ---- CURRENT COLUMNS DISPLAY ----
        st.subheader("📋 Current Columns")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create comprehensive column information table
            column_info = pd.DataFrame({
                'Column': list(data_info['dtypes'].keys()),
                'Data Type': list(data_info['dtypes'].values()),
                'Missing Values': [data_info['missing_values'][col] for col in data_info['dtypes'].keys()],
                'Sample Values': [
                    str(data_info['head'][col].iloc[0]) if not pd.isna(data_info['head'][col].iloc[0]) else 'NaN' 
                    for col in data_info['dtypes'].keys()
                ]
            })
            st.dataframe(column_info, use_container_width=True)
        
        with col2:
            # Dataset summary metrics
            st.metric("Total Columns", len(data_info['columns']))
            st.metric("Total Rows", data_info['shape'][0])
        
        # ---- COLUMN SELECTION FOR DELETION ----
        st.subheader("🎯 Select Columns to Delete")
        
        # Multi-select widget for choosing columns to remove
        columns_to_delete = st.multiselect(
            "Choose columns to delete:",
            options=data_info['columns'],
            help="Select one or more columns that you want to remove from the dataset. "
                 "Consider removing ID columns, irrelevant features, or columns with too many missing values."
        )
        
        if columns_to_delete:
            st.warning(f"⚠️ You are about to delete {len(columns_to_delete)} column(s): {', '.join(columns_to_delete)}")
            
            # Show what will remain
            remaining_columns = [col for col in data_info['columns'] if col not in columns_to_delete]
            st.info(f"📊 After deletion, {len(remaining_columns)} columns will remain: {', '.join(remaining_columns)}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🗑️ Delete Selected Columns", type="primary"):
                    with st.spinner("Deleting columns..."):
                        success, message = st.session_state.ml_backend.delete_columns(columns_to_delete)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.columns_deleted = True
                        
                        # Show updated data preview
                        updated_info = st.session_state.ml_backend.get_data_info()
                        st.subheader("📊 Updated Data Preview")
                        st.dataframe(updated_info['head'], use_container_width=True)
                        
                        # Reset any subsequent processing flags
                        if 'target_selected' in st.session_state:
                            st.session_state.target_selected = False
                        if 'models_trained' in st.session_state:
                            st.session_state.models_trained = False
                        
                        st.info("ℹ️ Column deletion completed. Please proceed to the next steps to reconfigure your analysis.")
                    else:
                        st.error(f"❌ {message}")
            
            with col2:
                if st.button("🔄 Clear Selection"):
                    st.rerun()
        
        else:
            st.info("💡 No columns selected for deletion. You can skip this step if you want to keep all columns.")
            
        # Guidelines for column deletion
        with st.expander("📖 Guidelines for Column Deletion"):
            st.markdown("""
            **When to delete columns:**
            - **ID columns**: Usually not useful for prediction (e.g., customer_id, row_id)
            - **Duplicate information**: Columns that contain the same information
            - **Too many missing values**: Columns with >70% missing data
            - **Irrelevant features**: Columns not related to your prediction goal
            - **High cardinality categorical**: Text columns with too many unique values
            
            **Be careful with:**
            - **Target column**: Don't delete what you want to predict!
            - **Important features**: Domain knowledge is crucial
            - **Date/time columns**: Might need feature engineering instead of deletion
            """)

# ================================
# STEP 3: TASK TYPE SELECTION
# ================================
# Users choose between classification and regression based on their target variable
# The system provides intelligent guidance for making the right choice

# Step 3: Task Selection
elif current_step == "3. Task Selection":
    st.header("🎯 Select Machine Learning Task")
    
    # Check if data is loaded before proceeding
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        # ---- TASK TYPE SELECTION ----
        # Radio button for selecting between classification and regression
        task_type = st.radio(
            "Choose the type of machine learning task:",
            ["Classification", "Regression"],
            help="Classification: Predict categories/classes. Regression: Predict continuous numerical values."
        )
        
        # Store selected task type in session state
        st.session_state.task_type = task_type.lower()
        
        # Display confirmation of selection
        st.info(f"📝 Selected task: **{task_type}**")
        
        # ---- TASK TYPE GUIDANCE ----
        # Provide detailed information about each task type
        if task_type == "Classification":
            st.markdown("""
            **Classification** is used when:
            - Predicting categories or classes
            - Target variable has discrete values
            - Examples: Email spam detection, image recognition, medical diagnosis
            """)
        else:
            st.markdown("""
            **Regression** is used when:
            - Predicting continuous numerical values
            - Target variable is numeric
            - Examples: House price prediction, temperature forecasting, sales prediction
            """)

# ================================
# STEP 4: TARGET COLUMN SELECTION
# ================================
# Users select which column they want to predict and get automatic analysis

elif current_step == "4. Target Column":
    st.header("🎯 Select Target Column")
    
    # Check prerequisites before allowing target selection
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    elif not hasattr(st.session_state, 'task_type'):
        st.warning("⚠️ Please select task type first!")
    else:
        # Get available columns from the dataset
        data_info = st.session_state.ml_backend.get_data_info()
        columns = data_info['columns']
        
        # ---- TARGET COLUMN SELECTION ----
        target_column = st.selectbox(
            "Choose the target column (what you want to predict):",
            options=columns,
            help="This is the column that contains the values you want to predict"
        )
        
        if target_column:
            # ---- AUTOMATIC TARGET ANALYSIS ----
            # Analyze the selected target column characteristics
            target_info = st.session_state.ml_backend.detect_task_type(target_column)
            
            if target_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display target column statistics
                    st.metric("Unique Values", target_info['unique_values'])
                    st.metric("Suggested Task", target_info['suggested_task'].title())
                    if target_info['class_type']:
                        st.metric("Classification Type", target_info['class_type'].title())
                
                with col2:
                    # ---- TARGET DISTRIBUTION VISUALIZATION ----
                    # Display target distribution
                    st.subheader("📊 Target Distribution")
                    target_dist = pd.Series(target_info['target_info'])
                    fig = px.bar(x=target_dist.index, y=target_dist.values, 
                               title="Target Value Counts")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # ---- TASK VALIDATION ----
                # Validate if the chosen task type is appropriate for this target
                is_valid, message = st.session_state.ml_backend.validate_task_choice(
                    target_column, st.session_state.task_type
                )
                
                if is_valid:
                    # Success: store target column and mark as selected
                    st.success(f"✅ {message}")
                    st.session_state.target_column = target_column
                    st.session_state.target_selected = True
                else:
                    # Warning: task type might not be suitable
                    st.error(f"❌ {message}")
                    st.warning("💡 Consider changing your task type based on the target column characteristics.")

# ================================
# STEP 5: DATA PREPROCESSING
# ================================
# Handle missing values, encode categorical variables, and scale features
# Target column is automatically protected from preprocessing

elif current_step == "5. Preprocessing":
    st.header("🔧 Data Preprocessing")
    
    # Check if previous steps are completed
    if not st.session_state.target_selected:
        st.warning("⚠️ Please complete previous steps first!")
    else:
        # ---- TARGET COLUMN PROTECTION INFO ----
        # Display target column protection info
        if hasattr(st.session_state, 'target_column'):
            st.info(f"🛡️ **Target column protection**: `{st.session_state.target_column}` will NOT be preprocessed (encoded/scaled) to preserve target integrity.")
        
        # ---- PREPROCESSING OPTIONS ----
        st.subheader("Choose preprocessing options:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values handling option
            handle_missing = st.checkbox(
                "Handle Missing Values",
                help="Fill missing values: numeric columns with mean, categorical with mode (excluding target column)"
            )
            
            # Categorical encoding option
            encode_categorical = st.checkbox(
                "Encode Categorical Variables",
                help="Convert categorical feature variables to numeric using label encoding (target column excluded)"
            )
        
        with col2:
            # Feature scaling option
            scale_features = st.checkbox(
                "Scale Features",
                help="Normalize feature values to improve model performance (target column excluded)"
            )
            
            # Scaling method selection (only shown if scaling is enabled)
            if scale_features:
                scaler_type = st.selectbox(
                    "Scaling Method:",
                    ["standard", "minmax"],
                    help="Standard: mean=0, std=1. MinMax: scale to [0,1]"
                )
            else:
                scaler_type = "standard"
        
        # ---- PREPROCESSING PREVIEW ----
        # Show what will be processed based on selected options
        if any([handle_missing, encode_categorical, scale_features]):
            data_info = st.session_state.ml_backend.get_data_info()
            
            with st.expander("📋 Preview: Columns that will be processed"):
                target_col = getattr(st.session_state, 'target_column', None)
                
                # Show which columns will be affected by missing value handling
                if handle_missing:
                    missing_cols = [col for col, count in data_info['missing_values'].items() 
                                  if count > 0 and col != target_col]
                    if missing_cols:
                        st.write("**Missing values will be filled in:**", ", ".join(missing_cols))
                    else:
                        st.write("**Missing values:** No columns with missing values found")
                
                # Show which columns will be encoded
                if encode_categorical:
                    categorical_cols = [col for col, dtype in data_info['dtypes'].items() 
                                      if dtype == 'object' and col != target_col]
                    if categorical_cols:
                        st.write("**Categorical encoding will be applied to:**", ", ".join(categorical_cols))
                    else:
                        st.write("**Categorical encoding:** No categorical feature columns found")
                
                # Show which columns will be scaled
                if scale_features:
                    numeric_cols = [col for col, dtype in data_info['dtypes'].items() 
                                  if pd.api.types.is_numeric_dtype(data_info['head'][col]) and col != target_col]
                    if numeric_cols:
                        st.write(f"**{scaler_type.title()} scaling will be applied to:**", ", ".join(numeric_cols))
                    else:
                        st.write("**Feature scaling:** No numeric feature columns found")
                
                # Always show protected target column
                if target_col:
                    st.write(f"**Protected (unchanged):** `{target_col}` (target column)")
        
        # ---- PREPROCESSING EXECUTION ----
        # Prepare preprocessing options dictionary
        preprocessing_options = {
            'handle_missing': handle_missing,
            'encode_categorical': encode_categorical,
            'scale_features': scale_features,
            'scaler_type': scaler_type
        }
        
        # Apply preprocessing button
        if st.button("🔄 Apply Preprocessing", type="primary"):
            # Ensure target column is set in backend before preprocessing
            if hasattr(st.session_state, 'target_column'):
                st.session_state.ml_backend.target_column = st.session_state.target_column
            
            with st.spinner("Applying preprocessing..."):
                success, message = st.session_state.ml_backend.preprocess_data(preprocessing_options)
            
            if success:
                # ---- SUCCESS HANDLING ----
                st.success(f"✅ {message}")
                st.session_state.preprocessing_done = True
                
                # Show updated data info
                data_info = st.session_state.ml_backend.get_data_info()
                st.subheader("📊 Updated Data Preview")
                st.dataframe(data_info['head'], use_container_width=True)
            else:
                # ---- ERROR HANDLING ----
                st.error(f"❌ {message}")

# ================================
# STEP 6: DATA SPLITTING
# ================================
# Split dataset into training and testing sets with configurable parameters

elif current_step == "6. Data Split":
    st.header("✂️ Split Data")
    
    if not st.session_state.target_selected:
        st.warning("⚠️ Please complete previous steps first!")
    else:
        st.subheader("Configure train-test split:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=50,
                value=20,
                help="Percentage of data to use for testing"
            )
        
        with col2:
            random_state = st.number_input(
                "Random State",
                min_value=0,
                value=42,
                help="Set seed for reproducible results"
            )
        
        train_size = 100 - test_size
        
        # Visualization of split
        fig = go.Figure(data=[
            go.Pie(labels=['Training', 'Testing'], 
                  values=[train_size, test_size],
                  hole=0.4)
        ])
        fig.update_layout(title="Data Split Visualization", height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔀 Split Data", type="primary"):
            with st.spinner("Splitting data..."):
                success, message = st.session_state.ml_backend.split_data(
                    st.session_state.target_column,
                    test_size=test_size/100,
                    random_state=random_state
                )
            
            if success:
                st.success(f"✅ {message}")
                st.session_state.data_split = True
            else:
                st.error(f"❌ {message}")

# ================================
# STEP 7: CROSS VALIDATION SETUP
# ================================
# Configure k-fold cross validation for robust model evaluation

elif current_step == "7. Cross Validation":
    st.header("🔄 Cross Validation Setup")
    
    if not hasattr(st.session_state, 'data_split'):
        st.warning("⚠️ Please split data first!")
    else:
        use_cv = st.checkbox(
            "Enable Cross Validation",
            help="Use cross-validation for more robust model evaluation"
        )
        
        if use_cv:
            cv_folds = st.slider(
                "Number of Folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
            
            st.info(f"📊 Cross-validation will use {cv_folds} folds")
            
            # Visualization of CV
            fig = go.Figure()
            for i in range(cv_folds):
                fig.add_trace(go.Bar(
                    name=f'Fold {i+1}',
                    x=[f'Fold {i+1}'],
                    y=[1],
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Cross Validation Folds",
                xaxis_title="Folds",
                yaxis_title="Proportion",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.session_state.use_cv = True
            st.session_state.cv_folds = cv_folds
        else:
            st.session_state.use_cv = False
            st.info("📝 Cross-validation disabled. Will use train-test split only.")

# ================================
# STEP 8: MODEL SELECTION AND TRAINING
# ================================
# Choose from available ML algorithms and train selected models

elif current_step == "8. Model Selection":
    st.header("🤖 Select Machine Learning Models")
    
    if not hasattr(st.session_state, 'data_split'):
        st.warning("⚠️ Please complete data splitting first!")
    else:
        available_models = st.session_state.ml_backend.get_available_models(st.session_state.task_type)
        
        st.subheader(f"Available {st.session_state.task_type.title()} Models:")
        
        selected_models = []
        
        # Create columns for model selection
        cols = st.columns(2)
        
        for i, (model_name, model_obj) in enumerate(available_models.items()):
            col = cols[i % 2]
            
            with col:
                if st.checkbox(model_name, key=f"model_{model_name}"):
                    selected_models.append(model_name)
                    st.caption(f"Algorithm: {model_obj.__class__.__name__}")
        
        if selected_models:
            st.success(f"✅ Selected {len(selected_models)} model(s): {', '.join(selected_models)}")
            
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models..."):
                    success, message = st.session_state.ml_backend.train_models(
                        selected_models, st.session_state.task_type
                    )
                
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.models_trained = True
                    st.session_state.selected_models = selected_models
                else:
                    st.error(f"❌ {message}")
        else:
            st.warning("⚠️ Please select at least one model to train.")

# ================================
# STEP 9: MODEL EVALUATION
# ================================
# Select evaluation metrics and assess model performance

elif current_step == "9. Evaluation":
    st.header("📊 Choose Evaluation Metrics")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
    else:
        st.subheader("Select evaluation metrics:")
        
        if st.session_state.task_type == "classification":
            col1, col2 = st.columns(2)
            
            with col1:
                accuracy = st.checkbox("Accuracy", value=True)
                precision = st.checkbox("Precision", value=True)
                recall = st.checkbox("Recall", value=True)
            
            with col2:
                f1_score = st.checkbox("F1 Score", value=True)
                confusion_matrix = st.checkbox("Confusion Matrix")
                classification_report = st.checkbox("Classification Report")
            
            selected_metrics = []
            if accuracy: selected_metrics.append('accuracy')
            if precision: selected_metrics.append('precision')
            if recall: selected_metrics.append('recall')
            if f1_score: selected_metrics.append('f1_score')
            if confusion_matrix: selected_metrics.append('confusion_matrix')
            if classification_report: selected_metrics.append('classification_report')
            
        else:  # regression
            col1, col2 = st.columns(2)
            
            with col1:
                mse = st.checkbox("Mean Squared Error (MSE)", value=True)
                rmse = st.checkbox("Root Mean Squared Error (RMSE)", value=True)
            
            with col2:
                mae = st.checkbox("Mean Absolute Error (MAE)", value=True)
                r2_score = st.checkbox("R² Score", value=True)
            
            selected_metrics = []
            if mse: selected_metrics.append('mse')
            if rmse: selected_metrics.append('rmse')
            if mae: selected_metrics.append('mae')
            if r2_score: selected_metrics.append('r2_score')
        
        if selected_metrics and st.button("📈 Evaluate Models", type="primary"):
            with st.spinner("Evaluating models..."):
                # Run cross-validation if enabled
                if hasattr(st.session_state, 'use_cv') and st.session_state.use_cv:
                    cv_success, cv_results = st.session_state.ml_backend.cross_validate_models(
                        st.session_state.cv_folds
                    )
                    st.session_state.cv_results = cv_results if cv_success else None
                
                # Evaluate on test set
                eval_success, eval_results = st.session_state.ml_backend.evaluate_models(selected_metrics)
            
            if eval_success:
                st.success("✅ Model evaluation completed!")
                st.session_state.evaluation_results = eval_results
                st.session_state.selected_metrics = selected_metrics
            else:
                st.error(f"❌ {eval_results}")

# ================================
# STEP 10: RESULTS AND COMPARISON
# ================================
# Display comprehensive results, visualizations, and model comparisons

elif current_step == "10. Results":
    st.header("📊 Results & Model Comparison")
    
    if not hasattr(st.session_state, 'evaluation_results'):
        st.warning("⚠️ Please complete model evaluation first!")
    else:
        # Display cross-validation results if available
        if hasattr(st.session_state, 'cv_results') and st.session_state.cv_results:
            st.subheader("🔄 Cross-Validation Results")
            cv_df = pd.DataFrame(st.session_state.cv_results).T
            cv_df.columns = ['Mean Score', 'Std Dev', 'All Scores']
            st.dataframe(cv_df[['Mean Score', 'Std Dev']], use_container_width=True)
            
            # Plot CV results
            fig = go.Figure()
            for model_name, results in st.session_state.cv_results.items():
                fig.add_trace(go.Box(
                    y=results['all_scores'],
                    name=model_name,
                    boxpoints='all'
                ))
            
            fig.update_layout(
                title="Cross-Validation Score Distribution",
                yaxis_title="Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display test set evaluation results
        st.subheader("🎯 Test Set Evaluation Results")
        
        # Create comparison dataframe
        comparison_df = st.session_state.ml_backend.get_model_comparison(
            st.session_state.evaluation_results
        )
        
        # Display metrics table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create visualizations for model comparison
        if len(st.session_state.selected_models) > 1:
            st.subheader("📈 Model Performance Comparison")
            
            # Bar chart for primary metric
            if st.session_state.task_type == "classification":
                primary_metric = 'accuracy' if 'accuracy' in st.session_state.selected_metrics else st.session_state.selected_metrics[0]
            else:
                primary_metric = 'r2_score' if 'r2_score' in st.session_state.selected_metrics else st.session_state.selected_metrics[0]
            
            if primary_metric in comparison_df.columns:
                fig = px.bar(
                    x=comparison_df.index,
                    y=comparison_df[primary_metric],
                    title=f"Model Comparison - {primary_metric.replace('_', ' ').title()}",
                    color=comparison_df[primary_metric],
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Radar chart for multiple metrics
            if len(st.session_state.selected_metrics) > 2:
                fig = go.Figure()
                
                for model_name in comparison_df.index:
                    values = []
                    metrics = []
                    for metric in st.session_state.selected_metrics:
                        if metric in comparison_df.columns:
                            values.append(comparison_df.loc[model_name, metric])
                            metrics.append(metric.replace('_', ' ').title())
                    
                    values.append(values[0])  # Close the radar chart
                    metrics.append(metrics[0])
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=model_name
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Multi-Metric Model Comparison",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Best model recommendation
        if st.session_state.task_type == "classification":
            best_metric = 'accuracy' if 'accuracy' in comparison_df.columns else comparison_df.columns[0]
            best_model = comparison_df[best_metric].idxmax()
            best_score = comparison_df.loc[best_model, best_metric]
        else:
            best_metric = 'r2_score' if 'r2_score' in comparison_df.columns else comparison_df.columns[0]
            best_model = comparison_df[best_metric].idxmax()
            best_score = comparison_df.loc[best_model, best_metric]
        
        st.success(f"🏆 **Best Model:** {best_model} with {best_metric.replace('_', ' ').title()}: {best_score:.4f}")
        
        # Display confusion matrix for classification
        if (st.session_state.task_type == "classification" and 
            'confusion_matrix' in st.session_state.selected_metrics):
            
            st.subheader("🔄 Confusion Matrices")
            
            cols = st.columns(min(len(st.session_state.selected_models), 2))
            
            for i, (model_name, results) in enumerate(st.session_state.evaluation_results.items()):
                if 'confusion_matrix' in results:
                    col = cols[i % 2]
                    
                    with col:
                        cm = np.array(results['confusion_matrix'])
                        
                        # Create confusion matrix using matplotlib/seaborn as backup
                        try:
                            # Try plotly first
                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                text=cm,
                                colorscale='Blues',
                                showscale=True
                            ))
                            
                            fig.update_layout(
                                title=f"Confusion Matrix - {model_name}",
                                xaxis_title="Predicted",
                                yaxis_title="Actual",
                                height=400,
                                width=400
                            )
                            
                            # Add axis labels with actual class names if available
                            if hasattr(st.session_state.ml_backend, 'class_labels') and st.session_state.ml_backend.class_labels is not None:
                                class_labels = st.session_state.ml_backend.class_labels
                                if st.session_state.ml_backend.label_encoder is not None:
                                    # For label encoded targets, use original labels
                                    label_names = [str(label) for label in class_labels]
                                else:
                                    # For numeric targets, use the values directly
                                    label_names = [f"Class {label}" for label in class_labels]
                            else:
                                # Fallback to generic class names
                                label_names = [f"Class {i}" for i in range(cm.shape[0])]
                            
                            fig.update_xaxes(
                                tickvals=list(range(cm.shape[1])),
                                ticktext=label_names
                            )
                            fig.update_yaxes(
                                tickvals=list(range(cm.shape[0])),
                                ticktext=label_names
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as plotly_error:
                            # Fallback to matplotlib/seaborn
                            st.warning(f"Plotly error: {str(plotly_error)}. Using matplotlib fallback.")
                            
                            import matplotlib.pyplot as plt
                            
                            # Get class labels
                            if hasattr(st.session_state.ml_backend, 'class_labels') and st.session_state.ml_backend.class_labels is not None:
                                class_labels = st.session_state.ml_backend.class_labels
                                if st.session_state.ml_backend.label_encoder is not None:
                                    label_names = [str(label) for label in class_labels]
                                else:
                                    label_names = [f"Class {label}" for label in class_labels]
                            else:
                                label_names = [f"Class {i}" for i in range(cm.shape[0])]
                            
                            # Create matplotlib figure
                            fig_mpl, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                      xticklabels=label_names, 
                                      yticklabels=label_names, ax=ax)
                            ax.set_title(f"Confusion Matrix - {model_name}")
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            
                            st.pyplot(fig_mpl)
                            plt.close(fig_mpl)

# ================================
# APPLICATION FOOTER
# ================================

# Footer with comprehensive workflow information
st.markdown("---")
st.markdown(
       """
    <div style='text-align: center; color: #666;'>
        <p>🤖 Machine Learning Dashboard | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from ml_backend import MLBackend

# Page configuration
st.set_page_config(
    page_title="Machine Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ml_backend' not in st.session_state:
    st.session_state.ml_backend = MLBackend()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'target_selected' not in st.session_state:
    st.session_state.target_selected = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Main title
st.title("🤖 Machine Learning Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
steps = [
    "1. Upload Data",
    "2. Delete Columns",
    "3. Task Selection",
    "4. Target Column",
    "5. Preprocessing",
    "6. Data Split",
    "7. Cross Validation",
    "8. Model Selection",
    "9. Evaluation",
    "10. Results"
]

current_step = st.sidebar.radio("Select Step:", steps)

# Step 1: Upload Data
if current_step == "1. Upload Data":
    st.header("📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset in CSV or Excel format"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            success, result = st.session_state.ml_backend.load_data(uploaded_file)
        
        if success:
            st.success("✅ Data loaded successfully!")
            st.session_state.data_loaded = True
            
            # Display data info
            data_info = st.session_state.ml_backend.get_data_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", data_info['shape'][0])
                st.metric("Columns", data_info['shape'][1])
            
            with col2:
                missing_count = sum(data_info['missing_values'].values())
                st.metric("Missing Values", missing_count)
                st.metric("Memory Usage", f"{result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Display data preview
            st.subheader("📊 Data Preview")
            st.dataframe(data_info['head'], use_container_width=True)
            
            # Display data types
            st.subheader("🔢 Column Information")
            dtype_df = pd.DataFrame({
                'Column': list(data_info['dtypes'].keys()),
                'Data Type': list(data_info['dtypes'].values()),
                'Missing Values': [data_info['missing_values'][col] for col in data_info['dtypes'].keys()]
            })
            st.dataframe(dtype_df, use_container_width=True)
            
        else:
            st.error(f"❌ Error loading data: {result}")

# Step 2: Delete Columns
elif current_step == "2. Delete Columns":
    st.header("🗑️ Delete Columns (Optional)")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        data_info = st.session_state.ml_backend.get_data_info()
        
        st.info("💡 You can delete unnecessary columns to improve model performance and reduce complexity.")
        
        # Display current columns with their info
        st.subheader("📋 Current Columns")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a dataframe showing column information
            column_info = pd.DataFrame({
                'Column': list(data_info['dtypes'].keys()),
                'Data Type': list(data_info['dtypes'].values()),
                'Missing Values': [data_info['missing_values'][col] for col in data_info['dtypes'].keys()],
                'Sample Values': [str(data_info['head'][col].iloc[0]) if not pd.isna(data_info['head'][col].iloc[0]) else 'NaN' 
                                for col in data_info['dtypes'].keys()]
            })
            st.dataframe(column_info, use_container_width=True)
        
        with col2:
            st.metric("Total Columns", len(data_info['columns']))
            st.metric("Total Rows", data_info['shape'][0])
        
        # Column selection for deletion
        st.subheader("🎯 Select Columns to Delete")
        
        # Multi-select for columns to delete
        columns_to_delete = st.multiselect(
            "Choose columns to delete:",
            options=data_info['columns'],
            help="Select one or more columns that you want to remove from the dataset"
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

# Step 3: Task Selection
elif current_step == "3. Task Selection":
    st.header("🎯 Select Machine Learning Task")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        task_type = st.radio(
            "Choose the type of machine learning task:",
            ["Classification", "Regression"],
            help="Classification: Predict categories/classes. Regression: Predict continuous numerical values."
        )
        
        st.session_state.task_type = task_type.lower()
        
        st.info(f"📝 Selected task: **{task_type}**")
        
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

# Step 4: Target Column Selection
elif current_step == "4. Target Column":
    st.header("🎯 Select Target Column")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    elif not hasattr(st.session_state, 'task_type'):
        st.warning("⚠️ Please select task type first!")
    else:
        data_info = st.session_state.ml_backend.get_data_info()
        columns = data_info['columns']
        
        target_column = st.selectbox(
            "Choose the target column (what you want to predict):",
            options=columns,
            help="This is the column that contains the values you want to predict"
        )
        
        if target_column:
            # Analyze target column
            target_info = st.session_state.ml_backend.detect_task_type(target_column)
            
            if target_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Unique Values", target_info['unique_values'])
                    st.metric("Suggested Task", target_info['suggested_task'].title())
                    if target_info['class_type']:
                        st.metric("Classification Type", target_info['class_type'].title())
                
                with col2:
                    # Display target distribution
                    st.subheader("📊 Target Distribution")
                    target_dist = pd.Series(target_info['target_info'])
                    fig = px.bar(x=target_dist.index, y=target_dist.values, 
                               title="Target Value Counts")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Validate task choice
                is_valid, message = st.session_state.ml_backend.validate_task_choice(
                    target_column, st.session_state.task_type
                )
                
                if is_valid:
                    st.success(f"✅ {message}")
                    st.session_state.target_column = target_column
                    st.session_state.target_selected = True
                else:
                    st.error(f"❌ {message}")
                    st.warning("💡 Consider changing your task type based on the target column characteristics.")

# Step 5: Preprocessing
elif current_step == "5. Preprocessing":
    st.header("🔧 Data Preprocessing")
    
    if not st.session_state.target_selected:
        st.warning("⚠️ Please complete previous steps first!")
    else:
        # Display target column protection info
        if hasattr(st.session_state, 'target_column'):
            st.info(f"🛡️ **Target column protection**: `{st.session_state.target_column}` will NOT be preprocessed (encoded/scaled) to preserve target integrity.")
        
        st.subheader("Choose preprocessing options:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.checkbox(
                "Handle Missing Values",
                help="Fill missing values: numeric columns with mean, categorical with mode (excluding target column)"
            )
            
            encode_categorical = st.checkbox(
                "Encode Categorical Variables",
                help="Convert categorical feature variables to numeric using label encoding (target column excluded)"
            )
        
        with col2:
            scale_features = st.checkbox(
                "Scale Features",
                help="Normalize feature values to improve model performance (target column excluded)"
            )
            
            if scale_features:
                scaler_type = st.selectbox(
                    "Scaling Method:",
                    ["standard", "minmax"],
                    help="Standard: mean=0, std=1. MinMax: scale to [0,1]"
                )
            else:
                scaler_type = "standard"
        
        # Show what will be processed
        if any([handle_missing, encode_categorical, scale_features]):
            data_info = st.session_state.ml_backend.get_data_info()
            
            with st.expander("📋 Preview: Columns that will be processed"):
                target_col = getattr(st.session_state, 'target_column', None)
                
                if handle_missing:
                    missing_cols = [col for col, count in data_info['missing_values'].items() 
                                  if count > 0 and col != target_col]
                    if missing_cols:
                        st.write("**Missing values will be filled in:**", ", ".join(missing_cols))
                    else:
                        st.write("**Missing values:** No columns with missing values found")
                
                if encode_categorical:
                    categorical_cols = [col for col, dtype in data_info['dtypes'].items() 
                                      if dtype == 'object' and col != target_col]
                    if categorical_cols:
                        st.write("**Categorical encoding will be applied to:**", ", ".join(categorical_cols))
                    else:
                        st.write("**Categorical encoding:** No categorical feature columns found")
                
                if scale_features:
                    numeric_cols = [col for col, dtype in data_info['dtypes'].items() 
                                  if pd.api.types.is_numeric_dtype(data_info['head'][col]) and col != target_col]
                    if numeric_cols:
                        st.write(f"**{scaler_type.title()} scaling will be applied to:**", ", ".join(numeric_cols))
                    else:
                        st.write("**Feature scaling:** No numeric feature columns found")
                
                if target_col:
                    st.write(f"**Protected (unchanged):** `{target_col}` (target column)")
        
        preprocessing_options = {
            'handle_missing': handle_missing,
            'encode_categorical': encode_categorical,
            'scale_features': scale_features,
            'scaler_type': scaler_type
        }
        
        if st.button("🔄 Apply Preprocessing", type="primary"):
            # Ensure target column is set in backend before preprocessing
            if hasattr(st.session_state, 'target_column'):
                st.session_state.ml_backend.target_column = st.session_state.target_column
            
            with st.spinner("Applying preprocessing..."):
                success, message = st.session_state.ml_backend.preprocess_data(preprocessing_options)
            
            if success:
                st.success(f"✅ {message}")
                st.session_state.preprocessing_done = True
                
                # Show updated data info
                data_info = st.session_state.ml_backend.get_data_info()
                st.subheader("📊 Updated Data Preview")
                st.dataframe(data_info['head'], use_container_width=True)
            else:
                st.error(f"❌ {message}")

# Step 6: Data Split
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

# Step 7: Cross Validation
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

# Step 8: Model Selection
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

# Step 9: Evaluation Metrics
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

# Step 10: Results and Comparison
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

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 Machine Learning Dashboard | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.impute import KNNImputer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import openpyxl
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.metrics import log_loss
from scipy.special import softmax
import tempfile

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import (
    Dense, 
    Dropout, 
    Conv1D, 
    MaxPooling1D, 
    Flatten,
    LSTM, 
    SimpleRNN,
    LeakyReLU,
    Activation
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import json
import datetime

# Optional ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# Initialize session state for df
if "df" not in st.session_state:
    st.session_state.df = None

# Initialize session state for trained models
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "model_type" not in st.session_state:
    st.session_state.model_type = None

# Set page configuration
st.set_page_config(page_title="ML-XPERT", layout="wide")

# Sidebar with a logo
st.sidebar.image("ml_xpert_logo.png", width=300)
st.sidebar.title("Sections")
sections = [
    "Data Loading",
    "Data Processing",
    "EDA",
    "ML Model Training",
    "DL Model Training",
    "Evaluate",
    "Report",
    "About Me",
]

selected_section = st.sidebar.radio(
    "Select a Section", sections, index=0, key="selected_section"
)


# Data Loading Function
def upload_data():
    st.markdown("## Data Loading Section")
    uploaded_file = st.file_uploader(
        "Upload a .csv or .xlsx file", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "csv":
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

        file_size = round(uploaded_file.size / 1024, 2)
        st.markdown(f"📂 **{uploaded_file.name}** - {file_size} KB")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.markdown(
            f"**The file contains** `{st.session_state.df.shape[0]}` **rows and** `{st.session_state.df.shape[1]}` **columns."
        )

        if st.button("Save Data"):
            st.session_state.df.to_csv("saved_data.csv", index=False)
            st.success("The information has been saved.")

        if st.button("Proceed to Data Processing"):
            st.session_state.df.to_csv("saved_data.csv", index=False)
            st.success("Dataset saved. Now proceed to Data Processing.")
            # st.experimental_rerun()


# Data Preprocessing Function
def preprocess_data():
    if st.session_state.df is None:
        if os.path.exists("saved_data.csv"):
            st.session_state.df = pd.read_csv("saved_data.csv")
            st.info("Loaded saved data. Please proceed with preprocessing.")
        else:
            st.warning("Please upload data in the 'Data Loading' section first.")
            st.image(
                "no_data_image.png", width=400
            )  # Display image if no data, adjust path if needed
            return  # Exit preprocessing if no data is loaded

    st.subheader("📊 Data Preprocessing")

    # 1️⃣ Display Dataset Shape
    st.write(
        f"**Total Rows:** {st.session_state.df.shape[0]}, **Total Columns:** {st.session_state.df.shape[1]}"
    )

    # 2️⃣ Remove Duplicates
    duplicates = st.session_state.df.duplicated().sum()
    if duplicates > 0:
        st.session_state.df = st.session_state.df.drop_duplicates()
        st.write(f"✅ Removed {duplicates} duplicate rows.")
    else:
        st.write("✅ No duplicate rows found.")

    # 3️⃣ Check and Remove Null Values
    st.subheader("🔍 Missing Values Handling")

    # Count initial null values
    null_counts = st.session_state.df.isnull().sum()
    st.write("**Initial Null Values Count per Column:**")
    st.write(null_counts[null_counts > 0])

    # User-defined threshold for removing rows
    row_threshold = st.number_input(
        "Enter max allowed null values per row:",
        min_value=0,
        max_value=st.session_state.df.shape[1],
        value=3,
    )
    st.session_state.df = st.session_state.df[
        st.session_state.df.isnull().sum(axis=1) <= row_threshold
    ]
    st.write(f"✅ Removed rows with more than {row_threshold} null values.")

    # User-defined threshold for removing columns
    col_threshold = st.number_input(
        "Enter max allowed null values per column:",
        min_value=0,
        max_value=st.session_state.df.shape[0],
        value=100,
    )
    cols_to_drop = st.session_state.df.columns[
        st.session_state.df.isnull().sum() > col_threshold
    ].tolist()
    if cols_to_drop:
        st.session_state.df.drop(columns=cols_to_drop, inplace=True)
        st.write(f"✅ Dropped columns: {cols_to_drop}")
    else:
        st.write("✅ No columns removed.")

    # 4️⃣ Impute Missing Values
    st.subheader("📌 Impute Missing Values")
    for col in st.session_state.df.columns:
        if st.session_state.df[col].isnull().sum() > 0:
            method = st.selectbox(
                f"Choose method to fill missing values for {col}:",
                ["Mean", "Median", "Mode"],
            )
            if method == "Mean":
                st.session_state.df[col].fillna(
                    st.session_state.df[col].mean(), inplace=True
                )
            elif method == "Median":
                st.session_state.df[col].fillna(
                    st.session_state.df[col].median(), inplace=True
                )
            elif method == "Mode":
                st.session_state.df[col].fillna(
                    st.session_state.df[col].mode()[0], inplace=True
                )
            st.write(f"✅ {method} imputation applied for {col}")

    # Re-check null values after imputation
    st.write("**Final Null Values Count (After Imputation):**")
    st.write(st.session_state.df.isnull().sum())

    # 5️⃣ Drop Unimportant Columns
    st.subheader("🗑️ Drop Less Important Columns")
    selected_columns = st.multiselect(
        "Select columns to remove:", st.session_state.df.columns
    )
    if selected_columns:
        st.session_state.df.drop(columns=selected_columns, inplace=True)
        st.write(f"✅ Dropped columns: {selected_columns}")

    # 6️⃣ Target Variable Selection
    st.subheader("🎯 Target Variable Selection")
    target_col = st.selectbox("Select the target column:", st.session_state.df.columns)

    # 7️⃣ Problem Type Selection
    problem_type = st.radio(
        "Is this a Classification or Regression problem?",
        ("Classification", "Regression"),
    )

    # 8️⃣ Encoding Categorical Columns
    st.subheader("🔄 Encoding Categorical Features")
    categorical_cols = st.session_state.df.select_dtypes(
        include=["object"]
    ).columns.tolist()
    encoding_methods = {}

    if categorical_cols:
        for col in categorical_cols:
            encoding_methods[col] = st.selectbox(
                f"Choose encoding for {col}:", ["Label Encoding", "One-Hot Encoding"]
            )

        for col, method in encoding_methods.items():
            if method == "Label Encoding":
                le = LabelEncoder()
                st.session_state.df[col] = le.fit_transform(st.session_state.df[col])
            elif method == "One-Hot Encoding":
                st.session_state.df = pd.get_dummies(st.session_state.df, columns=[col])

        st.write("✅ Categorical encoding applied!")

    # 9️⃣ Normalization & Standardization
    st.subheader("⚙️ Feature Scaling")
    scaling_cols = st.multiselect(
        "Select columns to apply scaling:", st.session_state.df.columns
    )
    scaling_method = st.radio(
        "Choose scaling method:",
        ["Standardization (Z-score)", "Normalization (Min-Max)"],
    )

    if scaling_cols:
        scaler = (
            StandardScaler()
            if scaling_method.startswith("Standard")
            else MinMaxScaler()
        )
        st.session_state.df[scaling_cols] = scaler.fit_transform(
            st.session_state.df[scaling_cols]
        )
        st.write(f"✅ {scaling_method} applied on selected columns!")

    # 🔟 Train-Test Split
    st.subheader("📚 Train-Test Split")
    test_size = st.slider(
        "Select Test Data Percentage:",
        min_value=0.1,
        max_value=0.5,
        step=0.05,
        value=0.2,
    )

    X = st.session_state.df.drop(columns=[target_col])
    y = st.session_state.df[target_col]

    # Use try-except to handle potential stratification errors
    try:
        if problem_type == "Classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
    except ValueError as e:
        if "The least populated class in y has only 1 member" in str(e):
            st.warning("Stratification not possible due to limited samples in some classes. Using regular train-test split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            raise e

    # Store processed data in session state to avoid errors in other sections
    st.session_state.processed_data = (X_train, X_test, y_train, y_test)

    st.write(f"✅ Training Data: {X_train.shape[0]} records")
    st.write(f"✅ Testing Data: {X_test.shape[0]} records")

    # 📊 Statistical Summary
    if st.button("Statistical Summary"):
        st.subheader("📊 Statistical Summary of Data")
        st.dataframe(st.session_state.df.describe())

    # 📥 Download Processed Data
    st.subheader("📥 Download Processed Data")
    file_format = st.radio("Select File Format:", ["CSV", "Excel"])

    def convert_df(current_df):
        if file_format == "CSV":
            return current_df.to_csv(index=False).encode("utf-8")
        elif file_format == "Excel":  # Corrected condition
            excel_buffer = io.BytesIO()  # ADDED: Create BytesIO buffer
            current_df.to_excel(
                excel_buffer, index=False, engine="openpyxl"
            )  # ADDED: Write to buffer with excel_writer
            return excel_buffer.getvalue()  # ADDED: Return bytes from buffer

    file_name = st.text_input("Enter file name:", "processed_data")

    if st.button("Download Data"):
        filedata = convert_df(
            st.session_state.df
        )  # call convert_df and store result to variable
        st.download_button(
            label="📥 Download Processed Data",
            data=filedata,  # use variable here
            file_name=f"{file_name}.{'csv' if file_format == 'CSV' else 'xlsx'}",  # corrected extension logic
            mime=(
                "text/csv"
                if file_format == "CSV"
                else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),  # corrected mime logic
        )

    return st.session_state.df, X_train, X_test, y_train, y_test


def eda_section():
    if st.session_state.df is None:
        # Try to load from processed data file
        if os.path.exists("processed_data.csv"):
            st.session_state.df = pd.read_csv("processed_data.csv")
            st.info("Loaded processed data for EDA analysis.")
        elif os.path.exists("saved_data.csv"):
            st.session_state.df = pd.read_csv("saved_data.csv")
            st.warning(
                "Using original saved data. Please process the data in the 'Data Processing' section first for better analysis."
            )
        else:
            st.warning(
                "Please upload and process data in the 'Data Loading' and 'Data Processing' sections first."
            )
            st.image("no_data_image.png", width=400)
            return

    st.markdown("## 📊 EDA Section")

    # Button to show EDA sections
    if st.button("Show EDA Details"):
        # Processed Dataset Preview
        st.markdown("### Processed Dataset Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

        # Statistical Summary
        st.markdown("### 📊 Statistical Summary of Processed Data")
        st.dataframe(st.session_state.df.describe())

        # Data Information
        st.markdown("### ℹ️ Data Information")
        col_info = pd.DataFrame(
            {
                "Column": st.session_state.df.columns,
                "Data Type": st.session_state.df.dtypes.astype(str),
                "Non-Null Count": st.session_state.df.count().values,
                "Null Count": st.session_state.df.isnull().sum().values,
                "Unique Values": [
                    st.session_state.df[col].nunique()
                    for col in st.session_state.df.columns
                ],
            }
        )
        st.dataframe(col_info)

    # Visualization Section
    st.markdown("### 📈 Data Visualization")

    # Create tabs for different types of visualizations
    viz_tabs = st.tabs(["Basic Plots", "Distribution Analysis", "Correlation Analysis"])

    with viz_tabs[0]:
        st.markdown("#### Basic Plots")

        # Column selection for visualization
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis column:", st.session_state.df.columns)
        with col2:
            y_col = st.selectbox(
                "Select Y-axis column:",
                st.session_state.df.columns,
                index=min(1, len(st.session_state.df.columns) - 1),
            )  # Default to second column if available

        # Plot type selection
        plot_type = st.selectbox(
            "Select plot type:",
            [
                "Scatter Plot",
                "Line Plot",
                "Bar Chart",
                "Histogram",
                "Box Plot",
                "Violin Plot",
                "KDE Plot",  # KDE Plot
            ],
        )

        # Plot generation
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            if plot_type == "Scatter Plot":
                sns.scatterplot(
                    x=st.session_state.df[x_col], y=st.session_state.df[y_col], ax=ax
                )
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            elif plot_type == "Line Plot":
                sns.lineplot(
                    x=st.session_state.df[x_col], y=st.session_state.df[y_col], ax=ax
                )
                ax.set_title(f"Line Plot: {x_col} vs {y_col}")
            elif plot_type == "Bar Chart":
                if st.session_state.df[x_col].nunique() > 30:
                    st.warning(
                        f"Column '{x_col}' has too many unique values for a bar chart. Consider selecting a different column."
                    )
                    # Create a simplified bar chart with top categories
                    top_categories = (
                        st.session_state.df[x_col].value_counts().nlargest(20).index
                    )
                    filtered_df = st.session_state.df[
                        st.session_state.df[x_col].isin(top_categories)
                    ]
                    sns.barplot(x=filtered_df[x_col], y=filtered_df[y_col], ax=ax)
                    ax.set_title(f"Bar Chart (Top 20 categories): {x_col} vs {y_col}")
                    plt.xticks(rotation=45, ha="right")
                else:
                    sns.barplot(
                        x=st.session_state.df[x_col],
                        y=st.session_state.df[y_col],
                        ax=ax,
                    )
                    ax.set_title(f"Bar Chart: {x_col} vs {y_col}")
                    plt.xticks(rotation=45, ha="right")
            elif plot_type == "Histogram":
                sns.histplot(st.session_state.df[x_col], bins=20, kde=True, ax=ax)
                ax.set_title(f"Histogram of {x_col}")
            elif plot_type == "Box Plot":
                sns.boxplot(
                    x=st.session_state.df[x_col], y=st.session_state.df[y_col], ax=ax
                )
                ax.set_title(f"Box Plot: {x_col} vs {y_col}")
                plt.xticks(rotation=45, ha="right")
            elif plot_type == "Violin Plot":
                sns.violinplot(
                    x=st.session_state.df[x_col], y=st.session_state.df[y_col], ax=ax
                )
                ax.set_title(f"Violin Plot: {x_col} vs {y_col}")
                plt.xticks(rotation=45, ha="right")
            elif plot_type == "KDE Plot":  # KDE Plot implementation
                sns.kdeplot(st.session_state.df[x_col], ax=ax, fill=True)
                ax.set_title(f"KDE Plot: {x_col}")

            plt.tight_layout()
            st.pyplot(fig)

            # Download button for the visualization
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            random_id = np.random.randint(10000, 99999)
            st.download_button(
                label="Download Visualization",
                data=img_buffer.getvalue(),
                file_name=f"{plot_type.replace(' ', '_').lower()}_{x_col}_{y_col}.png",
                mime="image/png",
                key=f"viz_download_{timestamp}_{random_id}"  # Add unique key with random component
            )
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info(
                "This might happen if the selected columns are not compatible with the chosen plot type. Try different columns or a different plot type."
            )

    with viz_tabs[1]:
        st.markdown("#### Distribution Analysis")

        # Select a column for distribution analysis
        dist_col_options = st.session_state.df.columns.tolist()
        selected_dist_col = st.selectbox(
            "Select a column for distribution analysis:", dist_col_options
        )

        # Distribution plot type selection
        dist_plot_type = st.selectbox(
            "Select distribution plot type:",
            [
                "Histogram",
                "Box Plot",
                "Violin Plot",
                "Pie Chart",  # Pie Chart added here
                "KDE Plot",
            ],
        )

        # Create distribution plots
        fig, ax = plt.subplots(figsize=(10, 10))

        try:
            if dist_plot_type == "Histogram":
                sns.histplot(st.session_state.df[selected_dist_col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {selected_dist_col}")

            elif dist_plot_type == "Box Plot":
                sns.boxplot(x=st.session_state.df[selected_dist_col], ax=ax)
                ax.set_title(f"Box Plot of {selected_dist_col}")

            elif dist_plot_type == "Violin Plot":
                sns.violinplot(x=st.session_state.df[selected_dist_col], ax=ax)
                ax.set_title(f"Violin Plot of {selected_dist_col}")

            elif dist_plot_type == "Pie Chart":
                if st.session_state.df[selected_dist_col].nunique() > 30:
                    st.warning(
                        f"Column '{selected_dist_col}' has many unique values for a pie chart. Displaying top 20 categories."
                    )
                    top_categories = (
                        st.session_state.df[selected_dist_col]
                        .value_counts()
                        .nlargest(20))
                else:
                    top_categories = st.session_state.df[selected_dist_col].value_counts()
                ax.pie(
                    top_categories,
                    labels=top_categories.index,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.set_title(f"Pie Chart: {selected_dist_col}")

            elif dist_plot_type == "KDE Plot":
                sns.kdeplot(st.session_state.df[selected_dist_col], ax=ax, fill=True)
                ax.set_title(f"KDE Plot of {selected_dist_col}")

            plt.tight_layout()
            st.pyplot(fig)

            # Download button for the distribution visualization
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            random_id = np.random.randint(10000, 99999)
            st.download_button(
                label="Download Visualization",
                data=img_buffer.getvalue(),
                file_name=f"{dist_plot_type.replace(' ', '_').lower()}_{selected_dist_col}.png",
                mime="image/png",
                key=f"dist_viz_download_{timestamp}_{random_id}"  # Add unique key with random component
            )

        except Exception as e:
            st.error(f"Error creating distribution visualization: {str(e)}")
            st.info(
                "This might happen if the selected column is not compatible with the chosen plot type. Try different columns or a different plot type."
            )

            # Statistics
            st.markdown("##### Statistical Insights")
            stats = st.session_state.df[selected_dist_col].describe()
            st.write(stats)

            # Skewness and Kurtosis
            if pd.api.types.is_numeric_dtype(
                st.session_state.df[selected_dist_col]
            ):  # Calculate skewness and kurtosis only for numeric columns
                skewness = st.session_state.df[selected_dist_col].skew()
                kurtosis = st.session_state.df[selected_dist_col].kurtosis()

                st.write(
                    f"**Skewness:** {skewness:.4f} ({'Highly Skewed' if abs(skewness) > 1 else 'Moderately Skewed' if abs(skewness) > 0.5 else 'Approximately Symmetric'})"
                )
                st.write(
                    f"**Kurtosis:** {kurtosis:.4f} ({'Heavy-tailed' if kurtosis > 1 else 'Light-tailed' if kurtosis < -1 else 'Normal-like tails'})"
                )
            else:
                st.write(
                    "Statistical insights (Skewness and Kurtosis) are only available for numerical columns."
                )

    with viz_tabs[2]:
        st.markdown("#### Correlation Analysis")

        # Get only numeric columns for correlation
        numeric_df = st.session_state.df.select_dtypes(include=["int64", "float64"])

        if numeric_df.shape[1] < 2:
            st.warning(
                "Need at least 2 numerical columns to perform correlation analysis."
            )
        else:
            # Correlation Method
            corr_method = st.radio(
                "Correlation Method:", ["Pearson", "Spearman", "Kendall"]
            )

            # Calculate correlation
            corr_matrix = numeric_df.corr(method=corr_method.lower())

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=1,
                vmin=-1,
                center=0,
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
            )

            plt.title(f"{corr_method} Correlation Heatmap")
            plt.tight_layout()
            st.pyplot(fig)

            # Show top correlations
            st.markdown("##### Top Positive Correlations")
            # Get upper triangle of correlation matrix
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            # Remove self-correlations (which are always 1) and duplicates
            corr_pairs = corr_pairs[corr_pairs < 0.999].drop_duplicates()
            st.write(corr_pairs.head(10))

            st.markdown("##### Top Negative Correlations")
            st.write(corr_pairs.tail(10))


def save_and_download_model(model, le=None, model_name=""):
    """Helper function to save and create download button for models"""
    st.markdown("##### Model Download")
    
    try:
        # Create a unique filename using timestamp and random number
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        random_id = np.random.randint(10000, 99999)
        
        # Use appropriate extension based on model type
        if isinstance(model, tf.keras.Model):
            model_filename = f"dl_model_{timestamp}_{random_id}.keras"
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                model.save(tmp_model.name)  # Save with .keras extension
                tmp_path = tmp_model.name
        else:
            model_filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}_{random_id}.joblib"
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_model:
                if le is not None:
                    # For classification models, save both model and label encoder
                    joblib.dump((model, le), tmp_model.name)
                else:
                    # For regression models, save only the model
                    joblib.dump(model, tmp_model.name)
                tmp_path = tmp_model.name
        
        # Read the saved model file
        with open(tmp_path, 'rb') as f:
            model_bytes = f.read()
        
        # Create download button with unique key
        st.download_button(
            label=f"Download {model_name}",
            data=model_bytes,
            file_name=model_filename,
            mime="application/octet-stream",
            key=f"model_download_{timestamp}_{random_id}"  # Add unique key with random component
        )
        
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except Exception as e:
            st.warning(f"Warning: Could not delete temporary file: {str(e)}")
            
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        st.info("Model was trained but couldn't be saved for download.")


def model_training_section(X_train, X_test, y_train, y_test):
    """Main function for model training and evaluation"""
    st.subheader("🤖 Model Training")
    
    # Model selection
    regression_models = [
        "Linear Regression",
        "Elastic Net Regression",
        "Decision Tree Regression",
        "Random Forest Regression",
        "Gradient Boosting Regression",
        "SVR Regression"
    ]
    
    classification_models = [
        "Logistic Regression",
        "Decision Tree Classifier",
        "Random Forest Classifier",
        "Support Vector Machine (SVM)",
        "K-Nearest Neighbors (KNN)",
        "Gradient Boosting Classifier"
    ]
    
    if XGBOOST_AVAILABLE:
        classification_models.append("XGBoost Classifier")
    if LIGHTGBM_AVAILABLE:
        classification_models.append("LightGBM Classifier")
    
    # Problem type selection - changed from selectbox to radio
    problem_type = st.radio(
        "Select Problem Type",
        ["Regression", "Classification"]
    )
    
    if problem_type == "Regression":
        selected_model = st.selectbox("Select Model", regression_models)
        train_regression_model(X_train, X_test, y_train, y_test, selected_model)
    else:
        selected_model = st.selectbox("Select Model", classification_models)
        train_classification_model(X_train, X_test, y_train, y_test, selected_model)

def train_regression_model(X_train, X_test, y_train, y_test, selected_model):
    """Train and evaluate regression models"""
    try:
        # Common parameters
        calculate_intercept = st.checkbox("Calculate Intercept", value=True)
        loss_functions = st.multiselect(
            "Select Loss Functions",
            ["MSE", "MAE", "RMSE"],
            default=["MSE", "RMSE"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_model == "Linear Regression":
                if st.button("Train Linear Regression Model"):
                    model = LinearRegression(fit_intercept=calculate_intercept)
                    train_and_evaluate_model(
                        model, X_train, X_test, y_train, y_test,
                        "Linear Regression", loss_functions
                    )
            
            elif selected_model == "Elastic Net Regression":
                l1_ratio = st.slider("L1 Ratio (0=Ridge, 1=Lasso)", 0.0, 1.0, 0.5)
                alpha = st.slider("Alpha (Regularization Strength)", 0.0, 1.0, 0.1)
                
                if st.button("Train Elastic Net Regression Model"):
                    model = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=calculate_intercept,
                        random_state=42
                    )
                    train_and_evaluate_model(
                        model, X_train, X_test, y_train, y_test,
                        "Elastic Net Regression", loss_functions
                    )
            
            elif selected_model == "Decision Tree Regression":
                max_depth = st.number_input("Maximum depth", 1, 50, 5)
                min_samples_split = st.number_input("Min samples split", 2, 20, 2)
                
                if st.button("Train Decision Tree Regression Model"):
                    model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42
                    )
                    train_and_evaluate_model(
                        model, X_train, X_test, y_train, y_test,
                        "Decision Tree Regression", loss_functions
                    )
            
            elif selected_model == "Random Forest Regression":
                n_estimators = st.number_input("Number of trees", 10, 1000, 100)
                max_depth = st.number_input("Maximum depth", 1, 50, 5)
                
                if st.button("Train Random Forest Regression Model"):
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42
                    )
                    train_and_evaluate_model(
                        model, X_train, X_test, y_train, y_test,
                        "Random Forest Regression", loss_functions
                    )
            
            elif selected_model == "Gradient Boosting Regression":
                n_estimators = st.number_input("Number of boosting stages", 10, 1000, 100)
                learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1)
                
                if st.button("Train Gradient Boosting Regression Model"):
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    train_and_evaluate_model(
                        model, X_train, X_test, y_train, y_test,
                        "Gradient Boosting Regression", loss_functions
                    )
            
            elif selected_model == "SVR Regression":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], index=1)
                C = st.number_input("C (Regularization)", 0.1, 10.0, 1.0)
                
                if st.button("Train SVR Model"):
                    model = SVR(kernel=kernel, C=C)
                    train_and_evaluate_model(
                        model, X_train, X_test, y_train, y_test,
                        "SVR Regression", loss_functions
                    )
        
        with col2:
            if st.session_state.trained_model is not None:
                save_and_download_model(
                    st.session_state.trained_model,
                    model_name=st.session_state.model_type
                )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data and parameters.")

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, metrics):
    """Train and evaluate a model with specified metrics"""
    try:
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store the model
        st.session_state.trained_model = model
        st.session_state.model_type = model_name
        
        st.success(f"{model_name} Model Trained!")
        
        # Display metrics
        st.markdown("##### Performance Metrics:")
        for metric in metrics:
            if metric == "MSE":
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            elif metric == "MAE":
                mae = mean_absolute_error(y_test, y_pred)
                st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            elif metric == "RMSE":
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        r2 = r2_score(y_test, y_pred)
        st.write(f"R-squared (R2): {r2:.4f}")
        
        # Display feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.markdown("##### Feature Importance:")
            importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.dataframe(importances)
        elif hasattr(model, 'coef_'):
            st.markdown("##### Feature Importance:")
            importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            st.dataframe(importances)
    
    except Exception as e:
        st.error(f"An error occurred during model training: {str(e)}")
        st.info("Please check your data and parameters.")

def train_classification_model(X_train, X_test, y_train, y_test, selected_model):
    """Train and evaluate classification models"""
    try:
        # Common parameters
        class_weight = st.selectbox(
            "Class Weight",
            ["balanced", "None"],
            index=1
        )
        class_weight = "balanced" if class_weight == "balanced" else None
        
        metrics = st.multiselect(
            "Select Metrics",
            ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            default=["Accuracy", "F1-Score"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_model == "Logistic Regression":
                # Hyperparameters
                penalty = st.selectbox(
                    "Select regularization type",
                    ["l1", "l2", "elasticnet", "none"],
                    index=1
                )
                
                C = 1.0
                l1_ratio = 0.5
                
                if penalty != "none":
                    C = st.number_input(
                        "Inverse of regularization strength (C)",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1
                    )
                    
                    if penalty == "elasticnet":
                        l1_ratio = st.number_input(
                            "L1 ratio (0 = L2, 1 = L1)",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.1
                        )
                
                if st.button("Train Logistic Regression Model"):
                    model = LogisticRegression(
                        penalty=penalty,
                        C=C,
                        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
                        class_weight=class_weight,
                        random_state=42
                    )
                    evaluate_classification_model(
                        model, X_train, X_test, y_train, y_test,
                        "Logistic Regression", metrics
                    )
            
            elif selected_model == "Decision Tree Classifier":
                max_depth = st.number_input("Maximum depth", 1, 50, 5)
                min_samples_split = st.number_input("Min samples split", 2, 20, 2)
                
                if st.button("Train Decision Tree Classifier"):
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        class_weight=class_weight,
                        random_state=42
                    )
                    evaluate_classification_model(
                        model, X_train, X_test, y_train, y_test,
                        "Decision Tree Classifier", metrics
                    )
            
            elif selected_model == "Random Forest Classifier":
                n_estimators = st.number_input("Number of trees", 10, 1000, 100)
                max_depth = st.number_input("Maximum depth", 1, 50, 5)
                
                if st.button("Train Random Forest Classifier"):
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        class_weight=class_weight,
                        random_state=42
                    )
                    evaluate_classification_model(
                        model, X_train, X_test, y_train, y_test,
                        "Random Forest Classifier", metrics
                    )
            
            elif selected_model == "Support Vector Machine (SVM)":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], index=1)
                C = st.number_input("C (Regularization)", 0.1, 10.0, 1.0)
                
                if kernel == "poly":
                    degree = st.number_input("Polynomial degree", 2, 5, 3)
                else:
                    degree = 3
                
                if st.button("Train SVM Model"):
                    model = SVC(
                        kernel=kernel,
                        C=C,
                        degree=degree,
                        class_weight=class_weight,
                        probability=True,
                        random_state=42
                    )
                    evaluate_classification_model(
                        model, X_train, X_test, y_train, y_test,
                        "SVM", metrics
                    )
            
            elif selected_model == "K-Nearest Neighbors (KNN)":
                n_neighbors = st.number_input("Number of neighbors", 1, 20, 5)
                weights = st.selectbox("Weight function", ["uniform", "distance"])
                
                if st.button("Train KNN Model"):
                    model = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights
                    )
                    evaluate_classification_model(
                        model, X_train, X_test, y_train, y_test,
                        "KNN", metrics
                    )
            
            elif selected_model == "Gradient Boosting Classifier":
                n_estimators = st.number_input("Number of boosting stages", 10, 1000, 100)
                learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1)
                
                if st.button("Train Gradient Boosting Model"):
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    evaluate_classification_model(
                        model, X_train, X_test, y_train, y_test,
                        "Gradient Boosting", metrics
                    )
            
            elif selected_model == "XGBoost Classifier" and XGBOOST_AVAILABLE:
                n_estimators = st.number_input("Number of boosting rounds", 10, 1000, 100)
                max_depth = st.number_input("Maximum depth", 1, 20, 6)
                learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.3)
                
                if st.button("Train XGBoost Model"):
                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    evaluate_classification_model(
                        model, X_train, X_test, y_train, y_test,
                        "XGBoost", metrics
                    )
            
            elif selected_model == "LightGBM Classifier" and LIGHTGBM_AVAILABLE:
                n_estimators = st.number_input("Number of boosting rounds", 10, 1000, 100)
                learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1)
                num_leaves = st.number_input("Number of leaves", 2, 131, 31)
                
                if st.button("Train LightGBM Model"):
                    model = lgb.LGBMClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        random_state=42
                    )
                    evaluate_classification_model(
                        model, X_train, X_test, y_train, y_test,
                        "LightGBM", metrics
                    )
        
        with col2:
            if st.session_state.trained_model is not None:
                save_and_download_model(
                    st.session_state.trained_model,
                    model_name=st.session_state.model_type
                )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data and parameters.")

def evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name, metrics):
    """Evaluate a classification model with specified metrics"""
    try:
        # Convert labels to integers if they're not
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Train the model
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Store the model
        st.session_state.trained_model = model
        st.session_state.model_type = model_name
        
        st.success(f"{model_name} Model Trained!")
        
        # Save and create download button for the model
        save_and_download_model(model, le=le, model_name=model_name)
        
        # Calculate loss based on problem type
        st.markdown("##### Loss Metrics:")
        try:
            if len(le.classes_) == 2:  # Binary classification
                # Binary cross-entropy loss
                bce_loss = log_loss(y_test_encoded, y_pred_proba[:, 1])
                st.write(f"Binary Cross-Entropy Loss: {bce_loss:.4f}")
            else:  # Multi-class classification
                # Categorical cross-entropy loss
                cce_loss = log_loss(y_test_encoded, y_pred_proba)
                st.write(f"Categorical Cross-Entropy Loss: {cce_loss:.4f}")
        except Exception as e:
            st.warning(f"Could not calculate loss: {str(e)}")
        
        # Display metrics
        st.markdown("##### Performance Metrics:")
        for metric in metrics:
            if metric == "Accuracy":
                accuracy = accuracy_score(y_test_encoded, y_pred)
                st.write(f"Accuracy: {accuracy:.4f}")
            elif metric == "Precision":
                precision = precision_score(y_test_encoded, y_pred, average='weighted')
                st.write(f"Precision: {precision:.4f}")
            elif metric == "Recall":
                recall = recall_score(y_test_encoded, y_pred, average='weighted')
                st.write(f"Recall: {recall:.4f}")
            elif metric == "F1-Score":
                f1 = f1_score(y_test_encoded, y_pred, average='weighted')
                st.write(f"F1-Score: {f1:.4f}")
            elif metric == "ROC-AUC":
                if len(le.classes_) == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
                    st.write(f"ROC-AUC: {roc_auc:.4f}")
                else:  # Multi-class
                    roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
                    st.write(f"ROC-AUC (OvR): {roc_auc:.4f}")
        
        # Display confusion matrix
        st.markdown("##### Confusion Matrix")
        cm = confusion_matrix(y_test_encoded, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3))  # Smaller figure size
        sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm, cmap='Blues', annot_kws={"size": 8})
        ax_cm.set_title('Confusion Matrix', fontsize=10)
        ax_cm.set_ylabel('True Label', fontsize=8)
        ax_cm.set_xlabel('Predicted Label', fontsize=8)
        ax_cm.tick_params(axis='both', which='major', labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_cm)
        
        # Display feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.markdown("##### Feature Importance")
            importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp, ax_imp = plt.subplots(figsize=(5, 4))  # Smaller figure size
            sns.barplot(data=importances.head(15), x='Importance', y='Feature', ax=ax_imp)
            ax_imp.set_title('Feature Importance', fontsize=10)
            ax_imp.set_xlabel('Importance Score', fontsize=8)
            ax_imp.set_ylabel('Feature Name', fontsize=8)
            ax_imp.tick_params(axis='both', which='major', labelsize=7)
            plt.tight_layout()
            st.pyplot(fig_imp)
            
            st.markdown("#### Feature Importance Table")
            st.dataframe(importances)
        
        elif hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_.shape) == 1:
                coefficients = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
            else:
                coefficients = pd.DataFrame(
                    model.coef_,
                    columns=X_train.columns
                )
            
            fig_coef, ax_coef = plt.subplots(figsize=(5, 4))  # Smaller figure size
            sns.barplot(data=coefficients.head(15), x='Coefficient', y='Feature', ax=ax_coef)
            ax_coef.set_title('Feature Coefficients', fontsize=10)
            ax_coef.set_xlabel('Coefficient Value', fontsize=8)
            ax_coef.set_ylabel('Feature Name', fontsize=8)
            ax_coef.tick_params(axis='both', which='major', labelsize=7)
            plt.tight_layout()
            st.pyplot(fig_coef)
            
            st.markdown("#### Feature Coefficients Table")
            st.dataframe(coefficients)
        
        else:
            st.info("Feature importance visualization is not available for this type of model.")
    
    except Exception as e:
        st.error(f"An error occurred during model evaluation: {str(e)}")
        st.info("Please check your data and parameters.")

def dl_model_training_section(X_train, X_test, y_train, y_test):
    """Deep Learning Model Training Section"""
    try:
        st.markdown("## 🧠 Deep Learning Model Training Section")
        
        # Convert DataFrames to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Determine problem type and prepare target data
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            problem_type = "binary"
            y_train = y_train.astype('float32')
            y_test = y_test.astype('float32')
        elif n_classes > 2:
            problem_type = "multiclass"
            # Convert to categorical
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            y_train = to_categorical(y_train_encoded)
            y_test = to_categorical(y_test_encoded)
        else:
            problem_type = "regression"
            y_train = y_train.astype('float32')
            y_test = y_test.astype('float32')
        
        # Model Selection
        dl_models = ["Artificial Neural Network (ANN)", 
                    "Convolutional Neural Network (CNN)",
                    "Recurrent Neural Network (RNN)"]
        
        selected_model = st.selectbox("Select Deep Learning Model", dl_models)
        
        # Display problem type
        st.info(f"Detected Problem Type: {problem_type.title()}")
        if problem_type == "multiclass":
            st.info(f"Number of Classes: {n_classes}")
        
        # Common hyperparameters
        num_layers = st.number_input("Number of Layers", min_value=1, max_value=10, value=3)
        
        # Lists to store layer configurations
        neurons_per_layer = []
        activation_functions = []
        
        # Available activation functions
        activation_options = ["ReLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU"]
        
        # Layer configuration
        st.markdown("### Layer Configuration")
        for i in range(num_layers):
            col1, col2 = st.columns(2)
            with col1:
                neurons = st.number_input(f"Neurons in Layer {i+1}", 
                                        min_value=1, 
                                        max_value=512, 
                                        value=64)
                neurons_per_layer.append(neurons)
            
            with col2:
                activation = st.selectbox(f"Activation Function for Layer {i+1}", 
                                        activation_options,
                                        key=f"activation_{i}")
                activation_functions.append(activation.lower())
        
        # Dropout configuration
        dropout_rate = st.slider("Dropout Rate", 
                               min_value=0.0, 
                               max_value=0.5, 
                               value=0.2, 
                               step=0.05)
        
        # Optimizer selection
        optimizer_options = {
            "Adam": Adam,
            "SGD": SGD,
            "RMSprop": RMSprop,
            "Adagrad": Adagrad
        }
        
        optimizer_choice = st.selectbox("Select Optimizer", list(optimizer_options.keys()))
        learning_rate = st.number_input("Learning Rate", 
                                      min_value=0.0001, 
                                      max_value=0.1, 
                                      value=0.001, 
                                      format="%.4f")
        
        # Loss function selection based on problem type
        if problem_type == "binary":
            loss_function = "binary_crossentropy"
            st.info("Using Binary Cross-Entropy loss for binary classification")
        elif problem_type == "multiclass":
            loss_function = "categorical_crossentropy"
            st.info("Using Categorical Cross-Entropy loss for multiclass classification")
        else:
            loss_function = "mean_squared_error"
            st.info("Using Mean Squared Error loss for regression")
        
        # Training parameters
        epochs = st.number_input("Number of Epochs", 
                               min_value=1, 
                               max_value=500, 
                               value=50)
        
        batch_size = st.number_input("Batch Size", 
                                    min_value=1, 
                                    max_value=256, 
                                    value=32)
        
        # Model training button
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a while."):
                try:
                    # Create the model based on selection
                    model = Sequential()
                    
                    # Input layer
                    if selected_model == "Artificial Neural Network (ANN)":
                        model.add(Dense(neurons_per_layer[0], 
                                      input_shape=(X_train.shape[1],),
                                      activation=activation_functions[0]))
                    elif selected_model == "Convolutional Neural Network (CNN)":
                        # Reshape data for CNN
                        X_train_reshaped = X_train.reshape(X_train.shape[0], 
                                                         X_train.shape[1], 1)
                        X_test_reshaped = X_test.reshape(X_test.shape[0], 
                                                       X_test.shape[1], 1)
                        
                        model.add(Conv1D(filters=neurons_per_layer[0], 
                                       kernel_size=3, 
                                       activation=activation_functions[0],
                                       input_shape=(X_train.shape[1], 1)))
                    elif selected_model == "Recurrent Neural Network (RNN)":
                        # Calculate proper sequence length and features
                        n_features = X_train.shape[1]  # Use all features
                        sequence_length = 1  # Default to 1 for single time step
                        
                        # Reshape the data for RNN (samples, time steps, features)
                        X_train_reshaped = X_train.reshape(X_train.shape[0], sequence_length, n_features)
                        X_test_reshaped = X_test.reshape(X_test.shape[0], sequence_length, n_features)
                        
                        model.add(LSTM(neurons_per_layer[0], 
                                     input_shape=(sequence_length, n_features),
                                     activation=activation_functions[0],
                                     return_sequences=num_layers > 1))
                    else:  # RNN
                        # Reshape data for RNN
                        sequence_length = min(5, X_train.shape[1])  # Adjust sequence length
                        n_features = X_train.shape[1] // sequence_length
                        
                        # Reshape the data into sequences
                        X_train_reshaped = X_train.reshape(X_train.shape[0], sequence_length, n_features)
                        X_test_reshaped = X_test.reshape(X_test.shape[0], sequence_length, n_features)
                        
                        model.add(LSTM(neurons_per_layer[0], 
                                     input_shape=(sequence_length, n_features),
                                     activation=activation_functions[0],
                                     return_sequences=num_layers > 1))
                    
                    # Add dropout after first layer
                    model.add(Dropout(dropout_rate))
                    
                    # Hidden layers
                    for i in range(1, num_layers-1):
                        if selected_model == "Convolutional Neural Network (CNN)":
                            model.add(Conv1D(filters=neurons_per_layer[i], 
                                           kernel_size=3, 
                                           activation=activation_functions[i]))
                        elif selected_model == "Recurrent Neural Network (RNN)":
                            model.add(LSTM(neurons_per_layer[i], 
                                         activation=activation_functions[i],
                                         return_sequences=i < num_layers-2))
                        else:
                            model.add(Dense(neurons_per_layer[i], 
                                          activation=activation_functions[i]))
                        
                        model.add(Dropout(dropout_rate))
                    
                    # Add Flatten layer for CNN before final dense layer
                    if selected_model == "Convolutional Neural Network (CNN)":
                        model.add(Flatten())
                    
                    # Output layer configuration based on problem type
                    if problem_type == "binary":
                        model.add(Dense(1, activation='sigmoid'))
                    elif problem_type == "multiclass":
                        model.add(Dense(n_classes, activation='softmax'))
                    else:  # regression
                        model.add(Dense(1))
                    
                    # Compile model
                    optimizer = optimizer_options[optimizer_choice](learning_rate=learning_rate)
                    model.compile(optimizer=optimizer,
                                loss=loss_function,
                                metrics=['accuracy'] if problem_type != "regression" else ['mae', 'mse'])
                    
                    # Prepare data
                    if selected_model == "Convolutional Neural Network (CNN)":
                        X_train_final = X_train_reshaped
                        X_test_final = X_test_reshaped
                    elif selected_model == "Recurrent Neural Network (RNN)":
                        # Use the reshaped data for RNN
                        X_train_final = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                        X_test_final = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                    else:
                        X_train_final = X_train
                        X_test_final = X_test
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Custom callback to update progress
                    class ProgressCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Training Progress: {(progress * 100):.1f}%")
                    
                    # Train model
                    history = model.fit(X_train_final, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=0.2,
                                      callbacks=[ProgressCallback()],
                                      verbose=0)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Plot training history
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Plot loss with adjusted figure size
                        st.markdown("##### Training and Validation Loss")
                        fig_loss, ax_loss = plt.subplots(figsize=(5, 3))  # Smaller figure size
                        ax_loss.plot(history.history['loss'], label='Training Loss')
                        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
                        ax_loss.set_xlabel('Epoch', fontsize=8)
                        ax_loss.set_ylabel('Loss', fontsize=8)
                        ax_loss.set_title('Training and Validation Loss', fontsize=10)
                        ax_loss.legend(fontsize=8)
                        ax_loss.tick_params(axis='both', which='major', labelsize=7)
                        plt.tight_layout()
                        st.pyplot(fig_loss)
                    
                    with col2:
                        # Plot accuracy or MAE based on problem type with adjusted figure size
                        metric_key = 'accuracy' if problem_type != "regression" else 'mae'
                        metric_name = 'Accuracy' if problem_type != "regression" else 'Mean Absolute Error'
                        
                        st.markdown(f"##### Training and Validation {metric_name}")
                        fig_metric, ax_metric = plt.subplots(figsize=(5, 3))  # Smaller figure size
                        ax_metric.plot(history.history[metric_key], label=f'Training {metric_name}')
                        ax_metric.plot(history.history[f'val_{metric_key}'], label=f'Validation {metric_name}')
                        ax_metric.set_xlabel('Epoch', fontsize=8)
                        ax_metric.set_ylabel(metric_name, fontsize=8)
                        ax_metric.set_title(f'Training and Validation {metric_name}', fontsize=10)
                        ax_metric.legend(fontsize=8)
                        ax_metric.tick_params(axis='both', which='major', labelsize=7)
                        plt.tight_layout()
                        st.pyplot(fig_metric)
                    
                    # Store the model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = selected_model
                    
                    # Save and create download button for the model
                    save_and_download_model(model, model_name=selected_model)
                    
                    st.success("Model training completed successfully!")
                
                except Exception as e:
                    st.error(f"An error occurred during model training: {str(e)}")
                    st.info("This might be due to incompatible data types or invalid model configuration.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure your data is properly preprocessed and try again.")


def evaluate_section():
    """Model Evaluation Section"""
    st.subheader("🔍 Model Evaluation")
    
    if st.session_state.trained_model is None:
        st.warning("Please train a model first!")
        return
    
    if not hasattr(st.session_state, "processed_data"):
        st.error("Please preprocess your data first!")
        return
    
    X_train, X_test, y_train, y_test = st.session_state.processed_data
    model = st.session_state.trained_model
    model_type = st.session_state.model_type
    
    st.info(f"Currently evaluating: {model_type}")
    
    # Determine if it's a deep learning model
    is_dl_model = isinstance(model, tf.keras.Model)
    
    try:
        # Create tabs for different evaluation aspects
        eval_tabs = st.tabs(["Model Performance", "Feature Analysis", "Predictions"])
        
        with eval_tabs[0]:
            st.markdown("### Model Performance")
            
            # Get predictions
            if is_dl_model:
                # Handle data reshaping for CNN and RNN
                if "CNN" in model_type:
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_pred = model.predict(X_test_reshaped)
                elif "RNN" in model_type:
                    sequence_length = min(5, X_test.shape[1])
                    n_features = X_test.shape[1] // sequence_length
                    X_test_reshaped = X_test.reshape(X_test.shape[0], sequence_length, n_features)
                    y_pred = model.predict(X_test_reshaped)
                else:
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Determine problem type
            if isinstance(y_test, pd.Series):
                y_test_values = y_test.values
            else:
                y_test_values = y_test
                
            # Handle reshape for numpy arrays only
            if isinstance(y_test_values, np.ndarray) and len(y_test_values.shape) > 1:
                unique_classes = np.unique(y_test_values.reshape(-1))
            else:
                unique_classes = np.unique(y_test_values)
                
            n_classes = len(unique_classes)
            
            if n_classes <= 2:  # Binary classification or regression
                if n_classes == 2:  # Binary classification
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred.round())
                    precision = precision_score(y_test, y_pred.round())
                    recall = recall_score(y_test, y_pred.round())
                    f1 = f1_score(y_test, y_pred.round())
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.metric("Precision", f"{precision:.4f}")
                    with col2:
                        st.metric("Recall", f"{recall:.4f}")
                        st.metric("F1-Score", f"{f1:.4f}")
                    
                    # Plot ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred)
                    
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 3))  # Smaller figure size
                    ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax_roc.plot([0, 1], [0, 1], 'k--')
                    ax_roc.set_xlabel('False Positive Rate', fontsize=8)
                    ax_roc.set_ylabel('True Positive Rate', fontsize=8)
                    ax_roc.set_title('ROC Curve', fontsize=10)
                    ax_roc.legend(loc='lower right', fontsize=8)
                    ax_roc.tick_params(axis='both', which='major', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig_roc)
                    
                    # Plot confusion matrix
                    cm = confusion_matrix(y_test, y_pred.round())
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 3))  # Smaller figure size
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm, cmap='Blues', annot_kws={"size": 8})
                    ax_cm.set_title('Confusion Matrix', fontsize=10)
                    ax_cm.set_ylabel('True Label', fontsize=8)
                    ax_cm.set_xlabel('Predicted Label', fontsize=8)
                    ax_cm.tick_params(axis='both', which='major', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig_cm)
                
                else:  # Regression
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                        st.metric("Root Mean Squared Error", f"{rmse:.4f}")
                    with col2:
                        st.metric("Mean Absolute Error", f"{mae:.4f}")
                        st.metric("R-squared Score", f"{r2:.4f}")
                    
                    # Plot actual vs predicted
                    fig_scatter, ax_scatter = plt.subplots(figsize=(5, 3))  # Smaller figure size
                    ax_scatter.scatter(y_test, y_pred, alpha=0.5, s=20)  # Smaller point size
                    ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
                    ax_scatter.set_xlabel('Actual Values', fontsize=8)
                    ax_scatter.set_ylabel('Predicted Values', fontsize=8)
                    ax_scatter.set_title('Actual vs Predicted Values', fontsize=10)
                    ax_scatter.tick_params(axis='both', which='major', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig_scatter)
                    
                    # Plot residuals
                    residuals = y_test - y_pred
                    fig_resid, ax_resid = plt.subplots(figsize=(5, 3))  # Smaller figure size
                    ax_resid.scatter(y_pred, residuals, alpha=0.5, s=20)  # Smaller point size
                    ax_resid.axhline(y=0, color='r', linestyle='--', lw=1)
                    ax_resid.set_xlabel('Predicted Values', fontsize=8)
                    ax_resid.set_ylabel('Residuals', fontsize=8)
                    ax_resid.set_title('Residual Plot', fontsize=10)
                    ax_resid.tick_params(axis='both', which='major', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig_resid)
            
            else:  # Multiclass classification
                # Convert predictions to class labels if necessary
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred_classes = np.argmax(y_pred, axis=1)
                else:
                    y_pred_classes = y_pred
                
                if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                    y_test_classes = np.argmax(y_test, axis=1)
                else:
                    y_test_classes = y_test
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_classes, y_pred_classes)
                precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
                recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
                f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("Precision", f"{precision:.4f}")
                with col2:
                    st.metric("Recall", f"{recall:.4f}")
                    st.metric("F1-Score", f"{f1:.4f}")
                
                # Plot confusion matrix
                cm = confusion_matrix(y_test_classes, y_pred_classes)
                fig_cm, ax_cm = plt.subplots()  # Use default figure size
                sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm, cmap='Blues')
                ax_cm.set_title('Confusion Matrix')
                ax_cm.set_ylabel('True Label')
                ax_cm.set_xlabel('Predicted Label')
                plt.tight_layout()
                st.pyplot(fig_cm)
                
                # Display classification report
                st.markdown("#### Classification Report")
                report = classification_report(y_test_classes, y_pred_classes)
                st.code(report)
        
        with eval_tabs[1]:
            st.markdown("### Feature Analysis")
            
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importances = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
                # Use x and y parameters explicitly for barplot
                sns.barplot(x=importances['Importance'].head(15), y=importances['Feature'].head(15), ax=ax_imp)
                ax_imp.set_title('Feature Importance', fontsize=10)
                ax_imp.set_xlabel('Importance Score', fontsize=8)
                ax_imp.set_ylabel('Feature Name', fontsize=8)
                ax_imp.tick_params(axis='both', which='major', labelsize=7)
                plt.tight_layout()
                st.pyplot(fig_imp)
                
                st.markdown("#### Feature Importance Table")
                st.dataframe(importances)
            
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(model.coef_.shape) == 1:
                    # For binary classification or regression (1D coefficients)
                    coefficients = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Coefficient': model.coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    fig_coef, ax_coef = plt.subplots(figsize=(5, 4))
                    # Use x and y parameters explicitly for barplot
                    sns.barplot(x=coefficients['Coefficient'].head(15), y=coefficients['Feature'].head(15), ax=ax_coef)
                    ax_coef.set_title('Feature Coefficients', fontsize=10)
                    ax_coef.set_xlabel('Coefficient Value', fontsize=8)
                    ax_coef.set_ylabel('Feature Name', fontsize=8)
                    ax_coef.tick_params(axis='both', which='major', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig_coef)
                    
                else:
                    # For multiclass classification (2D coefficients)
                    st.write("Multiclass Coefficients (showing first class):")
                    
                    # Create DataFrame for the first class coefficients
                    first_class_coef = model.coef_[0] if model.coef_.shape[0] > 0 else model.coef_
                    coefficients = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Coefficient': first_class_coef
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    fig_coef, ax_coef = plt.subplots(figsize=(5, 4))
                    # Use x and y parameters explicitly for barplot
                    sns.barplot(x=coefficients['Coefficient'].head(15), y=coefficients['Feature'].head(15), ax=ax_coef)
                    ax_coef.set_title('Feature Coefficients (Class 0)', fontsize=10)
                    ax_coef.set_xlabel('Coefficient Value', fontsize=8)
                    ax_coef.set_ylabel('Feature Name', fontsize=8)
                    ax_coef.tick_params(axis='both', which='major', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig_coef)
                
                st.markdown("#### Feature Coefficients Table")
                st.dataframe(coefficients)
            
            elif is_dl_model:
                st.info("Feature importance visualization is not available for this type of deep learning model.")
            
            else:
                st.info("Feature importance visualization is not available for this type of model.")
        
        with eval_tabs[2]:
            st.markdown("### Make Predictions")
            
            # Create input fields for each feature
            st.markdown("#### Enter feature values for prediction:")
            
            input_values = {}
            for feature in X_train.columns:
                min_val = float(X_train[feature].min())
                max_val = float(X_train[feature].max())
                mean_val = float(X_train[feature].mean())
                
                input_values[feature] = st.number_input(
                    f"{feature}",
                    value=mean_val,
                    min_value=min_val,
                    max_value=max_val,
                    help=f"Range: [{min_val:.2f}, {max_val:.2f}]"
                )
            
            if st.button("Make Prediction"):
                try:
                    # Create input array
                    input_array = np.array([[input_values[feature] for feature in X_train.columns]])
                    
                    # Make prediction
                    if is_dl_model:
                        if "CNN" in model_type:
                            input_array = input_array.reshape(1, input_array.shape[1], 1)
                        elif "RNN" in model_type:
                            sequence_length = min(5, input_array.shape[1])
                            n_features = input_array.shape[1] // sequence_length
                            input_array = input_array.reshape(1, sequence_length, n_features)
                    
                    prediction = model.predict(input_array)
                    
                    st.markdown("#### Prediction Result:")
                    if n_classes == 2:  # Binary classification
                        prob = prediction[0]
                        pred_class = "Positive" if prob >= 0.5 else "Negative"
                        st.metric("Predicted Class", pred_class)
                        st.metric("Probability", f"{prob[0]:.4f}" if isinstance(prob, np.ndarray) else f"{prob:.4f}")
                    
                    elif n_classes > 2:  # Multiclass classification
                        if len(prediction.shape) > 1:
                            pred_class = np.argmax(prediction[0])
                            probabilities = prediction[0]
                        else:
                            pred_class = prediction[0]
                            probabilities = None
                        
                        st.metric("Predicted Class", str(pred_class))
                        
                        if probabilities is not None:
                            st.markdown("#### Class Probabilities:")
                            for i, prob in enumerate(probabilities):
                                st.metric(f"Class {i}", f"{prob:.4f}")
                    
                    else:  # Regression
                        st.metric("Predicted Value", f"{prediction[0]:.4f}")
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.info("Please check your input values and ensure they are within reasonable ranges.")
    
    except Exception as e:
        st.error(f"An error occurred during evaluation: {str(e)}")
        st.info("This might be due to incompatible data types or model configuration.")

def report_section():
    """Generate a comprehensive report of the analysis and model performance"""
    st.markdown("## 📊 Analysis Report")
    
    if st.session_state.df is None:
        st.warning("Please load and process data first!")
        return
    
    if st.session_state.trained_model is None:
        st.warning("Please train a model first!")
        return
    
    if not hasattr(st.session_state, "processed_data"):
        st.error("Please preprocess your data first!")
        return
    
    try:
        # Create tabs for different report sections
        report_tabs = st.tabs(["Data Summary", "Model Performance", "Export Report"])
        
        with report_tabs[0]:
            st.markdown("### 📈 Data Summary")
            
            # Dataset overview
            st.markdown("#### Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{len(st.session_state.df):,}")
            with col2:
                st.metric("Features", f"{len(st.session_state.df.columns) - 1:,}")
            with col3:
                missing_values = st.session_state.df.isnull().sum().sum()
                st.metric("Missing Values", f"{missing_values:,}")
            
            # Data types summary
            st.markdown("#### Data Types Summary")
            fig_dtype, ax_dtype = plt.subplots(figsize=(5, 3))  # Smaller figure size
            dtype_counts = st.session_state.df.dtypes.value_counts()
            dtype_counts.plot(kind='bar', ax=ax_dtype)
            ax_dtype.set_title("Distribution of Data Types", fontsize=10)
            ax_dtype.set_xlabel("Data Type", fontsize=8)
            ax_dtype.set_ylabel("Count", fontsize=8)
            ax_dtype.tick_params(axis='both', which='major', labelsize=7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_dtype)
            
            # Feature statistics
            st.markdown("#### Feature Statistics")
            stats_df = st.session_state.df.describe(include='all').T
            stats_df['Missing'] = st.session_state.df.isnull().sum()
            stats_df['Missing %'] = (stats_df['Missing'] / len(st.session_state.df) * 100).round(2)
            st.dataframe(stats_df)
        
        with report_tabs[1]:
            st.markdown("### 🎯 Model Performance")
            
            X_train, X_test, y_train, y_test = st.session_state.processed_data
            model = st.session_state.trained_model
            model_type = st.session_state.model_type
            
            # Model information
            st.markdown("#### Model Information")
            st.info(f"Model Type: {model_type}")
            
            # Determine if it's a deep learning model
            is_dl_model = isinstance(model, tf.keras.Model)
            
            if is_dl_model:
                st.markdown("#### Model Architecture")
                # Get model summary
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                model_summary = "\n".join(stringlist)
                st.code(model_summary)
            
            # Get predictions
            if is_dl_model:
                if "CNN" in model_type:
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_pred = model.predict(X_test_reshaped)
                elif "RNN" in model_type:
                    sequence_length = min(5, X_test.shape[1])
                    n_features = X_test.shape[1] // sequence_length
                    X_test_reshaped = X_test.reshape(X_test.shape[0], sequence_length, n_features)
                    y_pred = model.predict(X_test_reshaped)
                else:
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Determine problem type
            if isinstance(y_test, pd.Series):
                y_test_values = y_test.values
            else:
                y_test_values = y_test
            
            # Handle reshape for numpy arrays only
            if isinstance(y_test_values, np.ndarray) and len(y_test_values.shape) > 1:
                unique_classes = np.unique(y_test_values.reshape(-1))
            else:
                unique_classes = np.unique(y_test_values)
                
            n_classes = len(unique_classes)
            
            # Performance metrics
            st.markdown("#### Performance Metrics")
            
            if n_classes <= 2:  # Binary classification or regression
                if n_classes == 2:  # Binary classification
                    metrics_dict = {
                        "Accuracy": accuracy_score(y_test, y_pred.round()),
                        "Precision": precision_score(y_test, y_pred.round()),
                        "Recall": recall_score(y_test, y_pred.round()),
                        "F1-Score": f1_score(y_test, y_pred.round())
                    }
                    
                    # Display metrics in columns
                    cols = st.columns(len(metrics_dict))
                    for col, (metric_name, value) in zip(cols, metrics_dict.items()):
                        with col:
                            st.metric(metric_name, f"{value:.4f}")
                    
                    # ROC curve
                    st.markdown("#### ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred)
                    
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 3))  # Smaller figure size
                    ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax_roc.plot([0, 1], [0, 1], 'k--')
                    ax_roc.set_xlabel('False Positive Rate', fontsize=8)
                    ax_roc.set_ylabel('True Positive Rate', fontsize=8)
                    ax_roc.set_title('ROC Curve', fontsize=10)
                    ax_roc.legend(loc='lower right', fontsize=8)
                    ax_roc.tick_params(axis='both', which='major', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig_roc)
                
                else:  # Regression
                    metrics_dict = {
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "R²": r2_score(y_test, y_pred)
                    }
                    
                    # Display metrics in columns
                    cols = st.columns(len(metrics_dict))
                    for col, (metric_name, value) in zip(cols, metrics_dict.items()):
                        with col:
                            st.metric(metric_name, f"{value:.4f}")
                    
                    # Scatter plot
                    st.markdown("#### Prediction vs Actual")
                    fig_scatter, ax_scatter = plt.subplots(figsize=(5, 3))  # Smaller figure size
                    ax_scatter.scatter(y_test, y_pred, alpha=0.5, s=20)  # Smaller point size
                    ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
                    ax_scatter.set_xlabel('Actual Values', fontsize=8)
                    ax_scatter.set_ylabel('Predicted Values', fontsize=8)
                    ax_scatter.set_title('Actual vs Predicted Values', fontsize=10)
                    ax_scatter.tick_params(axis='both', which='major', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig_scatter)
            
            else:  # Multiclass classification
                if len(y_pred.shape) > 1:
                    y_pred_classes = np.argmax(y_pred, axis=1)
                else:
                    y_pred_classes = y_pred
                
                if len(y_test.shape) > 1:
                    y_test_classes = np.argmax(y_test, axis=1)
                else:
                    y_test_classes = y_test
                
                metrics_dict = {
                    "Accuracy": accuracy_score(y_test_classes, y_pred_classes),
                    "Macro F1": f1_score(y_test_classes, y_pred_classes, average='macro'),
                    "Weighted F1": f1_score(y_test_classes, y_pred_classes, average='weighted')
                }
                
                # Display metrics in columns
                cols = st.columns(len(metrics_dict))
                for col, (metric_name, value) in zip(cols, metrics_dict.items()):
                    with col:
                        st.metric(metric_name, f"{value:.4f}")
                
                # Classification report
                st.markdown("#### Detailed Classification Report")
                report = classification_report(y_test_classes, y_pred_classes)
                st.code(report)
        
        with report_tabs[2]:
            st.markdown("### 📥 Export Report")
            
            # Create report dictionary
            report_dict = {
                "Dataset_Info": {
                    "Total_Samples": len(st.session_state.df),
                    "Features": len(st.session_state.df.columns) - 1,
                    "Missing_Values": int(st.session_state.df.isnull().sum().sum())
                },
                "Model_Info": {
                    "Model_Type": model_type,
                    "Problem_Type": "Classification" if n_classes > 1 else "Regression"
                },
                "Performance_Metrics": metrics_dict
            }
            
            # Convert to JSON
            report_json = json.dumps(report_dict, indent=4)
            
            # Create timestamp for unique filenames
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            random_id = np.random.randint(10000, 99999)
            
            # Create download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON report
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"ml_report_{timestamp}.json",
                    mime="application/json",
                    key=f"json_report_download_{timestamp}_{random_id}"
                )
            
            with col2:
                # PDF report
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    from reportlab.lib.styles import getSampleStyleSheet
                    from reportlab.lib import colors
                    import io
                    
                    # Create PDF buffer
                    pdf_buffer = io.BytesIO()
                    
                    # Create PDF document
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    elements = []
                    
                    # Add title
                    title_style = styles['Heading1']
                    elements.append(Paragraph(f"ML-XPERT Analysis Report - {timestamp}", title_style))
                    elements.append(Spacer(1, 12))
                    
                    # Add dataset info
                    elements.append(Paragraph("Dataset Information", styles['Heading2']))
                    elements.append(Spacer(1, 6))
                    
                    dataset_data = [
                        ["Total Samples", f"{report_dict['Dataset_Info']['Total_Samples']}"],
                        ["Features", f"{report_dict['Dataset_Info']['Features']}"],
                        ["Missing Values", f"{report_dict['Dataset_Info']['Missing_Values']}"]
                    ]
                    
                    dataset_table = Table(dataset_data, colWidths=[200, 200])
                    dataset_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(dataset_table)
                    elements.append(Spacer(1, 12))
                    
                    # Add model info
                    elements.append(Paragraph("Model Information", styles['Heading2']))
                    elements.append(Spacer(1, 6))
                    
                    model_data = [
                        ["Model Type", f"{report_dict['Model_Info']['Model_Type']}"],
                        ["Problem Type", f"{report_dict['Model_Info']['Problem_Type']}"]
                    ]
                    
                    model_table = Table(model_data, colWidths=[200, 200])
                    model_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(model_table)
                    elements.append(Spacer(1, 12))
                    
                    # Add performance metrics
                    elements.append(Paragraph("Performance Metrics", styles['Heading2']))
                    elements.append(Spacer(1, 6))
                    
                    metrics_data = [["Metric", "Value"]]
                    for metric, value in report_dict['Performance_Metrics'].items():
                        metrics_data.append([metric, f"{value:.4f}"])
                    
                    metrics_table = Table(metrics_data, colWidths=[200, 200])
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(metrics_table)
                    
                    # Build PDF
                    doc.build(elements)
                    
                    # Create download button for PDF
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"ml_report_{timestamp}.pdf",
                        mime="application/pdf",
                        key=f"pdf_report_download_{timestamp}_{random_id}"
                    )
                except Exception as e:
                    st.error(f"Could not generate PDF report: {str(e)}")
                    st.info("Please install ReportLab library with 'pip install reportlab' to enable PDF export.")
            
            with col3:
                # Model download
                if is_dl_model:
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                            tmp_path = tmp_model.name
                            # Close the file before saving to it
                            tmp_model.close()
                            
                            # Save the model
                            model.save(tmp_path)
                            
                            # Read the saved model
                            with open(tmp_path, 'rb') as f:
                                model_bytes = f.read()
                                
                            # Create download button
                            st.download_button(
                                label="Download Model",
                                data=model_bytes,
                                file_name=f"model_{timestamp}.keras",
                                mime="application/octet-stream",
                                key=f"report_model_download_dl_{timestamp}_{random_id}"
                            )
                            
                            # Clean up
                            try:
                                os.unlink(tmp_path)
                            except Exception as e:
                                st.warning(f"Warning: Could not delete temporary file: {str(e)}")
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
                else:
                    # For sklearn models
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_model:
                            tmp_path = tmp_model.name
                            # Close the file before saving to it
                            tmp_model.close()
                            
                            # Save the model
                            joblib.dump(model, tmp_path)
                            
                            # Read the saved model
                            with open(tmp_path, 'rb') as f:
                                model_bytes = f.read()
                                
                            # Create download button
                            st.download_button(
                                label="Download Model",
                                data=model_bytes,
                                file_name=f"model_{timestamp}.pkl",
                                mime="application/octet-stream",
                                key=f"report_model_download_ml_{timestamp}_{random_id}"
                            )
                            
                            # Clean up
                            try:
                                os.unlink(tmp_path)
                            except Exception as e:
                                st.warning(f"Warning: Could not delete temporary file: {str(e)}")
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
    
    except Exception as e:
        st.error(f"An error occurred while generating the report: {str(e)}")
        st.info("This might be due to incompatible data types or model configuration.")

# Run Functions
if selected_section == "Data Loading":
    upload_data()
elif selected_section == "Data Processing":
    result = preprocess_data()
    if result is not None:  # Check if preprocess_data returned values
        df, X_train, X_test, y_train, y_test = result
        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            st.session_state.processed_data = (X_train, X_test, y_train, y_test)
elif selected_section == "EDA":
    eda_section()
elif selected_section == "ML Model Training":
    if hasattr(st.session_state, "processed_data"):
        X_train, X_test, y_train, y_test = st.session_state.processed_data
        model_training_section(X_train, X_test, y_train, y_test)
    else:
        st.warning(
            "Please process the data first in the 'Data Processing' section to train ML models."
        )
elif selected_section == "DL Model Training":
    if hasattr(st.session_state, "processed_data"):
        X_train, X_test, y_train, y_test = st.session_state.processed_data
        dl_model_training_section(X_train, X_test, y_train, y_test)
    else:
        st.warning("Please process the data first in the 'Data Processing' section to train DL models.")
elif selected_section == "Evaluate":
    evaluate_section()
elif selected_section == "Report":
    report_section()
elif selected_section == "About Me":
    st.markdown("## About Me")
    st.markdown("""
    ### ML-XPERT: Your Machine Learning Companion
    
    ML-XPERT is a comprehensive machine learning application designed to make the data science workflow more accessible and efficient. It provides a user-friendly interface for:
    
    - 📊 Data Loading and Processing
    - 📈 Exploratory Data Analysis (EDA)
    - 🤖 Machine Learning Model Training
    - 🧠 Deep Learning Model Training
    - 🔍 Model Evaluation
    - 📑 Automated Reporting
    
    #### Features
    
    - Support for various data formats
    - Automated data preprocessing
    - Interactive visualizations
    - Multiple ML and DL models
    - Comprehensive model evaluation
    - Exportable reports and models
    
    #### How to Use
    
    1. Start by uploading your dataset in the Data Loading section
    2. Process your data using the Data Processing tools
    3. Explore your data with the EDA section
    4. Train models using either ML or DL sections
    5. Evaluate your models' performance
    6. Generate comprehensive reports
    
    #### Contact
    
    For questions, suggestions, or collaborations, please reach out through:
    - 📧 Email: your.email@example.com
    - 🌐 GitHub: github.com/yourusername
    - 💼 LinkedIn: linkedin.com/in/yourprofile
    """)

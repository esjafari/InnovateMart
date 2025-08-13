import os
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import numpy as np


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load simulated sales data from CSV and convert specific columns to categorical types.

    Returns:
        pd.DataFrame: DataFrame containing sales data with appropriate types.
    """
    data_path = os.path.join("data", "sales_data.csv")
    df = pd.read_csv(data_path, parse_dates=["Date"])

    # Convert specified columns to categorical type for model compatibility
    categorical_columns = [
        'store_id',
        'store_size',
        'promotion_active',
        'day_of_week',
        'month',
        'is_weekend'
    ]

    for col in categorical_columns:
        df[col] = df[col].astype(str).astype('category')

    return df


@st.cache_resource
def load_model() -> tuple[TemporalFusionTransformer, TimeSeriesDataSet]:
    """
    Load the pre-trained Temporal Fusion Transformer model and training dataset.

    Returns:
        model (TemporalFusionTransformer): The trained TFT model set to evaluation mode.
        training_dataset (TimeSeriesDataSet): The dataset object used during training.
    """
    models_dir = "models"
    training_path = os.path.join(models_dir, "training_dataset.pth")
    weights_path = os.path.join(models_dir, "tft_weights.pth")

    training_dataset = torch.load(training_path, weights_only=False)

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.2,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Load trained weights and switch model to evaluation mode
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    return model, training_dataset


def plot_actual_sales(df_store: pd.DataFrame) -> plt.Figure:
    """
    Plot historical daily sales for a single store.

    Args:
        df_store (pd.DataFrame): Filtered DataFrame for a single store sorted by Date.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object with the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_store["Date"], df_store["daily_sales"],
            label="Actual Sales",
            marker='o',
            linestyle='-')
    ax.set_title(f"Historical Daily Sales for Store {df_store['store_id'].iloc[0]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Sales")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_predictions(
    df_sales: pd.DataFrame,
    training_dataset: TimeSeriesDataSet,
    model: TemporalFusionTransformer,
    selected_store: str
) -> tuple[plt.Figure | None, dict | None]:
    """
    Plot actual vs predicted sales for selected store on validation data (last 30 days).

    Args:
        df_sales (pd.DataFrame): Entire sales DataFrame.
        training_dataset (TimeSeriesDataSet): Training dataset metadata.
        model (TemporalFusionTransformer): Pretrained TFT model.
        selected_store (str): Selected store_id from dropdown.

    Returns:
        Tuple containing:
            - Matplotlib Figure or None (if error)
            - Raw model predictions dictionary or None
    """
    # Ensure proper categorical types for prediction
    categorical_cols = [
        'store_id',
        'store_size',
        'promotion_active',
        'day_of_week',
        'month',
        'is_weekend'
    ]

    for col in categorical_cols:
        df_sales[col] = df_sales[col].astype(str).astype('category')

    # Filter validation data (last 30 days)
    val_threshold = df_sales["time_idx"].max() - 30
    df_val = df_sales[df_sales["time_idx"] > val_threshold]

    # Prepare dataset for prediction (predict=True for creating decoder timesteps)
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df_sales,
        predict=True,
        stop_randomization=True,
    )

    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=3
    )

    # Predict raw outputs from the model
    preds_raw, x, *rest = model.predict(val_dataloader, mode="raw", return_x=True)

    # Check keys presence
    if 'groups' not in x or 'decoder_time_idx' not in x:
        st.error("Prediction batch output missing required keys.")
        return None, None

    # Extract necessary arrays
    groups = x['groups'].detach().cpu().numpy().flatten()
    decoder_time_idx = x['decoder_time_idx'].detach().cpu().numpy()
    preds = preds_raw["prediction"].detach().cpu().numpy()
    median_pred = preds[:, :, 3]  # Median quantile index (0.5 quantile)

    min_date = df_sales['Date'].min()
    store_categories = df_sales['store_id'].cat.categories

    records = []
    for i in range(len(groups)):
        store_id_cat = store_categories[groups[i]]
        if store_id_cat != selected_store:
            continue
        for j in range(median_pred.shape[1]):
            prediction_date = min_date + pd.Timedelta(days=int(decoder_time_idx[i, j]))
            records.append({
                "store_id": store_id_cat,
                "Date": prediction_date,
                "predicted_sales": median_pred[i, j],
            })

    df_preds = pd.DataFrame(records)
    df_actual = df_val[df_val['store_id'] == selected_store]

    # Plot actual vs predicted sales
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_actual['Date'], df_actual['daily_sales'], label='Actual Sales', marker='o')
    ax.plot(df_preds['Date'], df_preds['predicted_sales'], label='Predicted Sales', marker='x')
    ax.set_title(f"Actual vs Predicted Sales (Validation) for Store {selected_store}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, preds_raw


def plot_variable_importance(
    model: TemporalFusionTransformer,
    raw_predictions: dict
) -> plt.Figure:
    """
    Plot variable importance from the trained TFT model using the interpret_output method.

    Args:
        model (TemporalFusionTransformer): Trained TFT model.
        raw_predictions (dict): Raw output from model.predict(mode="raw").

    Returns:
        matplotlib.figure.Figure: Bar plot of variable importance.
    """
    interpretation = model.interpret_output(raw_predictions, reduction="sum")
    var_importance = interpretation["encoder_variables"]

    # If result is tensor, convert to pandas Series for better plotting
    if not isinstance(var_importance, pd.DataFrame):
        var_importance = pd.Series(var_importance.cpu().numpy(), index=model.encoder_variables)

    var_importance_sorted = var_importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(var_importance_sorted.index, var_importance_sorted.values)
    ax.set_xlabel("Importance")
    ax.set_title("Encoder Variable Importance from TFT Model")
    plt.tight_layout()

    return fig


def main() -> None:
    """
    Main function to run the Streamlit app for InnovateMart sales forecasting.

    It loads data and model, provides UI for selecting stores,
    and displays sales history, predictions, and variable importance.
    """
    st.title("InnovateMart Sales Forecasting with Temporal Fusion Transformer")

    # Load simulated sales dataset and model (cached for performance)
    df_sales = load_data()
    model, training_dataset = load_model()

    # Dropdown for store selection
    store_options = df_sales['store_id'].cat.categories.tolist()
    selected_store = st.selectbox("Select Store:", store_options)

    # Display historical sales plot
    df_store = df_sales[df_sales['store_id'] == selected_store].sort_values("Date")
    st.subheader("Historical Daily Sales")
    fig_actual = plot_actual_sales(df_store)
    st.pyplot(fig_actual)

    # Display actual vs predicted sales on validation data
    st.subheader("Actual vs Predicted Sales (Validation Data)")
    fig_pred, raw_preds = plot_predictions(df_sales, training_dataset, model, selected_store)
    if fig_pred is not None:
        st.pyplot(fig_pred)

    # Display variable importance plot
    st.subheader("Model Input Variable Importance")
    if raw_preds is not None:
        fig_varimp = plot_variable_importance(model, raw_preds)
        st.pyplot(fig_varimp)


if __name__ == "__main__":
    main()

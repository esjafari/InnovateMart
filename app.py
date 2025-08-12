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
def load_data():
    """
    Load simulated sales data and convert categorical columns.
    """
    data_path = os.path.join("data", "sales_data.csv")
    df_sales = pd.read_csv(data_path, parse_dates=["Date"])
    categorical_cols = ['store_id', 'store_size', 'promotion_active', 'day_of_week', 'month', 'is_weekend']
    for col in categorical_cols:
        df_sales[col] = df_sales[col].astype(str).astype('category')
    return df_sales


@st.cache_resource
def load_model():
    """
    Load pre-trained TFT model and training dataset object.
    """
    models_dir = "models"
    training_path = os.path.join(models_dir, "training_dataset.pth")
    weights_path = os.path.join(models_dir, "tft_weights.pth")

    training = torch.load(training_path, weights_only=False)

    model = TemporalFusionTransformer.from_dataset(
        training,
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
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model, training


def plot_actual_sales(df_store):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_store["Date"], df_store["daily_sales"], label="Actual Sales", marker='o', linestyle='-')
    ax.set_title(f"Historical Daily Sales for {df_store['store_id'].iloc[0]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Sales")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_predictions(df_sales, training, model, selected_store):
    categorical_cols = ['store_id', 'store_size', 'promotion_active', 'day_of_week', 'month', 'is_weekend']
    for col in categorical_cols:
        df_sales[col] = df_sales[col].astype(str).astype('category')

    df_val = df_sales[df_sales["time_idx"] > (df_sales["time_idx"].max() - 30)]

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_sales,
        predict=True,
        stop_randomization=True,
    )
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=3)

    preds_raw, x, *rest = model.predict(val_dataloader, mode="raw", return_x=True)
    raw_predictions = preds_raw

    if 'groups' not in x or 'decoder_time_idx' not in x:
        st.error("Missing keys in prediction batch output.")
        return None, None

    groups = x['groups'].detach().cpu().numpy().flatten()
    decoder_time_idx = x['decoder_time_idx'].detach().cpu().numpy()

    preds = preds_raw["prediction"].detach().cpu().numpy()
    median_pred = preds[:, :, 3]  # median quantile

    min_date = df_sales['Date'].min()
    records = []

    store_categories = df_sales['store_id'].cat.categories

    for i in range(len(groups)):
        store_id_cat = store_categories[groups[i]]
        if store_id_cat != selected_store:
            continue
        for j in range(median_pred.shape[1]):
            records.append({
                "store_id": store_id_cat,
                "Date": min_date + pd.Timedelta(days=int(decoder_time_idx[i, j])),
                "predicted_sales": median_pred[i, j],
            })

    df_preds = pd.DataFrame(records)
    df_actual = df_val[df_val['store_id'] == selected_store]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_actual['Date'], df_actual['daily_sales'], label='Actual Sales', marker='o')
    ax.plot(df_preds['Date'], df_preds['predicted_sales'], label='Predicted Sales', marker='x')
    ax.set_title(f"Actual vs Predicted Sales (Validation) for {selected_store}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, raw_predictions


def plot_variable_importance(model, raw_predictions):
    interpretation = model.interpret_output(raw_predictions, reduction="sum")
    var_imp = interpretation["encoder_variables"]

    import pandas as pd
    if not isinstance(var_imp, pd.DataFrame):
        var_imp = pd.Series(var_imp.cpu().numpy(), index=model.encoder_variables)

    var_imp_sorted = var_imp.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(var_imp_sorted.index, var_imp_sorted.values)
    ax.set_xlabel("Importance")
    ax.set_title("Variable Importance from TFT Model (Encoder Variables)")
    plt.tight_layout()

    return fig


def main():
    st.title("InnovateMart Sales Forecasting with Temporal Fusion Transformer")

    df_sales = load_data()
    model, training = load_model()

    store_options = df_sales['store_id'].cat.categories.tolist()
    selected_store = st.selectbox("Select Store:", store_options)

    df_store = df_sales[df_sales['store_id'] == selected_store].sort_values("Date")
    st.subheader("Historical Daily Sales")
    fig1 = plot_actual_sales(df_store)
    st.pyplot(fig1)

    st.subheader("Actual vs Predicted Sales (Validation Data)")
    fig2, raw_predictions = plot_predictions(df_sales, training, model, selected_store)
    if fig2:
        st.pyplot(fig2)

    st.subheader("Model Input Variable Importance")
    if raw_predictions is not None:
        fig3 = plot_variable_importance(model, raw_predictions)
        st.pyplot(fig3)


if __name__ == "__main__":
    main()

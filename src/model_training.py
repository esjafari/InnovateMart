import os
import pandas as pd
import lightning.pytorch as pl
import torch
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from src import data_simulation
from src import data_preparation


def train_model(training, train_dataloader, val_dataloader):
    """
    Train the Temporal Fusion Transformer model.
    
    Args:
        training (TimeSeriesDataSet): Training dataset.
        train_dataloader: Dataloader for training.
        val_dataloader: Dataloader for validation.
    
    Returns:
        tft (TemporalFusionTransformer): Trained model.
        trainer (pl.Trainer): Trainer instance.
    """
    pl.seed_everything(42)

    tft = TemporalFusionTransformer.from_dataset(
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

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=30
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return tft, trainer


def evaluate_model(tft, val_dataloader):
    """
    Predict on validation data in raw mode.
    
    Args:
        tft (TemporalFusionTransformer): Trained model.
        val_dataloader: Validation dataloader.
    
    Returns:
        raw_predictions: Raw predictions from the model.
        x: Input data batch dictionary.
    """
    raw_predictions, x, _, _, _ = tft.predict(val_dataloader, mode="raw", return_x=True)
    return raw_predictions, x


def prepare_predictions_dataframe(raw_predictions, x, df_sales):
    """
    Convert model raw predictions to DataFrame with actual and predicted sales.
    
    Args:
        raw_predictions: Raw output predictions.
        x: Batch inputs dictionary.
        df_sales: Original sales DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with store_id, Date, actual_sales, predicted_sales.
    """
    preds = raw_predictions["prediction"].detach().cpu().numpy()
    median_pred = preds[:, :, 3]  # median quantile (0.5)

    decoder_time_idx = x['decoder_time_idx'].detach().cpu().numpy()
    groups = x['groups'].detach().cpu().numpy().flatten()
    store_categories = df_sales['store_id'].cat.categories

    records = []
    min_date = df_sales['Date'].min()

    # Flatten predictions to long format DataFrame
    for i in range(median_pred.shape[0]):
        store_id_cat = store_categories[groups[i]]
        for j in range(median_pred.shape[1]):
            date = min_date + pd.Timedelta(days=int(decoder_time_idx[i, j]))
            actual = x['decoder_target'][i, j].item() if 'decoder_target' in x else None
            records.append({
                'store_id': store_id_cat,
                'Date': date,
                'predicted_sales': median_pred[i, j],
                'actual_sales': actual
            })

    return pd.DataFrame(records)


def main():
    """
    Simulate data, train model, evaluate, and save outputs.
    """
    print("Simulating sales data...")
    df_sales = data_simulation.simulate_sales_data()
    print(f"Data simulation complete. Rows: {len(df_sales)}")

    # Define directories relative to parent of current file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    predictions_dir = os.path.join(base_dir, "predictions")

    # Create directories if missing
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Save sales data as CSV
    sales_path = os.path.join(data_dir, "sales_data.csv")
    print(f"Saving sales data to '{sales_path}'...")
    df_sales.to_csv(sales_path, index=False)

    print("Preparing datasets...")
    training, validation, train_dl, val_dl = data_preparation.prepare_datasets(df_sales)
    print("Datasets ready.")

    print("Training model...")
    tft, trainer = train_model(training, train_dl, val_dl)
    print("Training complete.")

    print("Evaluating model...")
    raw_preds, x = evaluate_model(tft, val_dl)
    print("Evaluation complete.")

    df_preds = prepare_predictions_dataframe(raw_preds, x, df_sales)

    # Save predictions to CSV
    preds_path = os.path.join(predictions_dir, "predictions.csv")
    print(f"Saving predictions to '{preds_path}'...")
    df_preds.to_csv(preds_path, index=False)

    # Save model weights and training dataset
    weights_path = os.path.join(models_dir, "tft_weights.pth")
    training_ds_path = os.path.join(models_dir, "training_dataset.pth")

    print(f"Saving model weights to '{weights_path}' and training dataset to '{training_ds_path}'...")
    torch.save(tft.state_dict(), weights_path)
    torch.save(training, training_ds_path)
    print("All assets saved.")

    return tft, trainer, df_sales, df_preds


if __name__ == "__main__":
    main()

from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer


def prepare_datasets(df_sales):
    """
    Prepare TimeSeriesDataSet objects and dataloaders for training and validation.

    Args:
        df_sales (pd.DataFrame): Simulated sales data.

    Returns:
        training (TimeSeriesDataSet), validation (TimeSeriesDataSet),
        train_dataloader, val_dataloader
    """
    max_time_idx = df_sales['time_idx'].max()
    training_cutoff = max_time_idx - 30

    training = TimeSeriesDataSet(
        df_sales[df_sales.time_idx <= training_cutoff],
        time_idx='time_idx',
        target='daily_sales',
        group_ids=['store_id'],
        max_encoder_length=60,
        max_prediction_length=30,
        static_categoricals=['store_id', 'store_size'],
        static_reals=['city_population'],
        time_varying_known_categoricals=['promotion_active', 'day_of_week', 'month', 'is_weekend'],
        time_varying_known_reals=['time_idx', 'day_of_month'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=['daily_sales'],
        target_normalizer=GroupNormalizer(groups=['store_id'], transformation='softplus'),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df_sales, predict=True, stop_randomization=True)

    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=3)

    return training, validation, train_dataloader, val_dataloader

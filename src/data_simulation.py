import numpy as np
import pandas as pd


def calculate_growth_rate(store_size, city_population):
    """
    Calculate the annual growth rate for a given store size and city population.

    Args:
        store_size (str): One of 'small', 'medium', or 'large'.
        city_population (int): Population of the city where the store is located.

    Returns:
        float: Growth rate value.
    """
    base_rate = {
        'small': np.random.normal(-0.01, 0.02),
        'medium': np.random.normal(0.03, 0.01),
        'large': np.random.normal(0.05, 0.015)
    }[store_size]
    population_factor = np.log(city_population / 50000) * 0.005
    return base_rate + population_factor

def simulate_sales_data():
    """
    Simulate realistic daily sales data for multiple stores (~3 years).
    Includes seasonal patterns, promotions, external shock, and random noise.

    Returns:
        pd.DataFrame: Sales data in long format with relevant features.
    """
    np.random.seed(42)
    num_stores = 4
    start_date = '2021-01-01'
    end_date = '2023-12-31'
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    store_sizes = ['small', 'medium', 'large']
    city_populations = {'small': 50000, 'medium': 200000, 'large': 1000000}
    shock_store = 'store_2'
    shock_date = pd.to_datetime('2022-06-01')

    # Create store info with growth rates
    stores = []
    for i in range(num_stores):
        size = store_sizes[i % len(store_sizes)]
        population = city_populations[size]
        growth_rate = calculate_growth_rate(size, population)
        stores.append({
            'store_id': f'store_{i+1}',
            'store_size': size,
            'city_population': population,
            'growth_rate': growth_rate
        })

    # Weekday and monthly seasonal factors
    weekday_effects = {
        0: 0.9, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.10, 5: 1.30, 6: 1.30
    }

    monthly_factors = {
        1:1.0, 2:0.9, 3:0.92, 4:1.05, 5:1.1, 6:1.05,
        7:1.0, 8:0.95, 9:1.0, 10:1.0, 11:1.05, 12:1.3
    }

    # Define promotion dates per store
    promotions = {}
    for store in stores:
        promotion_days = []
        for date in date_range:
            if date.weekday() in [4, 5]:  # Friday and Saturday
                promotion_days.append(date)
            if date.month == 11 and date.day in [10, 20, 25]:
                promotion_days.append(date)
            if np.random.random() < 0.1:
                promotion_days.append(date)
        promotions[store['store_id']] = set(promotion_days)

    # Base sales mapping per store size
    records = []
    for store in stores:
        population = store['city_population']
        base_sales = {
            'small': 50 * (population / 50000) ** 0.5,
            'medium': 120 * (population / 200000) ** 0.5,
            'large': 300 * (population / 1000000) ** 0.5
        }
        for current_date in date_range:
            years_passed = (current_date - pd.to_datetime(start_date)).days / 365.25
            sales_base = base_sales[store['store_size']] * (1 + store['growth_rate'] * years_passed)
            sales_base *= weekday_effects[current_date.weekday()]
            sales_base *= monthly_factors[current_date.month]

            promotion_active = int(current_date in promotions[store['store_id']])
            if promotion_active:
                sales_base *= 1.5

            # Apply shock effect if applicable
            if store['store_id'] == shock_store and current_date >= shock_date:
                days_since_shock = (current_date - shock_date).days
                shock_factor = max(0.7, 1 - 0.3 * min(1, days_since_shock / 60))
                sales_base *= shock_factor

            # Add Gaussian noise proportional to sales_base (5%)
            noise = np.random.normal(0, sales_base * 0.05)
            daily_sales = max(10, sales_base + noise)

            records.append({
                'Date': current_date,
                'store_id': store['store_id'],
                'daily_sales': daily_sales,
                'promotion_active': str(promotion_active),
                'store_size': store['store_size'],
                'city_population': store['city_population'],
                'day_of_week': str(current_date.weekday()),
                'month': str(current_date.month),
                'day_of_month': current_date.day,
                'is_weekend': str(int(current_date.weekday() >= 4))
            })

    df_sales = pd.DataFrame(records)
    
    # Convert categorical columns to category dtype
    cat_cols = ['store_id', 'store_size', 'promotion_active', 
                'day_of_week', 'month', 'is_weekend']
    for col in cat_cols:
        df_sales[col] = df_sales[col].astype('category')

    # Create a time index feature for modeling purposes
    df_sales['time_idx'] = (df_sales['Date'] - 
                            df_sales['Date'].min()).dt.days
    
    return df_sales

import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

def fetch_index_data(symbol, period="daily", start_date=None, end_date=None):
    """
    Fetch and process historical index data from akshare for a given symbol.
    
    Parameters:
    - symbol (str): The securities code (e.g., '000016' for SSE 50 Index).
    - period (str): The data period ('daily', 'weekly', 'monthly'). Default is 'daily'.
    - start_date (str): Start date in 'YYYYMMDD' format. Default is 1000 days before current date.
    - end_date (str): End date in 'YYYYMMDD' format. Default is current date.
    
    Returns:
    - pandas.DataFrame: DataFrame with columns ['timestamps', 'open', 'close', 'high', 'low', 'volume', 'amount']
                        containing the most recent 512 rows.
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1000)).strftime("%Y%m%d")

    # Fetch historical data using akshare
    index_zh_a_hist_df = ak.index_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date)

    # Create a new DataFrame with the desired columns
    new_df = index_zh_a_hist_df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']].copy()

    # Rename the columns to the specified English names
    new_df.columns = ['timestamps', 'open', 'close', 'high', 'low', 'volume', 'amount']

    # Convert the timestamps column to datetime format
    new_df['timestamps'] = pd.to_datetime(new_df['timestamps'])

    # Select the most recent 512 rows
    recent_df = new_df.tail(512).reset_index(drop=True)

    return recent_df


def generate_trading_days(start_date, num_days, holidays=None):
    """
    Generate a sequence of future trading days, excluding weekends and specified holidays.
    
    Parameters:
    - start_date (str or datetime): Start date for the sequence (format 'YYYY-MM-DD' or datetime object).
    - num_days (int): Desired number of trading days in the output sequence.
    - holidays (list of str, optional): List of holiday dates in 'YYYY-MM-DD' format to exclude.
                                       Default includes 2025 Chinese stock market holidays (National Day and Mid-Autumn).
    
    Returns:
    - pandas.DatetimeIndex: A sequence of num_days trading days starting from start_date, excluding weekends and holidays.
    """
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Default holidays for 2025 Chinese stock market (National Day + Mid-Autumn Festival)
    if holidays is None:
        holidays = [
            '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05',
            '2025-10-06', '2025-10-07', '2025-10-08', '2026-01-01', '2026-02-16',
            '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20', '2026-04-06',
            '2026-05-01', '2026-05-04', '2026-05-05', '2026-06-19', '2026-09-24'
        ]
    
    # Convert holidays to datetime
    holidays = pd.to_datetime(holidays)
    
    # Generate initial sequence of business days (excludes weekends)
    future_dates = pd.bdate_range(start=start_date, periods=num_days, freq='B')
    
    # Exclude holidays
    future_dates = future_dates[~future_dates.isin(holidays)]
    
    # Supplement additional dates if needed
    while len(future_dates) < num_days:
        # Calculate remaining days needed
        needed = num_days - len(future_dates)
        # Start from the day after the last date in the current sequence
        next_start = future_dates[-1] + timedelta(days=1)
        # Generate additional business days
        extra_dates = pd.bdate_range(start=next_start, periods=needed, freq='B')
        # Exclude holidays from extra dates
        extra_dates = extra_dates[~extra_dates.isin(holidays)]
        # Append and trim to exact length
        future_dates = future_dates.append(extra_dates)[:num_days]
    
    # Ensure exactly num_days
    future_dates = future_dates[:num_days]
    
    return future_dates


def plot_prediction(kline_df, pred_df):
    # Combine close prices
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = 'Prediction'
    close_df = pd.concat([sr_close, sr_pred_close], axis=1)

    # Combine volumes
    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = 'Prediction'
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    # 可选：旋转 x 轴标签以显示日期（如果索引是日期，会自动格式化）
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
# model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 2. Instantiate Predictor
# predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
predictor = KronosPredictor(model, tokenizer, device="xpu", max_context=512)

# 3. Prepare Data
df = fetch_index_data(symbol="000300")

lookback = 512
pred_len = 90

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
# y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
y_timestamp = pd.Series(generate_trading_days(start_date=datetime.now().strftime("%Y-%m-%d"), num_days=pred_len))

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=0.9,
    top_p=0.9,
    sample_count=5,
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
# 先将历史数据的索引设置为 timestamps，使其成为时间序列
kline_df = df.set_index('timestamps') 

# 生成预测的未来时间戳，从历史最后日期的下一天开始
last_historical_date = kline_df.index[-1] + timedelta(days=1)  # 下一天
y_future = generate_trading_days(start_date=last_historical_date.strftime("%Y-%m-%d"), num_days=pred_len)

# 设置 pred_df 的索引为未来时间戳
pred_df.index = y_future

# 创建扩展 DataFrame：concat 历史和预测数据
kline_df_extended = pd.concat([kline_df, pred_df], axis=0, join='outer')

# 更新扩展 DataFrame 中的 close 和 volume（如果有 NaN，会自动填充）
kline_df_extended.loc[pred_df.index, ['close', 'volume']] = pred_df[['close', 'volume']]

# visualize
plot_prediction(kline_df_extended, pred_df)



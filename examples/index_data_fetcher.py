import akshare as ak
import pandas as pd
import os
from datetime import datetime
import logging

class IndexDataFetcher:
    """
    指数数据获取器类，用于获取指定指数的行情数据并保存到本地文件
    """
    
    def __init__(self, data_dir="./data"):
        """
        初始化指数数据获取器
        
        Args:
            data_dir (str): 数据保存目录路径，默认为../data
        """
        self.data_dir = data_dir
        self.logger = self._setup_logger()
        
        # 创建数据目录（如果不存在）
        os.makedirs(data_dir, exist_ok=True)
        self.logger.info(f"数据目录已创建或已存在: {os.path.abspath(data_dir)}")
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('IndexDataFetcher')
        logger.setLevel(logging.INFO)
        
        # 如果还没有处理器，则添加
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return logger
    
    def fetch_index_data(self, symbol, period="daily", start_date="19700101", 
                        end_date=None, rows=None):
        """
        获取指数历史数据
        
        Args:
            symbol (str): 指数代码，如"000016"（上证50）
            period (str): 周期，默认为"daily"（日线），可选"weekly"（周线），"monthly"（月线）
            start_date (str): 开始日期，格式为YYYYMMDD
            end_date (str): 结束日期，格式为YYYYMMDD，默认为当前日期
            rows (int): 获取的数据行数，None表示获取所有数据
            
        Returns:
            pandas.DataFrame: 包含指数行情数据的DataFrame，失败时返回None
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        try:
            self.logger.info(f"开始获取指数 {symbol} 数据，周期: {period}, "
                           f"时间范围: {start_date} 至 {end_date}")
            
            # 使用AKShare获取指数历史数据 [7](@ref)
            df = ak.index_zh_a_hist(
                symbol=symbol, 
                period=period, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if df.empty:
                self.logger.warning(f"未获取到指数 {symbol} 的数据")
                return None
            
            # 重命名列以保持一致性 [7](@ref)
            df = df.rename(columns={
                "日期": "timestamps",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount"
            })
            
            # 选择需要的列
            required_columns = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
            df = df[required_columns]
            
            # 如果指定了行数，则取最后rows行
            if rows is not None and isinstance(rows, int):
                df = df.tail(rows)
            
            self.logger.info(f"成功获取指数 {symbol} 数据，共 {len(df)} 行")
            return df
            
        except Exception as e:
            self.logger.error(f"获取指数 {symbol} 数据时发生错误: {str(e)}")
            return None
    
    def save_to_csv(self, df, symbol):
        """
        将DataFrame保存为CSV文件
        
        Args:
            df (pandas.DataFrame): 要保存的数据框
            symbol (str): 指数代码，用于文件名
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        if df is None or df.empty:
            self.logger.warning("无数据可保存")
            return False
        
        try:
            # 构建文件名
            filename = f"{symbol}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # 保存到CSV文件 [5](@ref)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"数据已保存到: {os.path.abspath(filepath)}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存数据时发生错误: {str(e)}")
            return False
    
    def get_index_data(self, symbol, period="daily", start_date="19700101", 
                      end_date=None, rows=None, save=True):
        """
        获取指数数据并可选地保存到文件
        
        Args:
            symbol (str): 指数代码
            period (str): 数据周期
            start_date (str): 开始日期
            end_date (str): 结束日期
            rows (int): 数据行数
            save (bool): 是否保存到文件
            
        Returns:
            pandas.DataFrame: 指数数据DataFrame
        """
        # 获取数据
        df = self.fetch_index_data(symbol, period, start_date, end_date, rows)
        
        # 保存数据
        if save and df is not None:
            self.save_to_csv(df, symbol)
        
        return df


# 使用示例
if __name__ == "__main__":
    # 创建指数数据获取器实例
    fetcher = IndexDataFetcher()
    
    # 获取上证50指数数据并保存
    symbol = "000016"
    df = fetcher.get_index_data(
        symbol=symbol,
        period="daily",
        start_date="20220101",
        end_date="20250930",
        rows=600,  # 获取最后100行数据
        save=True
    )
    
    # 显示数据前5行
    if df is not None:
        print(f"\n获取到的 {symbol} 指数数据前5行:")
        print(df.head())
        
        print(f"\n数据形状: {df.shape}")
        print(f"\n数据列名: {list(df.columns)}")
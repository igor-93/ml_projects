import numpy as np
import pandas as pd

"""Define functions to compute a bunch of technical indicators.
"""

__all__ = ["sma", 'ewm', "acc_indicator", "macd", "cci", "atr", "roc", "smi", "rsi",
           "wvad", "boll", "boll_percentb", "boll_bandwidth"]


def volatility(df: pd.DataFrame, window):
    df_ret = np.log(df)
    df_ret = pd.DataFrame(df_ret, index=df.index, columns=df.columns).diff()
    return df_ret.rolling(window).std()


def sma(df, window, column=None):
    """Rolling mean
    """
    if isinstance(df, pd.DataFrame):
        return df[column].rolling(window).mean(), 'non-stationary'
    else:
        return df.rolling(window).mean(), 'non-stationary'


def ewm(df, window, column=None):
    """Exponentially weighted rolling mean
    """
    if isinstance(df, pd.DataFrame):
        return df[column].ewm(span=window, min_periods=window).mean(), 'non-stationary'
    else:
        return df.ewm(span=window, min_periods=window).mean(), 'non-stationary'


def acc_indicator(df, recent_sma_win, long_sma_win, ao_sma_win, high_col="High", low_col="Low"):
    """Acceleration indicator.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe. Needs high and low columns.
    recent_sma_win : int
        Short time-window
    long_sma_win : int
        Long time-window
    ao_sma_win : int
        Window for computing the rolling mean of acceleration common.
    high_col : str
        Column name in `df` for high price.
    low_col : str
        Column name in `df` for low price.

    Returns:
    --------
    acc_ts : Series
        The acceleration indicator for each time-point (the first
        `long_sma_win + ao_sma_win - 2` time-points will default to NaN).
    """
    recent_sma_win = int(recent_sma_win)
    long_sma_win = int(long_sma_win)
    ao_sma_win = int(ao_sma_win)

    median_price = (df[high_col] + df[low_col]) / 2.0
    sma_median_5, _ = sma(median_price, recent_sma_win)
    sma_median_34, _ = sma(median_price, long_sma_win)
    ao = sma_median_5 - sma_median_34
    ao_sma_win, _ = sma(ao, ao_sma_win)
    acc_ts = ao - ao_sma_win
    return acc_ts, 'stationary'


def macd(df, ewa_short, ewa_long, ewa_signal, price_col="adj_close"):
    """Moving Average Convergence Divergence
    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    ewa_short : int
        Exponentially weighted average time-window for a short time-span.
        A common choice for the short time-window is 12 intervals.
    ewa_long : int
        Exponentially weighted average time-window for a longer time-span.
        A common choice for the long time-window is 26 intervals.
    ewa_signal : int
        Time-window for the EWA of the difference between long and short
        averages.
    price_col : str
        Column name in `df` used for defining the current indicator (e.g. "open",
        "close", etc.)
    Returns:
    --------
    macd_ts : Series
        Moving average convergence-divergence indicator for the time series.
    """
    ewa_short = int(ewa_short)
    ewa_long = int(ewa_long)
    ewa_signal = int(ewa_signal)
    ewa12 = df[price_col].ewm(span=ewa_short).mean()
    ewa26 = df[price_col].ewm(span=ewa_long).mean()
    macd_ts = ewa12 - ewa26
    signal_line = macd_ts.ewm(span=ewa_signal).mean()
    return macd_ts - signal_line, 'stationary'


def cci(df, window, high_col="High", low_col="Low", price_col="adj_close"):
    """Commodity Channel Index. Kind of momentum indicator.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    window : int
        Length of rolling window for calculation of the index.
    high_col : str
        Column name in `df` for high price.
    low_col : str
        Column name in `df` for low price.
    price_col : str
        Column name in `df` for current (adj_close) price.

    Returns:
    --------
    cci_ts : Series
        A stationary time-series containing the CCI of the input common.

    Notes:
    ------
    The window size should be large enough that some change occurs within it to avoid
    divisions by zero.
    """
    window = int(window)
    typical_price = (df[high_col] + df[low_col] + df[price_col]) / 3.0
    sma_ts, _ = sma(typical_price, window)
    mad = typical_price.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_ts = typical_price - sma_ts
    cci_ts = cci_ts / (0.03 * mad)
    # Division by 0 may occur when mad == 0, we remove the inf values
    cci_ts[np.isinf(cci_ts)] = 0
    # + 100 is overbought, -100 is oversold
    return cci_ts, 'stationary'


def atr(df, window, high_col="High", low_col="Low", price_col="adj_close"):
    """Average True Range, measuring price volatility.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    window : int
        Length of rolling window for calculation of the ATR.
    high_col : str
        Column name in `df` for high price.
    low_col : str
        Column name in `df` for low price.
    price_col : str
        Column name in `df` for current (last) price.

    Returns:
    --------
    atr_appx : Series
        A time-series containing the dataframe of the input common.
    """
    window = int(window)
    m1 = df[high_col] - df[low_col]
    m2 = (df[high_col] - df[price_col].shift()).abs()
    m3 = (df[low_col] - df[price_col].shift()).abs()
    tr = pd.concat([m1, m2, m3], axis=1).min(axis=1)
    atr_appx = tr.rolling(window).mean()
    return atr_appx, 'kind-of-stationary'


def roc(df, window, price_col="adj_close"):
    """Calculate price rate of change. This is the same as percentage change!

    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    window : int
        Length of rolling window for calculation of momentum.
    price_col : str
        Column name in `df` for current (last) price.

    Returns:
    --------
    roc_ts : Series
        A stationary time-series containing the rate of change of input common.
    """
    window = int(window)
    roc_ts = df[price_col].pct_change(periods=window)
    return roc_ts, 'stationary'


def smi(df, window, signal_window, high_col="High", low_col="Low", price_col="adj_close"):
    """Stochastic Momentum Index, similar to William's %R
    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    window : int
        Length of rolling window for calculation of price mean.
    signal_window : int
        Length of rolling window for calculation of the SMI.
    high_col : str
        Column name in `df` for high price.
    low_col : str
        Column name in `df` for low price.
    price_col : str
        Column name in `df` for current (last) price.

    Returns:
    --------
    smi_ts : Series
        A stationary time-series containing the rate of change of input common.

    Notes:
    ------
    Using [http://www.blastchart.com/Community/IndicatorGuide/Indicators/StochasticMomentumIndex.aspx]
    for reference on this indicator.
    """
    window = int(window)
    high = df[high_col].rolling(window).max()
    low = df[low_col].rolling(window).min()
    center = (high + low) / 2.0
    h = df[price_col] - center
    d = (high - low)
    # Smooth using twice EMA
    hs1 = h.ewm(span=signal_window).mean()
    hs2 = hs1.ewm(span=signal_window).mean()
    ds1 = d.ewm(span=signal_window).mean()
    ds2 = ds1.ewm(span=signal_window).mean() / 2
    # Calculate percentage
    smi_ts = hs2 / ds2 * 100

    return smi_ts, 'stationary'


def rsi(df, rsi_window, price_col="adj_close"):
    """Relative Strength Index.
    We use Cutler's RSI (see [https://en.wikipedia.org/wiki/Relative_strength_index])
    formula which uses SMA instead of SMMA.
    The RSI indicates whether an asset is overbought or oversold, with a common
    threshold being 30 for oversold assets and 70 for overbought assets.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    rsi_window : int
        Length of rolling window for calculation of the RSI (on top of the
        resampling).
    price_col : str
        Column name in `df` for current (last) price.

    Returns:
    --------
    rsi_ts : Series
        A stationary time-series containing the RSI of input common.

    Notes:
    ------
    Typically the dataframe passed as input should be resampled to some adequate
    time-scale.
    The `rsi_window` must also be sufficiently long to ensure both
    losses and gains occur within the window. NaNs will occur in regions with no
    price changes if these have not yet been removed from the dataframe.
    """
    pct_change = roc(df, window=1, price_col=price_col)[0]

    # These will raise a warning if no change is detected AT ALL in a full window.
    ave_gain = pct_change.rolling(rsi_window).apply(lambda x: np.mean(x[x > 0]), raw=True)
    ave_loss = pct_change.rolling(rsi_window).apply(lambda x: np.abs(np.mean(x[x < 0])), raw=True)

    rs = ave_gain / ave_loss
    rsi_ts = 100.0 - (100.0 / (1.0 + rs))
    # Avoid issues with time-series where no change is detected (ave_gain, ave_loss will be 0)
    rsi_ts[np.isnan(rsi_ts)] = 0.0

    return rsi_ts, "stationary"


def wvad(df, window, trade_window, high_col="high", low_col="low",
         open_col="open", close_col="adj_close", price_col="adj_close",
         volume_col="volume"):
    """Compute the Williams variable accumulation distribution.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    window : int
        window size for rolling window calculation.
    trade_window : int
        window size for shift length.

    Returns:
    --------
    signal : Series
        The WVAD time-series of the input common. Elements of the series are either
        0, 1 or -1.
    """
    window = int(window)
    high = df[high_col].rolling(window).max()
    low = df[low_col].rolling(window).min()
    o = df[open_col].shift(window)
    volume = df[volume_col].rolling(window).sum()
    wvad_ts = ((df[close_col] - o) / (high - low)) * volume
    wvad_ts = wvad_ts.cumsum()
    price_move = df[price_col] / df[price_col].shift(trade_window) - 1.0
    wvad_move = wvad_ts / wvad_ts.shift(trade_window) - 1.0
    buy = (price_move < 0) & (wvad_move > 0)
    sell = (price_move > 0) & (wvad_move < 0)
    signal = pd.Series(0.0, index=wvad_ts.index)
    signal[buy] = 1.0
    signal[sell] = -1.0
    return signal, 'stationary'


def boll(df, window, K=2, price_col="adj_close"):
    """Bollinger band calculation.
    Provides a relative definition of high and low prices of a market.
    A comprehensive explanation of this indicator is available
    [https://commodity.com/technical-analysis/bollinger-bands/]

    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    window : int
        window size for rolling window calculation. Typical window value for
        the Bollinger band calculation is 20 (see wikipedia)
    K : int
        The size of the bollinger bands around the mean (in standard deviations)
        Wikipedia recommends a value of 2.

    Returns:
    --------
    lower_band : Series
        The lower Bollinger band.
    middle_band : Series
        The middle Bollinger band (simple moving average of the price).
    upper_band : Series
        The upper Bollinger band.
    """
    window = int(window)
    K = int(K)
    # prices normalized around bollinger band
    middle, _ = sma(df[price_col], window=window)
    s = df[price_col].rolling(window).std()
    return middle - (K * s), middle, middle + (K * s)


def boll_percentb(df, window, price_col="adj_close"):
    """%b indicator derived from Bollinger bands. (Igor's weird version)
    This indicator shows where the price is in relation to the bollinger bands,
    %b is equal to 1 when price is at the upper band, equal to -1 at the lower band.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    window : int
        window size for rolling window calculation. Typical window value for
        the Bollinger band calculation is 20 (see wikipedia)

    Returns:
    --------
    percentb_ts : Series
        The Bollinger time-series calculated on the input common. First `window`
        time-points will be NaN.
    """

    lb, mb, ub = boll(df, window, K=2, price_col=price_col)
    percentb_ts = (df[price_col] - mb) / (ub - lb)
    # Division by 0 may occur when ub == lb, we remove the inf values
    percentb_ts[np.isinf(percentb_ts)] = 0

    return percentb_ts, "stationary"


def boll_bandwidth(df, window, price_col="adj_close"):
    """Computes the normalized width of the Bollinger bands.
    This is a way of analyzing volatility.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe.
    window : int
        window size for rolling window calculation. Typical window value for
        the Bollinger band calculation is 20 (see wikipedia)

    Returns:
    --------
    bwidth_ts : Series
        The bandwidth of Bollinger bands.
    """
    lb, mb, ub = boll(df, window, K=2, price_col=price_col)
    bwidth_ts = (ub - lb) / mb

    return bwidth_ts, "stationary"


import pytz
from datetime import datetime

import numpy as np
import pandas as pd


def infer_freq(index, return_if_missing=False):
    """Infer the frequency of the DataFrame

    Parameters
    ----------
    index : DateTimeIndex
        index to infer the frequency from
    return_if_missing : bool, optional
        if True, function returns boolean that is True if there were missing common

    Returns
    -------
    resolution : str
        resolution of the index, .e.g T for minute, S for seconds etc.
    missing : bool
        it is True if there are missing rows in index. It is returned only if return_if_missing is
        True

    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError('index must be pd.DatetimeIndex, not ', type(index))
    difference = np.diff(index.values)
    uniq, counts = np.unique(difference, return_counts=True)
    if len(uniq) > 1:
        missing = True
    else:
        missing = False
    most_common_interval = uniq[np.argmax(counts)]
    dt = pd.to_timedelta(most_common_interval)

    if return_if_missing:
        return dt.resolution, missing
    else:
        return dt.resolution


def is_dst(time, tz):
    """
    The function is used to return whether it is summer time for given UTC time and timezone.

    Parameters
    ----------
    time : datetime
        timestamp to check for
    tz : string or pytz.timezone
        timezone to check for

    Returns
    -------
    bool
        True if the timestamp is in summer time, False if not.

    """
    dst = pytz.timezone(tz)._utc_transition_times[1:]
    dst_in_year = [date for date in dst if date.year == time.year]
    spring = dst_in_year[0]
    fall = dst_in_year[1]
    if (time.replace(tzinfo=None) >= spring) & (time.replace(tzinfo=None) < fall):
        return True
    else:
        return False


def normalize_tz(df, timezone=None):
    """
    Normalize the index in DataFrame or Series to remove difference between summer and winter time.
    Explanation:
        In the summer, time advances 1h. So In the US in the summer it is 1h more then in winter.
        Since markets open always at 9:30 US time, it means that in the summer they open 1h too
        early according to UTC.
        So this function adds 1h to UTC timestamps in the summer.

    Parameters
    ----------
    df : DataFrame
        common to normalize
    timezone : string or pytz.timezone
        if 'GMT Offset' column does not exist it is necessary to provide timezone to
        check summer time.

    Returns
    -------
    DataFrame
        normalized common

    """

    is_dst_ = df.index.map(lambda x: is_dst(x, timezone)).values
    # dst - > dst - 1
    df_summer = df.iloc[np.where(is_dst_ == 1)[0]]
    df_winter = df.iloc[np.where(is_dst_ == 0)[0]]

    # convert summer to winter time by adding 1h in summer
    if 'Date-Time' not in df_summer.columns:
        df_summer.index.name = 'Date-Time'
        df_winter.index.name = 'Date-Time'
        df_summer = df_summer.reset_index()
    df_summer['Date-Time'] += pd.to_timedelta(1, unit='h')
    df_summer = df_summer.set_index('Date-Time')

    if 'Date-Time' in df_winter.columns:
        df_winter = df_winter.drop(columns=['Date-Time'])
    df_normalized_tz = pd.concat([df_summer, df_winter], axis=0)
    df_normalized_tz.sort_index(inplace=True)
    return df_normalized_tz


def datetime_to_ms(timestamp):
    """

    Parameters
    ----------
    timestamp : datetime.datetime


    Returns
    -------
    ms : int
        milliseconds since Unix epoch

    """
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be datetime.datetime, but it is {}".format(type(timestamp)))
    return int(timestamp.timestamp() * 1000)

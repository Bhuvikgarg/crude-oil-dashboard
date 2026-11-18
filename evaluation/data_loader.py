"""
Historical data loader with on-disk caching.

Yahoo Finance limits: 1h bars -> last ~730 days, 15m bars -> last ~60 days.
Data is cached as pickles under evaluation/data_cache/ so that every run of the
evaluation is reproducible on the *same* snapshot (pass refresh=True to re-download).
"""
import os
import logging
import pandas as pd
import yfinance as yf

MACRO_TICKERS = ['CL=F', 'BZ=F', 'DX-Y.NYB', 'RB=F', 'HO=F', 'GC=F', '^OVX', 'ES=F', 'XLE']
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')


def _cached(name, loader, refresh):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    if os.path.exists(path) and not refresh:
        return pd.read_pickle(path)
    df = loader()
    if df is None or len(df) == 0:
        raise RuntimeError(f'Download for {name} returned no rows (network / Yahoo limits?)')
    df.to_pickle(path)
    return df


def load_history(refresh=False):
    """Return dict(macro_1h, cl_15m, cl_daily) with as much history as Yahoo allows."""
    macro_1h = _cached('macro_1h', lambda: yf.download(
        MACRO_TICKERS, period='730d', interval='1h', progress=False, auto_adjust=False), refresh)
    cl_15m = _cached('cl_15m', lambda: yf.Ticker('CL=F').history(period='60d', interval='15m'), refresh)
    cl_daily = _cached('cl_daily', lambda: yf.Ticker('CL=F').history(period='5y', interval='1d'), refresh)

    # Normalise everything to UTC so slicing across sources is unambiguous
    for df in (macro_1h, cl_15m, cl_daily):
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
    macro_1h = macro_1h.sort_index()
    logging.info('Loaded macro_1h=%d rows, cl_15m=%d rows, cl_daily=%d rows',
                 len(macro_1h), len(cl_15m), len(cl_daily))
    return {'macro_1h': macro_1h, 'cl_15m': cl_15m.sort_index(), 'cl_daily': cl_daily.sort_index()}

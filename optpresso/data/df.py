"""Optpresso utilities for Panda DataFrames"""
import os
import warnings
from datetime import date

from tempfile import mkstemp

from typing import List, Tuple, Optional, Dict, Any

import requests
import numpy as np
from numpy.typing import NDArray
import pandas as pd


def dataframe_from_csv(path_or_uri: str) -> pd.DataFrame:
    """Create a Pandas DataFrame from either a file path or a URI"""
    if path_or_uri.startswith(("http", "www")):
        # Don't use NamedTemporaryFile to support windoze
        fd, path = mkstemp(suffix=".csv")
        try:
            # Close to ensure windoze doesn't have two open fd on same file
            os.close(fd)
            with open(path, "wb") as ofs:
                resp = requests.get(path_or_uri, stream=True)
                for chunk in resp.iter_content():
                    ofs.write(chunk)
            return pd.read_csv(path)
        finally:
            os.remove(path)
    elif os.path.isfile(path_or_uri):
        return pd.read_csv(path_or_uri)
    else:
        raise ValueError(f"{path_or_uri} is not a file or a URI")


def convert_datetime_to_epoch_time(date):
    return (date.astype("uint64") / 1e9).astype("uint32")


def find_columns_to_drop(
    df: pd.DataFrame, skipped_columns: Optional[List[str]] = None
) -> List[str]:
    if skipped_columns is None:
        skipped_columns = set()
    droppable_columns = []
    for col in df.columns:
        if col in skipped_columns:
            continue
        # Any column that is mostly NaNs can be considered droppable
        if np.count_nonzero(df[col].isna()) > 0.5 * len(df[col]):
            droppable_columns.append(col)
    return droppable_columns


def prepare_df_for_modeling(
    frame: pd.DataFrame,
    result_col: str,
    encoders: Optional[Dict[str, Any]] = None,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[NDArray, NDArray]:
    """Converts a dataframe into features and the expected result for use with models

    Uses a stable sort on the column names to ensure the same ordering of features.

    Parameters
    ----------

    frame: DataFrame
        DataFrame to convert into numpy arrays for use with Models

    result_col: string
        Column to use as the expected output

    encoders: Optional Dictionary containing encoders
        Expects encoders to have `transform` method to call on columns

    drop_cols: List of strings
        Columns to remove from DataFrame before preparing

    Returns
    -------

    tuple of NDArrays: features and result values
    """
    if drop_cols is None:
        drop_cols = []
    new_frame = frame.copy(deep=True)
    for col in drop_cols:
        try:
            new_frame = new_frame.drop([col], axis=1)
        except KeyError:
            continue
    if encoders is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col, encoder in encoders.items():
                if col not in new_frame:
                    if col not in drop_cols:
                        print("No such column:", col)
                    continue
                try:
                    vals = encoder.transform(new_frame[col].values.reshape(-1, 1))
                except TypeError:
                    vals = encoder.transform(new_frame[col].values)
                new_frame[col] = [vals[i] for i in range(len(vals))]

    y = np.asarray(new_frame[result_col])
    for drop_col in [result_col]:
        try:
            new_frame = new_frame.drop(drop_col, axis=1)
        except KeyError:
            continue
    # Stable sort the columns so the ordering is consistent
    cols = list(sorted(new_frame.columns.tolist()))
    frame_array = new_frame[cols].to_numpy()
    x = np.zeros((frame_array.shape[0], np.hstack(frame_array[0]).shape[0]))
    # Surely there is a way to vectorize this?
    for i in range(frame_array.shape[0]):
        x[i] = np.hstack(frame_array[i])

    return x, y

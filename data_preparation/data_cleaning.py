from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd

def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def _fix_numeric_types(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in object_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].str.lower()

    return df

def _drop_constant_columns(df: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    constant_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            constant_cols.append(col)

    return df.drop(columns=constant_cols) if constant_cols else df

def _handle_missing_values(df: pd.DataFrame, target_col: Optional[str] = None,) -> pd.DataFrame:
    df = df.copy()

    if target_col is not None and target_col in df.columns:
        df = df[df[target_col].notna()].copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        if df[col].isna().all():
            fill_val = "missing"
        else:
            fill_val = df[col].mode(dropna=True)[0]
        df[col] = df[col].fillna(fill_val)

    return df

def _clip_outliers(df: pd.DataFrame, quantile_low: float = 0.01, quantile_high: float = 0.99, exclude: Optional[List[str]] = None,) -> pd.DataFrame:
    df = df.copy()
    exclude = set(exclude or [])
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    for col in numeric_cols:
        low = df[col].quantile(quantile_low)
        high = df[col].quantile(quantile_high)
        df[col] = df[col].clip(lower=low, upper=high)

    return df

def clean_dataframe(df: pd.DataFrame, target_col: Optional[str] = None, clip_outliers_flag: bool = True,) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = _standardize_column_names(cleaned)

    cleaned = _drop_duplicates(cleaned)

    exclude_for_numeric = [target_col] if target_col is not None else []
   
    cleaned = _fix_numeric_types(cleaned, exclude=exclude_for_numeric)

    cleaned = _drop_constant_columns(cleaned, target_col=target_col)

    cleaned = _handle_missing_values(cleaned, target_col=target_col)

    if clip_outliers_flag:
        exclude_for_outliers = [target_col] if target_col is not None else []
        cleaned = _clip_outliers(cleaned, exclude=exclude_for_outliers)

    return cleaned

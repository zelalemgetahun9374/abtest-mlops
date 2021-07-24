import pandas as pd

def count_values(df, column):
    new_df = df[column].value_counts().reset_index()
    new_df = new_df.rename(
        columns={'index': column, column: "count"})
    return new_df

def find_agg(df: pd.DataFrame, agg_column: str, agg_metric: str, col_name: str, order=False) -> pd.DataFrame:
    new_df = df.groupby(agg_column)[agg_column].agg(agg_metric).reset_index(name=col_name).sort_values(by=col_name,
        ascending=order)
    return new_df.reset_index(drop=True)
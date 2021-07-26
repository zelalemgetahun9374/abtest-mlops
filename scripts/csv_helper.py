import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('../')))

class CsvHelper():

  def __init__(self):
    pass

  def save_csv(self, df, csv_path, index=False):
    try:
      df.to_csv(csv_path, index=index)

    except Exception:
        pass

  def read_csv(self, csv_path, missing_values=[]):
    try:
      df = pd.read_csv(csv_path, na_values=missing_values)
      return df
    except FileNotFoundError:
        pass

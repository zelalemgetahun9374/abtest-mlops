from csv_helper import CsvHelper

helper = CsvHelper()
df = helper.read_csv("../data/AdSmartABdata.csv")
df.drop('platform_os', inplace=True, axis=1)

helper.save_csv(df, "../data/AdSmartABdata.csv")
import pandas as pd
import ta

# Load datas
df = pd.read_csv('./ETF_dfs/XIC.csv', sep=',')

# Clean NaN values
# df = ta.utils.dropna(df)

# df[:] = pd.to_numeric(df[:])
print(df.dtypes)

# Add ta features filling NaN values
df = ta.add_all_ta_features(df, "1. open", "2. high", "3. low", "4. close", "6. volume", fillna=True)
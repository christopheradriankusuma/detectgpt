import os
import pandas as pd

files = [f for f in os.listdir() if f.endswith('.csv')]
print(files)
for f in files:
    print(f)
    print(pd.read_csv(f, index_col=0))
    print()

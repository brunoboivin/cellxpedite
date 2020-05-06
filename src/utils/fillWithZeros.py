import os
import sys
import pandas as pd
from src.utils import cxpPrinter

def fillWithZeros(input_dir):
    files = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if '.csv' in f]
    for f in files:
        cxpPrinter.cxpPrint('Working on file ' + f)
        
        # fill na values with 0's
        df = pd.read_csv(f)
        df = df.fillna(0)
        df.to_csv(f, index=False)


if __name__ == "__main__":
    fillWithZeros(*sys.argv[1:])
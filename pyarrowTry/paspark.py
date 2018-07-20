
import pyarrow as pa
import pandas as pd
import numpy as np

def test():
    df = pd.DataFrame({'one': [-1, np.nan, 2.5],
                        'two': ['foo', 'bar', 'baz'],
                        'three': [True, False, True]})

    table = pa.Table.from_pandas(df)

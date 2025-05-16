""" Yeo-Johnson transformation wrapper"""
import pandas as pd
import numpy as np

def transform(y: float, lmda: float) -> float:
    return y

def inv_transform(z: float, lmda: float) -> float:
    return z

def search(df: list | np.ndarray | pd.Series | pd.DataFrame) -> float:
    return 0.0

if __name__ == "__main__":

    p: int = 10
    for _ in range(0, p):
        n: int = 11
        xp: np.ndarray = np.random.lognormal(mean=0.0, sigma=1.0, size=n)
        if _ > 0:
            X = pd.DataFrame(data=np.column_stack([X, xp]))
        else:
            X: np.ndarray = xp
    print(X)
    
    s: pd.Series = pd.Series(data=X.mean(axis=0), index=None, name="x")
    print(s)

    l: list[float] = s.to_list()
    print(l)

    df: pd.DataFrame = pd.DataFrame(data=X)
    _: set = set([f"'{_}':x{_}" for _ in range (0, p)])
    df.rename(columns=_, inplace=True)
    print(df)
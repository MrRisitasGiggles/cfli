import pandas as pd
import numpy as np
from datetime import datetime
from os.path import join
import matplotlib.pyplot  as plt

root_dir: str = r"C:\Users\Werhner Wangra\OneDrive\Documents\Python\cfli\cache"
img_dir: str = r"C:\Users\Werhner Wangra\OneDrive\Documents\Python\cfli\img"

csv_file: str = "CPIAUCSL.csv"
cpi: pd.DataFrame = pd.read_csv(join(root_dir, csv_file))
cpi['DATEX'] = cpi['observation_date']\
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
cpi.drop(columns=["observation_date"], inplace=True)
cpi.set_index(keys="DATEX", inplace=True)
cpi = cpi / cpi.shift(periods=12) - 1

csv_file: str = "GDPC1.csv"
gdp: pd.DataFrame = pd.read_csv(join(root_dir, csv_file))
gdp['DATEX'] = gdp['observation_date']\
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
gdp.drop(columns=["observation_date"], inplace=True)
gdp.set_index(keys="DATEX", inplace=True)
gdp = gdp / gdp.shift(periods=12) - 1

csv_file: str = "T10Y3M.csv"
rate: pd.DataFrame = pd.read_csv(join(root_dir, csv_file))
rate['DATEX'] = rate['observation_date']\
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
rate.drop(columns=["observation_date"], inplace=True)
rate.set_index(keys="DATEX", inplace=True)
rate = rate / 100.0

csv_file: str = "UNRATE.csv"
unrate: pd.DataFrame = pd.read_csv(join(root_dir, csv_file))
unrate['DATEX'] = unrate['observation_date']\
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
unrate.drop(columns=["observation_date"], inplace=True)
unrate.set_index(keys="DATEX", inplace=True)
unrate = unrate / 100.0

# Join all variables into a single data frame.
df: pd.DataFrame = pd.concat([cpi, gdp, rate, unrate], join="inner",
                             axis=1,
                             ignore_index=False)

df_std: pd.DataFrame = df.copy()
for _col in df.select_dtypes("number").columns.to_list():
    df[_col] = df[_col]\
        .interpolate(method="linear", axis=0, inplace=False)
    df_std[_col] = df[_col]\
        .apply(lambda z: (z-df[_col].mean())/df[_col].std())
    plt.hist(df_std[_col].to_numpy(), bins=21, density=True)
    plt.savefig(join(img_dir, f"{_col} - Standardised.png"))
    plt.close()
    plt.hist(df[_col].to_numpy(), bins=21, density=True)
    plt.savefig(join(img_dir, f"{_col} - Actual.png"))
    plt.close()

# Calculate means, and standard deviations.
mu, sd = df.select_dtypes("number").mean(),\
    df.select_dtypes("number").std()
print(mu, sd, sep="\n\n")

# Calculate correlation matrix.
corr_mat: pd.DataFrame = df.corr(method="spearman", numeric_only=True)

# Calculate Cholesky decomposition.
evals, evecs = np.linalg.eig(corr_mat)
for _, eval in enumerate(evals):
    evals[_] = max(1e-4, eval)
corr_mat_ = np.diag(evals)
corr_mat_ = np.linalg\
    .matmul(np.linalg.matmul(evecs, corr_mat_),
            np.matrix_transpose(evecs))
corr_mat_: pd.DataFrame =\
    pd.DataFrame(data=corr_mat, index=corr_mat.index,
                 columns=corr_mat.columns.to_flat_index())
print(corr_mat, corr_mat_, sep="\n\n")

z: np.ndarray = np.random.multivariate_normal(mean=0.0*mu, cov=corr_mat_,
                                              size=10000)
z: pd.DataFrame = pd.DataFrame(data=z, index=None,
                               columns=corr_mat_.columns.to_list())
z = mu + sd * z

z.sort_values(by=["CPIAUCSL", "GDPC1", "T10Y3M", "UNRATE"], inplace=True)
y: list[float] = [0.0] * (1 + z.shape[0])
for row in range(0, z.shape[0]):
    for _row in range(row, z.shape[0]):
        if z["CPIAUCSL"].iloc[_row] <= z["CPIAUCSL"].iloc[row]\
            and z["GDPC1"].iloc[_row] <= z["GDPC1"].iloc[row]\
                and z["T10Y3M"].iloc[_row] <= z["T10Y3M"].iloc[row]\
                    and z["UNRATE"].iloc[_row] <= z["UNRATE"].iloc[row]:
            y[row+1] = y[row] + 1 / z.shape[0]

z["y"] = pd.Series(data=y[1:], index=z.index, name="frequency")
z.sort_values(by=["y"], inplace=True)
z.to_clipboard()

# print(z.corr(method="spearman"))

for _col in z.columns.to_list():
    plt.hist(z[_col].to_numpy(), bins=41, density=True)
    plt.savefig(join(img_dir, f"{_col} - Simulation.png"))
    plt.close()

print(z)
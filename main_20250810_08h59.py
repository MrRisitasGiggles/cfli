import pandas as pd
import numpy as np
from datetime import datetime
from os.path import join
import matplotlib.pyplot  as plt
from scipy.stats import norm as norm
import statsmodels.api as sm
import src.yjtrf as yjtrf

root_dir: str = "./cache"
img_dir: str = "./img"

csv_file: str = "CPIAUCSL.csv"
cpi: pd.DataFrame = pd.read_csv(join(root_dir, csv_file))
cpi['DATEX'] = cpi['observation_date']\
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
cpi.drop(columns=["observation_date"], inplace=True)
cpi.set_index(keys="DATEX", inplace=True)
cpi = cpi / cpi.shift(periods=12) - 1
cpi_mu, cpi_sd = cpi["CPIAUCSL"].mean(skipna=True),\
        cpi["CPIAUCSL"].std(skipna=True)
cpi["CPIAUCSL"] = cpi["CPIAUCSL"].apply(lambda z: (z-cpi_mu)/cpi_sd)
cpi_l: float = yjtrf.soek(s=pd.Series(data=cpi["CPIAUCSL"],
                                      index=cpi.index, name="CPIAUCSL"))
print(f"\nCPI: Yeo-Johnson transform is: {cpi_l:>5.2}")
cpi["CPIAUCSL"] = cpi["CPIAUCSL"].apply(lambda z: yjtrf.trf(x=z, l=cpi_l))

csv_file: str = "GDPC1.csv"
gdp: pd.DataFrame = pd.read_csv(join(root_dir, csv_file))
gdp['DATEX'] = gdp['observation_date']\
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
gdp.drop(columns=["observation_date"], inplace=True)
gdp.set_index(keys="DATEX", inplace=True)
gdp = gdp / gdp.shift(periods=12) - 1
gdp_mu, gdp_sd = gdp["GDPC1"].mean(skipna=True), gdp["GDPC1"].std(skipna=True)
gdp["GDPC1"] = gdp["GDPC1"].apply(lambda z: (z-gdp_mu)/gdp_sd)
gdp_l: float = yjtrf.soek(s=pd.Series(data=gdp["GDPC1"],
                                      index=gdp.index, name="GDPC1"))
print(f"\nGDP: Yeo-Johnson transform is : {gdp_l:>5.2}")
gdp["GDPC1"] = gdp["GDPC1"].apply(lambda z: yjtrf.trf(x=z, l=gdp_l))

csv_file: str = "T10Y3M.csv"
rate: pd.DataFrame = pd.read_csv(join(root_dir, csv_file))
rate['DATEX'] = rate['observation_date']\
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
rate.drop(columns=["observation_date"], inplace=True)
rate.set_index(keys="DATEX", inplace=True)
rate = rate / 100.0
rate_mu, rate_sd = rate["T10Y3M"].mean(skipna=True),\
        rate["T10Y3M"].std(skipna=True)
rate["T10Y3M"] = rate["T10Y3M"].apply(lambda z: (z-rate_mu)/rate_sd)
rate_l: float = yjtrf.soek(s=pd.Series(data=rate["T10Y3M"],
                                      index=gdp.index, name="T10Y3M"))
print(f"\nRate: Yeo-Johnson transform is : {rate_l:>5.2}")
rate["T10Y3M"] = rate["T10Y3M"].apply(lambda z: yjtrf.trf(x=z, l=rate_l))

csv_file: str = "UNRATE.csv"
unrate: pd.DataFrame = pd.read_csv(join(root_dir, csv_file))
unrate['DATEX'] = unrate['observation_date']\
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
unrate.drop(columns=["observation_date"], inplace=True)
unrate.set_index(keys="DATEX", inplace=True)
unrate = unrate / 100.0
unrate_mu, unrate_sd = unrate["UNRATE"].mean(skipna=True),\
        unrate["UNRATE"].std(skipna=True)
unrate["UNRATE"] = unrate["UNRATE"].apply(lambda z: (z-unrate_mu)/unrate_sd)
unrate_l: float = yjtrf.soek(s=pd.Series(data=unrate["UNRATE"],
                                      index=unrate.index, name="UNRATE"))
print(f"\nUnemployment Rate: Yeo-Johnson transform is : {unrate_l:>5.2}")
unrate["UNRATE"] = unrate["UNRATE"].apply(lambda z: yjtrf.trf(x=z, l=unrate_l))

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
print("\n")
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
print("\n")
print(corr_mat, corr_mat_, sep="\n\n")

z: np.ndarray = np.random.multivariate_normal(mean=0.0*mu, cov=corr_mat_,
                                              size=500000)
z: pd.DataFrame = pd.DataFrame(data=z, index=None,
                               columns=corr_mat_.columns.to_list())
z = mu + sd * z
z["CPIAUCSL"] = z["CPIAUCSL"].apply(lambda z: yjtrf.invtrf(y=z, l=cpi_l))
z["CPIAUCSL"] = cpi_mu + cpi_sd * z["CPIAUCSL"]
z["GDPC1"] = z["GDPC1"].apply(lambda z: yjtrf.invtrf(y=z, l=gdp_l))
z["GDPC1"] = gdp_mu + gdp_sd * z["GDPC1"]
z["T10Y3M"] = z["T10Y3M"].apply(lambda z: yjtrf.invtrf(y=z, l=rate_l))
z["T10Y3M"] = rate_mu + rate_sd * z["T10Y3M"]
z["UNRATE"] = z["UNRATE"].apply(lambda z: yjtrf.invtrf(y=z, l=unrate_l))
z["UNRATE"] = unrate_mu + unrate_sd * z["UNRATE"]

z.sort_values(by=["GDPC1", "UNRATE", "T10Y3M"], inplace=True,
              ascending=[True, False, False])
z["y"] = pd.Series(data=range(1, z.shape[0] + 1), index=z.index,
                   name="frequency")
z["y"] = z["y"] /(1 + z.shape[0])
z["y"] = z["y"].apply(lambda x: norm.ppf(x))
# print(z.corr(method="spearman"))

X: np.ndarray = z[["CPIAUCSL", "GDPC1", "T10Y3M", "UNRATE"]].to_numpy()
X = sm.add_constant(X)
Y: np.ndarray = z["y"].to_numpy()
model = sm.OLS(Y, X)
results = model.fit()
z["mc"] = 50.0 + 12.5 * pd.Series(data=results.fittedvalues,
                                  index=z.index, name="mc")
z.to_clipboard()
print("\n\n")
print(results.summary())

for _col in z.columns.to_list():
    plt.hist(z[_col].to_numpy(), bins=41, density=True)
    plt.savefig(join(img_dir, f"{_col} - Simulation.png"))
    plt.close()

print("\n")
print(z)

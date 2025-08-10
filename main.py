""" No dosctring?! """
import datetime

import pandas as pd

from src import sarima_forecast as src_frcst

START_DATE=datetime.datetime.strptime("2000-01-31","%Y-%m-%d").date()
FINAL_DATE=datetime.datetime.today().date() + pd.offsets.MonthEnd(0)

root_dir: str = "./cache"
img_dir: str = "./img"
csv_files: list[str] = ["CPIAUCSL.csv","GDPC1.csv",
                        "T10Y3M.csv","UNRATE.csv"]

macrovars: list[str] = []
for _csv_file in csv_files:
    macrovar: str = _csv_file.strip()\
        .split(".", maxsplit=1)[0].upper()
    locals()[macrovar] =\
        src_frcst.import_file(csv_file=f"{root_dir}/{_csv_file}")
    macrovars.append(macrovar)

_0: pd.DataFrame =\
    pd.DataFrame(index=pd.date_range(start=START_DATE, end=FINAL_DATE,
                    freq="ME").tolist())

for macrovar in macrovars:
    _0 = pd.merge(left=_0, right=locals()[macrovar],
                how="left", validate="1:1",
                left_index=True, right_index=True,
                sort=True)
print(f"\n_0:\n{_0.describe()}\n")

src_frcst.get_scenarios(df=_0, macrovars=macrovars, img_dir=img_dir)

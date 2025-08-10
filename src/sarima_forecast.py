""" Function to forecast floating values in a dataframe that is
datetime indexed."""
import math
import datetime
import logging

import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt

from src import yjtrf

def get_scenarios(df: pd.DataFrame, macrovars: list[str],
                    img_dir: str) -> None:
    """ No docstring?! """

    _vandag: datetime.datetime = datetime.datetime.today()
    _log: str = f"Scenario Generation - "\
        + f"{_vandag.date().isoformat()} " + f"{_vandag.hour:01d}h"\
            + f"{_vandag.minute:01d}m" + f"{_vandag.second:01d}s"
    logger: logging.Logger = logging.getLogger("scenario_generation")
    logging.basicConfig(filename=f"./{_log.strip()}.log",
                        level=logging.INFO, encoding="utf-8")

    _1: pd.DataFrame = df.copy()
    for macrovar in macrovars:
        _1[macrovar] =\
            _1[macrovar].interpolate(method="linear",
                                     limit_area="inside")
    if logger is not None:
        logger.info(f"\n_1:\n{_1.describe()}\n")

    for macrovar in macrovars:
        plt.title(label=f"Histogram of {macrovar}")
        plt.hist(_1[macrovar].to_numpy(),
                    bins=1+int(2.33*math.log(_1.shape[0],10)),
                        density=True)
        plt.savefig(f"{img_dir}/{macrovar} - Actual.png")
        plt.close()
        plt.title(label=f"Lineplot of {macrovar}")
        plt.plot(_1.index, _1[macrovar], label=f"{macrovar}")
        plt.xlabel(xlabel="Date")
        plt.ylabel(ylabel=f"{macrovar}")
        plt.savefig(f"{img_dir}/{macrovar} - Timeseries.png")
        plt.close()

    _stats: dict = {}
    for macrovar in macrovars:
        _stats[macrovar] = {}
        if _1[macrovar].min(skipna=True, numeric_only=True) > 0.0:
            _stats[macrovar]["log"] = True
            _min: float = _1[macrovar].apply(lambda x: math.log(1.0+x)\
                if not pd.isna(x) else math.nan)\
                    .min(skipna=True, numeric_only=True)
            _max: float = _1[macrovar].apply(lambda x: math.log(1.0+x)\
                if not pd.isna(x) else math.nan)\
                    .max(skipna=True, numeric_only=True)
        else:
            _stats[macrovar]["log"] = False
            _min: float = _1[macrovar]\
                .min(skipna=True, numeric_only=True)
            _max: float = _1[macrovar]\
                .max(skipna=True, numeric_only=True)
        _stats[macrovar]["min"] = _min
        _stats[macrovar]["max"] = _max

    _2: pd.DataFrame = _1.copy()
    for macrovar in macrovars:
        _min: float = _stats[macrovar]["min"]
        _max: float = _stats[macrovar]["max"]
        if _stats[macrovar]["log"]:
            _2[macrovar] = _2[macrovar].apply(lambda x:\
                0.1*(math.log(1.0+x)-_min)/(_max-_min)\
                    if not pd.isna(x) else math.nan)
        else:
            _2[macrovar] = _2[macrovar].apply(lambda x:\
                0.1*(x-_min)/(_max-_min)\
                    if not pd.isna(x) else math.nan)
        _lambda: float = yjtrf.soek(s=_2[macrovar].dropna())
        _stats[macrovar]["lambda"] = _lambda

    if logger is not None:
        logger.info(f"\n_stats:\n{_stats}\n")
        logger.info(f"\n_2:\n{_2}\n")

    _3: pd.DataFrame = _2.copy()
    for macrovar in macrovars:
        _: float = _stats[macrovar]["lambda"]
        _3[macrovar] = _3[macrovar].apply(lambda x: yjtrf.trf(x, _)
                            if not pd.isna(x) else math.nan)
        _3[macrovar] = _3[macrovar].interpolate(method="linear",
                                                    limit_area="inside")
        plt.title(label=f"Histogram of transformed {macrovar}")
        plt.hist(_3[macrovar].dropna().to_numpy(),
                    bins=1+int(2.33*math.log(_1.shape[0],10)),
                        density=True)
        plt.savefig(f"{img_dir}/{macrovar} - Transformed.png")
        plt.close()

    _4: pd.DataFrame = _3.copy()
    for macrovar in macrovars:
        _mu: float = _3[macrovar].mean(skipna=True, numeric_only=True)
        _sd: float = _3[macrovar].std(skipna=True, numeric_only=True)
        _stats[macrovar]["mean"] = _mu
        _stats[macrovar]["std"] = _sd
        _4[macrovar] = _4[macrovar].apply(lambda x:\
            (x-_mu)/_sd if not pd.isna(x) else math.nan)
    if logger is not None:
        logger.info(f"\n_4:\n{_4}\n")


    for macrovar in macrovars:
        _: pd.DataFrame = quick_arima(s=_4[macrovar])
        plt.title(label=f"Forecasts of {macrovar}")
        plt.xlabel(xlabel="Date")
        plt.ylabel(ylabel=f"{macrovar}")
        for _col in _.columns.tolist():
            _min: float = _stats[macrovar]["min"]
            _max: float = _stats[macrovar]["max"]
            _mu: float = _stats[macrovar]["mean"]
            _sd: float = _stats[macrovar]["std"]
            _lambda: float = _stats[macrovar]["lambda"]
            _[_col] = _[_col].apply(lambda x:\
                _mu + _sd * x if not pd.isna(x) else math.nan)
            _[_col] = _[_col].apply(lambda x:\
                yjtrf.invtrf(y=x, l=_lambda)\
                    if not pd.isna(x) else math.nan)
            _[_col] = _[_col].apply(lambda x:\
                    _min + 10.0*(_max-_min) * x\
                        if not pd.isna(x) else math.nan)
            if _stats[macrovar]["log"]:
                _[_col] = _[_col].apply(lambda x: math.exp(x)-1.0\
                        if not pd.isna(x) else math.nan)
            plt.plot(_.index, _[_col], label=f"{_col}")
        plt.savefig(f"{img_dir}/{macrovar} - Modelled.png")
        plt.close()

def quick_arima(s: pd.Series, logger: logging.Logger = None)\
    -> pd.DataFrame:
    """ No docstring?! """

    _s: pd.Series = pd.Series(data=s, name=f"{s.name}")
    _s.dropna(inplace=True)
    # https://www.google.com/search?q=Python+autofit+SARIMA+with+pmdarima&sxsrf=AE3TifPuFhFg_a-8uVGgzY3afC-9_6h7Qw%3A1754817334497
    model = pm.auto_arima(
        _s,
        seasonal=True,  # Set to True for SARIMA, False for ARIMA
        m=12,           # Seasonal period (e.g., 12 for monthly data)
        start_p=0,      # Starting order for p
        start_q=0,      # Starting order for q
        max_p=5,        # Maximum order for p
        max_q=5,        # Maximum order for q
        max_P=2,        # Maximum order for P (seasonal AR)
        max_Q=2,        # Maximum order for Q (seasonal MA)
        trace=True,     # Show fitting progress
        suppress_warnings=True, # Suppress warnings
        error_action='ignore', # Handle errors gracefully
        stepwise=True   # Use stepwise search for efficiency
    )
    if logger is not None:
        logger.info(model.summary())
    _steps: int = 72
    tdf: datetime.datetime = _s.index[0] + pd.offsets.MonthEnd(0)
    t_n: datetime.datetime = _s.dropna().index[-1]\
        + pd.offsets.MonthEnd(0)
    _p: list[float] = [0.01, 0.05, 0.10, 0.20, 0.25, 0.45]
    fitted: pd.Series =\
        pd.Series(data=model.predict_in_sample(),name=f"{_s.name}_MEAN")
    prediction: pd.Series =\
        pd.Series(data=model.predict(n_periods=_steps-1),
                    name=f"{_s.name}_MEAN")
    prediction = pd.concat([fitted, prediction], axis=0,
                            ignore_index=True)
    prediction.index =\
        pd.date_range(start=tdf,periods=prediction.shape[0],freq="ME")
    for p in _p:
        _, conf_int =\
            model.predict(n_periods=_steps, return_conf_int=True,alpha=p)
        ci: pd.DataFrame =\
            pd.DataFrame(data=conf_int,
                         index=pd.date_range(start=t_n, periods=_steps,
                            freq="ME").tolist(),
                         columns=[f"{_s.name}_{int(100*p):>01}",
                                  f"{_s.name}_{int(100*(1.0-p)):>01}"])
        prediction: pd.DataFrame =\
            pd.merge(left=prediction, right=ci, how="left",
                     left_index=True, right_index=True, validate="1:1",
                     sort=True)
    prediction = pd.merge(left=prediction, right=_s, how="left",
                          left_index=True, right_index=True,
                          sort=True, validate="1:1")
    return prediction

def import_file(csv_file: str, logger: logging.Logger = None)\
    -> pd.DataFrame:
    """ No docstring?! """
    _name: str = csv_file.strip().split(".", maxsplit=1)[0].upper()
    locals()[_name]: pd.DataFrame = pd.read_csv(csv_file)
    locals()[_name]['DATEX'] = locals()[_name]['observation_date']\
                                    .apply(lambda x: pd.offsets.MonthEnd(0)+\
                                    datetime.datetime\
                                        .strptime(x, "%Y-%m-%d"))
    locals()[_name].drop(columns=["observation_date"], inplace=True)
    locals()[_name] = locals()[_name]\
        .groupby(by=["DATEX"], as_index=True).median().reset_index()
    locals()[_name].set_index(keys="DATEX", inplace=True)
    if logger is not None:
        logger.info(f"\n{_name}\n{locals()[_name].describe()}\n")
    return locals()[_name]

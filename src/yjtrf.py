"""Yeo-Johnson transform functionality."""
import math
import unittest
import numpy as np
import pandas as pd

def trf(x: float, l: float) -> float:
    """
    Performs a Yeo-Johnson transform on a single number given a value for
    lambda.

    Parameters:
    -----------
        x: `float`
            - The number to transform.
        l: `float`
            - The value of the Yeo-Johnson parameter.
    
    Returns:
    --------
        y: `float`
            - The Yeo-Johnson transformed random variable.
    """
    try:
        if isinstance(x, float) or isinstance(x, int):
            if isinstance(l, float) or isinstance(l, int):
                y: float = _trf(_x=x, _lambda=l)
            else:
                y = x
                print(f"\nNOTE: {__name__}: trf(x,l) -> Wrong type passed "\
                        + f"for lambda -> {l}.")
        else:
            y = x
            print(f"\nWARNING: {__name__}: trf(x,l) -> Wrong types passed "\
                    + f"for x -> {x} and lambda -> {l}.")
    except ValueError as _xcptn:
        print(f"\nERROR: {__name__}: trf(x,l) -> {_xcptn}")
        y = x
    finally:
        return y 


def invtrf(y: float, l: float) -> float:
    """
    Performs an inverse Yeo-Johnson transform on a single number given a value
    for lambda.

    Parameters:
    -----------
        y: `float`
            - The number to untransform.
        l: `float`
            - The value of the Yeo-Johnson parameter.
    
    Returns:
    --------
        x: `float`
            - The untransformed random variable.
    """
    try:
        x: float = float(y)
        if isinstance(y, float) or isinstance(y, int):
            if isinstance(l, float) or isinstance(l, int):
                x: float = _invtrf(_y=y, _lambda=l)
            else:
                x = y
                print(f"\nNOTE: {__name__}: invtrf(y,l) -> wrong type passed "\
                        + f"for lambda -> {l}.")
        else:
            x = y
            print(f"\nWARNING: {__name__}: invtrf(y,l) -> Wrong types passed "\
                    + f"for x -> {x} and lambda -> {l}.")
    except ValueError as _xcptn:
        print(f"\nERROR: {__name__}: invtrf(y,l) -> {_xcptn}")
        x = y
    finally:
        return x

def soek(s: pd.Series | np.ndarray | list, _min: float = -5.0,
         _max: float = +5.0) -> float:
    """Searches for an optimal lambda for the Yeo-Johnson transform.

    Parameters:
    ------------
        1. s: `pd.Series` | `np.ndarray` | `list[float]
            Iterable containing floats or integers to transform.
        2. _min: float | int
            Minimum value for the transform lambda. Defaults to -5.0.
        3. _max: float | int
            Maximum value for the transform lambda. Defaults to +5.0.
    Returns:
    --------
    best_lambda: `float`
        - The best lambda from a skewness perspective in the given interval.
    """
    if isinstance(s, pd.Series):
        _s: pd.Series = s.copy()
    elif isinstance(s, np.ndarray):
        _s: pd.Series = pd.Series(data=s.flatten().tolist(), index=None,
                                  name=f"{s=}".split(sep="=")[0])
    elif isinstance(s, np.ndarray):
        _s: list = [_ for _ in s]
        _s: pd.Series = pd.Series(data=_s, index=None, name=f"{s=}"\
                .split(sep="=")[0])

    if abs(_s.skew()) > 1.0e-1:
        _steps: int = 201
        best_skew = math.inf
        for _ in range(0, _steps):
            test_lambda = 1e-4 + _min + _*(_max-_min)/_steps
            test_skew = _s.apply(lambda z: trf(x=z, l=test_lambda))\
                    .skew(skipna=True)
            if abs(test_skew) <= abs(best_skew):
                best_lambda = test_lambda
                best_skew = test_skew
                if abs(best_skew) <= 1.0e-1:
                    break

        lowr_lambda: float = best_lambda - (_max-_min)/_steps
        lowr_skew: float = _s.apply(lambda z: trf(x=z, l=lowr_lambda))\
                .skew(skipna=True)
        uppr_lambda: float = best_lambda + (_max-_min)/_steps
        uppr_skew: float = _s.apply(lambda z: trf(x=z, l=uppr_lambda))\
                .skew(skipna=True)

        for _ in range(0, 10):
            if abs(lowr_skew) <= abs(uppr_skew):
                if abs(lowr_skew) <= abs(best_skew):
                    uppr_lambda, uppr_skew = best_lambda, best_skew
                    best_lambda = (uppr_skew * lowr_lambda\
                            + lowr_skew * uppr_lambda)\
                            /(lowr_skew+uppr_skew)
                    best_skew = s.apply(lambda z: trf(x=z, l=best_lambda))\
                            .skew(skipna=True)
                else:
                    break
            elif abs(uppr_skew) <= abs(lowr_skew):
                if abs(uppr_skew) <= abs(best_skew):
                   lowr_lambda, lowr_skew = best_lambda, best_skew
                   best_lambda = (lowr_skew * uppr_lambda\
                           + uppr_skew * lowr_lambda)\
                           /(lowr_skew+uppr_skew)
                   best_skew = s.apply(lambda z: trf(x=z, l=best_lambda))\
                            .skew(skipna=True)
                else:
                    break
    else:
        best_lambda = 1.0

    return best_lambda

def _trf(_x: float, _lambda: float) -> float:
    """ Calculation of Yeo-Johnson transform of a variable"""

    _eps: float = 1.0e-5 # Used to adjust for numerical error.
    if (_lambda > 1.0 - _eps and _lambda < 1.0 + _eps) or _lambda is None:
        z: float = _x
    elif math.isnan(_x):
        z: float = _x
    elif math.isinf(_x):
        z: float = _x
    else:
        if (_lambda <= -_eps or _lambda > _eps) and _x >= 0.0:
            z: float = ((_x + 1.0) ** _lambda - 1.0) / _lambda
        elif (_lambda <= _eps and _lambda > -_eps) and _x >= 0.0:
            z: float = math.log(_x + 1.0)
        elif (_lambda <= 2.0 - _eps or _lambda > 2.0 + _eps) and _x < 0.0:
            z: float = -((-_x + 1) ** (2.0 - _lambda) - 1.0) / (2.0 - _lambda)
        else:
            z: float = -math.log(-_x + 1.0)
    return z

def _invtrf(_y: float, _lambda: float) -> float:
    """ Calculation of Yeo-Johnson inverse transformation of a variable."""

    _eps: float = 1.0e-5
    if (_lambda > 1.0 - _eps and _lambda < 1.0 + _eps) or _lambda is None:
        z: float = _y
    elif math.isnan(_y):
        z: float = _y
    elif math.isinf(_y):
        z: float = _y
    else:
        if (_lambda <= -_eps or _lambda > _eps) and _y >= _eps:
            z: float = (1 + _lambda * _y) ** (1.0/_lambda) - 1.0
        elif (_lambda <= _eps and _lambda > -_eps) and _y >= _eps:
            z: float = math.exp(_y) - 1.0
        elif (_lambda <= 2.0 - _eps or _lambda > 2.0 + _eps) and _y < -_eps:
            z: float = 1.0 - (1.0 - (2.0-_lambda)*_y)**(1.0/(2.0-_lambda))
        else:
            z: float = 1.0 - math.exp(-_y)
    return z

class Test_yftrf(unittest.TestCase):
    """
    Test class to test the transformation.

    Attributes:
    -----------
        None

    Methods:
    --------
        - test_normal()
            Tests whether a lambda close to 1.0 is returned for symmetric
            distribution.
        - test_lognormal()
            Tests whether a lambda close to zero is returned for a right-
            skewed distribution.
    """

    def test_normal(self):
        """Function to test transformation of the log-normal distribution."""

        print("\n")
        print("=" * 70)
        print(" " * 25 + " Normal distribution " + " " * 25)
        print("-" * 70)
        n: int = 1001
        _lambda: float = 0.0
        N: int = 500
        for _tests in range(0, N):
            x: list[float] = np.random.normal(loc=0.0, scale=1.0, size=n)
            s: list[float] = [x[0]]
            for _ in range(1, n):
                s.append(-x[_])
                s.append(+x[_])
            s: pd.Series = pd.Series(data=s, index=None, name="x")
            # print(f"\nThe skewness before transformation is: {s.skew()}.")
            _l: float = soek(s=s)
            _lambda += _l/N
            s0: pd.Series = s.apply(lambda z: trf(x=z, l=_l))
            # print(f"\nThe skewness after transformation is: {s0.skew()}.")
            # print(f"\nSummary statistics prior to transformation:\n{s.describe()}")
            # print(f"\nSummary statistics prior to transformation:\n{s0.describe()}")
            s1: pd.Series = s0.apply(lambda z: invtrf(y=z, l=_l))
            self.assertTrue(abs(s0.skew()) <= abs(s.skew()))
            self.assertEqual(round(number=100*s.sum(), ndigits=4)/100,
                             round(number=100*s1.sum(), ndigits=4)/100)
        print(f"\nThe Yeo-Johnson lambda is: {_lambda:>7.2f}.")

    def test_lognormal(self):
        """Function to test transformation of the log-normal distribution."""

        print("\n")
        print("=" * 70)
        print(" " * 23 + " Lognormal distribution " + " " * 23)
        print("-" * 70)
        n: int = 1001
        N: int = 500
        _lambda: float = 0.0
        for _tests in range(0, N):
            x: list[float] = np.random.normal(loc=0.0, scale=1.0, size=n)
            s: list[float] = [math.exp(x[0])]
            for _ in range(1, n):
                s.append(math.exp(-x[_]))
                s.append(math.exp(+x[_]))
            s: pd.Series = pd.Series(data=s, index=None, name="x")
            # print(f"\nThe skewness before transformation is: {s.skew()}.")
            _l: float = soek(s=s)
            _lambda += _l/N
            # print(f"\nThe Yeo-Johnson lambda is: {_l}.")
            s0: pd.Series = s.apply(lambda z: trf(x=z, l=_l))
            # print(f"\nThe skewness after transformation is: {s0.skew()}.")
            # print(f"\nSummary statistics prior to transformation:\n{s.describe()}")
            # print(f"\nSummary statistics prior to transformation:\n{s0.describe()}")
            s1: pd.Series = s0.apply(lambda z: invtrf(y=z, l=_l))
            self.assertTrue(abs(s0.skew()) <= abs(s.skew()))
            self.assertEqual(round(number=100*s.sum(), ndigits=4)/100,
                             round(number=100*s1.sum(), ndigits=4)/100)
        print(f"\nThe Yeo-Johnson lambda is: {_l:>7.2f}.")

    def test_neglognormal(self):
        """Function to test transformation of the log-normal distribution."""

        print("\n")
        print("=" * 70)
        print(" " * 20 + " Negative Lognormal distribution " + " " * 20)
        print("-" * 70)
        n: int = 1001
        N: int = 500
        _lambda: float = 0.0
        for _tests in range(0, N): 
            x: list[float] = np.random.normal(loc=0.0, scale=1.0, size=n)
            s: list[float] = [math.exp(x[0])]
            for _ in range(1, n):
                s.append(-math.exp(-x[_]))
                s.append(-math.exp(+x[_]))
            s: pd.Series = pd.Series(data=s, index=None, name="x")
            # print(f"\nThe skewness before transformation is: {s.skew()}.")
            _l: float = soek(s=s)
            _lambda += _l/N
            # print(f"\nThe Yeo-Johnson lambda is: {_l}.")
            s0: pd.Series = s.apply(lambda z: trf(x=z, l=_l))
            # print(f"\nThe skewness after transformation is: {s0.skew()}.")
            # print(f"\nSummary statistics prior to transformation:\n{s.describe()}")
            # print(f"\nSummary statistics prior to transformation:\n{s0.describe()}")
            s1: pd.Series = s0.apply(lambda z: invtrf(y=z, l=_l))
            self.assertTrue(abs(s0.skew()) <= abs(s.skew()))
            self.assertEqual(round(number=100*s.sum(), ndigits=4)/100,
                             round(number=100*s1.sum(), ndigits=4)/100)
        print(f"\nThe Yeo-Johnson lambda is: {_l:>7.2f}.")

if __name__ == "__main__":
    unittest.main()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def generate_data() -> pd.DataFrame:
    # ---- Simulate monthly data for 6 years ----
    T = 72
    time = np.arange(T)
    months = pd.date_range("2019-01-01", periods=T, freq="M")

    # Trend & seasonality
    trend = 0.1 * time
    seasonal = 5 * np.sin(2 * np.pi * time / 12)

    # Covariates
    detailing = 10 + 2 * np.sin(2 * np.pi * (time - 2) / 12) + np.random.normal(0, 1, T)
    seminar = 6 + 1.5 * np.cos(2 * np.pi * (time - 4) / 12) + np.random.normal(0, 1, T)
    email = 8 + np.random.normal(0, 0.5, T)

    # True relationship
    beta0, beta1, beta2, beta3 = 30, 1.5, 2.0, 0.5

    # Lag seminar by 1 month
    seminar_lag1 = np.roll(seminar, 1)
    seminar_lag1[0] = seminar_lag1[1]

    # Generate noise with AR(1)
    eps = np.zeros(T)
    phi = 0.6
    for t in range(1, T):
        eps[t] = phi * eps[t - 1] + np.random.normal(0, 1)

    # Response
    y = (
        beta0
        + trend
        + seasonal
        + beta1 * detailing
        + beta2 * seminar_lag1
        + beta3 * email
        + eps
    )

    # Put in DataFrame
    data = pd.DataFrame(
        {"y": y, "detailing": detailing, "seminar_lag1": seminar_lag1, "email": email},
        index=months,
    )

    return data


def diagnosis(result, data) -> None:
    resid = result.resid

    # 1. ACF/PACF of residuals
    # to check if residuals are white noise (no autocorrelation)
    # ACF/PACF of residuals should be mostly inside 95% CI.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(resid, lags=24, ax=axes[0])
    plot_pacf(resid, lags=24, ax=axes[1])
    plt.show()

    # 2. Normality check
    # to check if residuals are normally distributed for inference
    st.probplot(resid, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    # 3. Ljung–Box test (autocorrelation)
    # Ljung–Box p-value > 0.05 → no significant autocorrelation.
    # Lb_stat is like the sum of autocorrelation coefficients.
    # If Lb_stat is large, it means there is significant autocorrelation.
    lb_test = acorr_ljungbox(resid, lags=[12], return_df=True)
    print(lb_test)

    # 4. Plot fitted vs actual
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data["y"], label="Observed")
    plt.plot(data.index, result.fittedvalues, label="Fitted")
    plt.legend()
    plt.title("Observed vs Fitted")
    plt.show()

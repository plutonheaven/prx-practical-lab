# Chapter 2: SPP with differential corrections

In this project, we will compute the **Differential GNSS** solution, which consists in:
- using code observations
- using the observations from a nearby station to correct our own observations
- using a Weighted Least Squares algorithm to compute the solution at each epoch
- estimating the receiver position and clock

To do so, the following steps are proposed:
- apply corrections to code observations
- compute the predicted observations for corrected code observations
- compute the Jacobian matrix of the observation model
- compute the DGNSS solution for each epoch (snapshot solution)

Additionally, along the project, we will:
- observe the characteristics of the differentially-corrected code observations
- compare the DGNSS and the SPP solution

## 1.🧑Coding the DGNSS solution
### 1.1. Apply differential corrections
In the `src/gnss.py` module, write a function `apply_differential_corrections`.

Summary:
- Considering a *base* receiver with known position and a *rover* receiver, this function applies differential corrections using the following formula:
$$C_{cor}^s=C_{rov}^s - (C_{bas}^s - geo_{bas}^s)$$
  -  where $geo_{base}^s$ is the geometric distance between satellite $s$ and the base station.
- The difference must be made between code observations sharing the same constellation, prn, signal type and epoch.

Args:
- `df_rover: pd.DataFrame`, a dataframe loaded from a `prx` file for the rover receiver.
- `df_base: pd.DataFrame`, , a dataframe loaded from a `prx` file for the base receiver.
- `pos_base: np.array`, the base receiver precise position.

Returns:
- The same dataframe as `df_rover` with an additional column `C_obs_corr_m` containing the differentially-corrected code observations.

> 💡Hints:
> - You can define an `MultiIndex` for a dataframe based on several columns using the function `pd.DataFrame.set_index`.
> - Once an index is defined, you can perform operations (such as `df1 + df2` or `df1.col - df2.col`) on the whole dataframe.
> - If a value is absent on one of the 2 dataframes, the result is replaced by a `NaN`. You should remove rows containing `NaN` with the function `pd.DataFrame.dropna()`.
> - After performing the operation, you can move back the `MultiIndex` to columns with the function `pd.DataFrame.reset_index()`


Example of function call:
```python
import src.prx_tools as prx
import src.gnss as gnss
from src.constants import TLSG_2024001_ECEF

df_rover = prx.load_prx_file("data/TLSE00FRA_R_20240010000_01D_30S_MO.csv")
df_base = prx.load_prx_file("data/TLSG00FRA_R_20240010000_01D_30S_MO.csv")
df_rover = gnss.apply_differential_corrections(df_rover, df_base, TLSG_2024001_ECEF)
```

To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter2.py::test_differential_code_corrections
```

### 1.2. Corrected code observation model
We are now working with a corrected code observation model, which can be simplified to the terms depending only on the unknown states. All other terms (sat clock, satellite hardware biases, atmospheric delays) have been corrected.

In `src/gnss.py`, write a function `obs_model_corrected_code`.

Summary:
- The corrected code observation model is $C_{cor}^{sat}=\lVert \mathbf{r}_{rx} - \mathbf{r}^{sat}\rVert + {dt}_{rx}$

Args:
- `df: pd.DataFrame`, a dataframe loaded from a `prx` file for the rover receiver.
- `rx_pos: np.array`,  the receiver position in meters, expressed in the ECEF frame.
- `rx_clk: float=0`, the receiver clock bias in meters. A default value of 0 should be used.

Returns:
- - an `np.array` containing the predicted corrected code observation values.

To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter2.py::test_corrected_code_model
```

### 1.3. Jacobian matrix of the corrected code observation model
If you look at the corrected code observation model, you will realize that it has the same Jacobian matrix as the uncorrected code observation model.

Therefore, we can re-use your function `src.gnss::jacobian_code`!

### 1.4. Covariance matrix of the observations
We will use the same models as the one already coded in you function `src.gnss::obs_covariance_mat`.

### 1.5. Weighted Least Squares
The same estimation algorithm as for the SPP solution can be used. Just make sure that you correctly define the observation vector, observation model, jacobian matrix and observation covariance matrix.

### 1.6. Compute the DGNSS solution at each epoch
Create a script at the repository root named `main_dgnss.py` and write the code to compute the DGNSS solution.

You may re-use the same script structure as `main_spp.py`, except that you will now use differentially corrected code observations.

The DGNSS solution shall be recorded in a `results` dataframe containing the columns `"epoch","pos_x","pos_y","pos_z","clk_b"`.

## 2.🔎Analyze the DGNSS solution
### 2.1. Analyze the positioning results in the ENU frame
Use the functions `compute_enu_pos_error`, `plot_enu_error` and `plot_enu_error_cdf` from the module `src/helpers.py` to compute, plot and characterize the position error in the East/North/Up frame.
Use the function `helpers.analyze_results` to display various statistics on a `results` dataframe, and to compute the accuracy score for this solution.

Take the time to compare the SPP and the DGNSS solution:
- How do would you characterize the distribution of the positioning error, notably in terms of bias and dispersion?
- Can you give some hints about the reasons for those differences?  
<mark> How to collect those analyses??

## 2.2. Analyze the differentially-corrected code pseudorange residuals
In the module `src/gnss.py`, create a function `residuals_corrected_code`.

This function is very similar to your `residuals_uncorrected_code`, except that you will have to consider:
- the column `C_obs_corr_m` for the observation vector,
- the function `obs_model_corrected_code` for the observation model.

It shall return `df_prx` with an additional column `residual_code`.

To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter2.py::test_corrected_code_residual
```

Take the time to compare the observation residuals for the SPP and the DGNSS solution:
- What has been improved by applying differential corrections?
- What has been degraded?  
<mark> How to collect those analyses??

## 3.🏅Compete in the 2025 positioning leaderboard
Try to obtain the best accuracy for the DGNSS solution. You may take into account the satellite elevation in the positioning algorithm to see if it improves the solution accuracy.

Save your best solution in the folder `results/DGNSS.feather` using the function  `pd.DataFrame.to_feather` and add this file to your online repository. It will be evaluated and ranked against the other student teams' solutions.

You can use the following line to save your results dataframe in `feather` format:
```python
results.to_feather("results/DGNSS.feather")
```

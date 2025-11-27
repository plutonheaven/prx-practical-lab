# 📚 Chapter 2: Differential GNSS solution
**Objective**  
In this project, we will compute the **Differential GNSS** solution, which consists in:
- using code observations
- using the observations from a nearby station to correct our own observations
- using a Weighted Least Squares algorithm to compute the solution at each epoch
- estimating the receiver position and clock

**Steps Overview**
1. Implement core functions for DGNSS.
2. Run main_dgnss.py to compute position estimates.
3. Analyze ENU errors and residuals.
4. Compare SPP and DGNSS solutions.

📦**Deliverables**
- Complete python module `src/gnss.py`
- Python script: `main_dgnss.py`
- Updated powerpoint answer doc with plots and analysis

## 1.🧑‍💻Coding Tasks
All functions shall be coded in the `src/gnss.py` module.

### 1.1.Function: `apply_differential_corrections( df_rover: pd.DataFrame, df_base: pd.DataFrame, pos_base: np.array)`
**Summary:**
- Considering a *base* receiver with known position and a *rover* receiver, this function applies differential corrections using the following formula:
$$C_{cor}^s=C_{rov}^s - (C_{bas}^s - geo_{bas}^s)$$
  -  where $geo_{base}^s$ is the geometric distance between satellite $s$ and the base station.
- Ensure differential corrections are applied only to matching observations (same constellation, PRN, signal type, and epoch).

**Args:**
- `df_rover: pd.DataFrame`, a dataframe loaded from a `prx` file for the rover receiver.
- `df_base: pd.DataFrame`, , a dataframe loaded from a `prx` file for the base receiver.
- `pos_base: np.array`, the base receiver precise position.

**Returns:**
- The same dataframe as `df_rover` with an additional column `C_obs_corr_m` containing the differentially-corrected code observations.

**Tests:**  
To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter2.py::test_differential_code_corrections
```

Example of function call:
```python
import src.prx_tools as prx
import src.gnss as gnss
from src.constants import TLSG_2024001_ECEF

df_rover = prx.load_prx_file("data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.zip")
df_base = prx.load_prx_file("data/TLSG00FRA_R_20240010000_01D_30S_MO.csv.zip")
df_rover = gnss.apply_differential_corrections(df_rover, df_base, TLSG_2024001_ECEF)
```

> 💡**Hints:**
> - Define a `MultiIndex` for a dataframe based on several columns using the function `pd.DataFrame.set_index`.
> - Once an index is defined, you can perform operations (such as `df1 + df2` or `df1.col - df2.col`) on the whole dataframe.
> - If a value is absent on one of the 2 dataframes, the result is replaced by a `NaN`. You should remove rows containing `NaN` with the function `pd.DataFrame.dropna()`.
> - After performing the operation, you can move back the `MultiIndex` to columns with the function `pd.DataFrame.reset_index()`

### 1.2. Function: `obs_model_corrected_code(df: pd.DataFrame, rx_pos: np.array, rx_clk: float=0)`
We have to define a corrected code observation model, which can be simplified to the terms depending only on the unknown states. All other terms (sat clock, satellite hardware biases, atmospheric delays) have been corrected.

**Summary:**
- The corrected code observation model is $C_{cor}^{sat}=\lVert \mathbf{r}_{rx} - \mathbf{r}^{sat}\rVert + {dt}_{rx}$

**Args:**
- `df: pd.DataFrame`, a dataframe loaded from a `prx` file for the rover receiver.
- `rx_pos: np.array`,  the receiver position in meters, expressed in the ECEF frame.
- `rx_clk: float=0`, the receiver clock bias in meters. A default value of 0 should be used.

**Returns:**
- an `np.array` containing the predicted corrected code observation values.

**Test:**  
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

### 1.6. Script: `main_dgnss.py`
Create a script at the repository root named `main_dgnss.py` and write the code to compute the DGNSS solution, using:
- `data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.zip` for the rover receiver data,
- `data/TLSG00FRA_R_20240010000_01D_30S_MO.csv.zip` for the base receiver data,.

You may re-use the same script structure as `main_spp.py`, except that you will now use differentially corrected code observations.

The DGNSS solution shall be recorded in a `results` dataframe containing the columns `"epoch","pos_x","pos_y","pos_z","clk_b"`.

## 2.🔎Analysis Tasks
Complete the powerpoint document by pasting figures and writing analyses.

❓ **Questions to Answer**
1. Compare the DGNSS and the SPP solutions? How would you characterize the improvmement?
2. (optional) What patterns do you observe in residuals? What has changed compared to the SPP residuals?

### 2.1. Analyze the positioning results in the ENU frame
Use the functions `compute_enu_pos_error`, `plot_enu_error` and `plot_enu_error_cdf` from the module `src/helpers.py` to compute, plot and characterize the position error in the East/North/Up frame.
Use the function `helpers.analyze_results` to display various statistics on a `results` dataframe, and to compute the accuracy score for this solution.

### 2.2. Analyze the differentially-corrected code pseudorange residuals (optional)
For the estimation residual analysis, an new function has to be coded in `src/main.py`.

**Function:** `residuals_corrected_code(df_prx: pd.DataFrame, results: pd.DataFrame)`

**Summary:**
- This function is very similar to your `residuals_uncorrected_code`, except that you will have to consider:
  - the column `C_obs_corr_m` for the observation vector,
  - the function `obs_model_corrected_code` for the observation model.
- It shall return `df_prx` with an additional column `residual_code`.

**Test:**  
To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter2.py::test_corrected_code_residual
```

## 3.🏅Compete in the 2025 positioning leaderboard
Document your best solution in your report. It will be evaluated and ranked against the other student teams' solutions.

## 4. 💰Treasure hunt
Once you have a working DGNSS script, apply it to one of the couple of files in the folder `data/treasure_hunt`.
If the algorithm is working you should see a clue about the treasure hunt...
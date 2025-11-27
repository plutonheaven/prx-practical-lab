# 📚 Chapter 1: Single Point Positioning (SPP)

**Objective**  
In this project, we will compute the **Single Point Positioning** solution, which consists in:
- using code observations
- using the broadcast navigation message for correction computation (satellite position and clock, atmospheric effect, ...)
- using a (Weighted) Least Squares algorithm to compute the solution at each epoch
- estimating the receiver position and clock

**Steps Overview**
1. Implement core functions for SPP.
2. Run main_spp.py to compute position estimates.
3. Analyze ENU errors and residuals.
4. Improve accuracy using elevation-based strategies.

📦**Deliverables**
- Complete python module `src/gnss.py`
- Python script: `main_spp.py`
- Updated powerpoint answer doc with plots and analysis

## 1.🧑‍💻Coding Tasks
All functions shall be coded in the `src/gnss.py` module.

### 1.1. Function: `obs_model_code(df, rx_pos, rx_clk=0)`
**Summary:**
- Computes the predicted code observation from data `prx` file in a `pd.DataFrame`.
- The code observation model is:  
$C^{sat}=\lVert \mathbf{r}_{rx} - \mathbf{r}^{sat}\rVert + {dt}_{rx} + s_{rx}^{sat} - ({dt}^{sat}+{dt}^{sat}_{rel}-t_{gd}^{sat})+i_{rx}^{sat}+t_{rx}^{sat}$
    - $\mathbf{r}_{rx}$ is the receiver Earth Centered Earth Fixed (ECEF) position
    - $\mathbf{r}^{sat}$ is the satellite ECEF position
    - $s_{rx}^{sat}$ is the Sagnac effect (due to Earth rotation during signal propagation)
    - ${dt}_{rx}$ is the receiver clock bias
    - ${dt}^{sat}$ is the satellite clock bias
    - ${dt}^{sat}_{rel}$ is the relativistic clock effect (due to the difference of the gravitational potential and to the relative speed)
    - $t_{gd}^{sat}$ is the satellite hardware bias
    - $i_{rx}^{sat}$ is the ionospheric delay
    - $t_{rx}^{sat}$ is the tropospheric delay
    - All values are expressed in meters

**Args:**
- `df: pd.DataFrame`, a dataframe loaded from a `prx` (csv-formatted) file.
- `rx_pos: np.array`, the receiver position in meters, expressed in the Earth-Centered, Earth-Fixed (ECEF) frame.
- `rx_clk: float=0`, the receiver clock bias in meters. A default value of 0 should be used.

**Returns:**
- an `np.array` containing the predicted code observation values

**Tests:**  
To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter1.py::test_uncorrected_code_model
```

**Example Usage:**
```python
import src.prx_tools as prx
import src.gnss as gnss
from src.constants import TLSE_2024001_ECEF

df_prx = prx.load_prx_file("data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.zip")
code_predicted = gnss.obs_model_code(df_prx, TLSE_2024001_ECEF)
```

> 💡Hints:
> - To compute the geometric distance, use the `np.linalg.norm` function.
> - The relevant information is contained in the columns `"sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m", "sat_clock_offset_m", "sagnac_effect_m", "relativistic_clock_effect_m", "sat_code_bias_m", "iono_delay_m", "tropo_delay_m"` of the dataframe.

### 1.2. Function: `jacobian_code(df: pd.DataFrame, rx_pos: np.array)` 
**Summary:**
- Computes the jacobian matrix of the code observation model at a receiver position `rx_pos`.
- The elements of the jacobian matrix are equal to the partial derivative of the observation model with regards to each state.
    - for a position state `x/y/z`, $\frac{\partial C}{\partial x}=\frac{x_{rx} - x^{sat}}{\lVert \mathbf{r}_{rx} - \mathbf{r}^{sat} \rVert}$
    - for the clock bias state `b`, $\frac{\partial C}{\partial b}=1$

**Args:**
- `df: pd.DataFrame`, a dataframe loaded from a `prx` file.
- `rx_pos: np.array`, the receiver position in meters, expressed in the ECEF frame.

**Returns:**
- an `np.array` of shape `(m,4)` representing the Jacobian matrix, `m` being the number of observations.

**Test:**
To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter1.py::test_jacobian_code
```

**Example Usage:**
```python
import src.prx_tools as prx
import src.gnss as gnss
from src.constants import TLSE_2024001_ECEF

df_prx = prx.load_prx_file("data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.zip")
jac = gnss.jacobian_code(df_prx, TLSE_2024001_ECEF)
```

### 1.3. Function: `obs_covariance_mat(df: pd.DataFrame, model: str="identical)`
**Summary:**
- Computes the observation covariance matrix.
- At first, a single model has to be coded with a variance which is identical for each observation. This will result in a covariance matrix equal to the identity matrix multiplied by the observation variance.

**Args:**
- `df: pd.DataFrame`, a DataFrame loaded from a `prx` file.
- `model: str="identical"`, a string describing the model. Default value should be `"identical"`, to model observations that are independent and identically-distributed.

**Returns:**
- an `np.array` of shape `(m,m)` containing the observation covariance matrix, `m` being the number of observations.

**Test:**
To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter1.py::test_cov_mat_identical
```

**Example Usage:**
```python
import src.prx_tools as prx
import src.gnss as gnss

df_prx = prx.load_prx_file("data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.zip")
cov = gnss.obs_covariance_mat(df_prx, "identical")
```
### 1.4. Function: `wls(obs: np.array, obs_pred: np.array, jac: np.array, cov: np.array)`
**Summary:**
- Computes the linear weighted least square estimate.
- We assume that the observation model is linearized around a good approximation of the true state, so that only a single iteration is necessary.
- Let us assume that we linearized the observation model around the state vector $\mathbf{x}_0$. Then, the WLS estimate of $\mathbf{x}$ is $\hat{\mathbf{x}}=\mathbf{x_0} + \delta \hat{\mathbf{x}}$, where the state update is $\delta \hat{\mathbf{x}} =(\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{R}^{-1}(\mathbf{y}-\mathbf{h}(\mathbf{\mathbf{x}_0}))$

**Args:**
- `obs: np.array`, the observations
- `obs_pred: np.array`, the predicted observations
- `jac: np.array`, the observation model jacobian matrix
- `cov: np.array`, the observation covariance matrix

**Returns:**
- a `np.array` of shape`(n,)` containing the **state update** to be applied to the linearization point, `n` being number of states.

**Test:**
```bash
uv run pytest tests/test_chapter1.py::test_wls
```

**Example Usage:**
```python
import src.prx_tools as prx
import src.gnss as gnss

df = ...
obs = ...
obs_pred = gnss.obs_model_code(df, TLSE_2024001_ECEF)
jac = gnss.jacobian_code(df, TLSE_2024001_ECEF)
cov = gnss.obs_covariance_mat(df, "identical")
dx_est = gnss.wls(obs, obs_pred, jac, cov)
```

### 1.5. Script: `main_spp.py`
Now, it is time to use all the functions added to the `src/gnss.py` in a script!

Create a script at the repository root named `main_spp.py` and write the code to compute the SPP solution on the `prx` file ` data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.zip`.  
Store the estimated position and clock bias for each epoch in a `pd.DataFrame` named `results` and having the columns `"epoch","pos_x","pos_y","pos_z","clk_b"`

When the `prx` file is loaded, the resulting dataframe contains observations for several epochs.  
The estimation algorithm has to be applied at each epoch, resulting in a 4-element estimated state vector at each epoch.

> 💡Hints:
> - Use a `for` loop on the iterator created by `pd.DataFrame.groupby("time_of_reception_in_receiver_time")` to loop over the different epochs.
> - For each epoch, compute the estimated state vector.
> - Store the estimated position and clock bias for each epoch in a `pd.DataFrame` named `results` and having the columns `"epoch","pos_x","pos_y","pos_z","clk_b"`

## 2.🔎Analysis Tasks
Complete the powerpoint document by pasting figures and writing analyses.

❓ **Questions to Answer**
1. What is the ENU error distribution for SPP?
2. (optional) What patterns do you observe in residuals?
3. (optional) Which factors contribute to large residuals?

### 2.1. Analyze the positioning results in the ENU frame
Use the functions `compute_enu_pos_error`, `plot_enu_error` and `plot_enu_error_cdf` from the module `src/helpers.py` to compute, plot and characterize the position error in the East/North/Up frame.

Additionally, the function `helpers.analyze_results` display various statistics on a `results` dataframe, and returns an **accuracy score** based on the average between the 50-th percentile and the 95-th percentile of the 3D position error.

Here is an example to visualize the score of your solution:
```python
results = ...  # computes the SPP solution and store the pos_x, pos_y and pos_z values
results = helpers.compute_enu_pos_error(  # adds the ENU position error as new columns
    results, TLSE_2024001_ECEF
)
helpers.analyze_results(results)  # display error stats and score
```

### 2.2. Estimation residual analysis (optional)
For the estimation residual analysis, an additional function has to be coded in `src/main.py`.

**Function:** `residuals_uncorrected_code(df_prx: pd.DataFrame, results: pd.DataFrame)`

**Summary:**
- The estimation residuals are obtained by computing $\mathbf{y}-\mathbf{h}(\mathbf{\hat{x}})$, where:
  - $\mathbf{y}$ is the observation vector,
  - $\mathbf{h}(\mathbf{x})$ is the observation model applied at state vector $\mathbf{x}$,
  - $\mathbf{\hat{x}}$ is the estimation result.
- This function adds a column `residual_code` with the residual values to the `prx` dataframe

**Args:**
- `df_prx: pd.DataFrame`, the dataframe loaded from a `prx` file.
- `results: pd.DataFrame`, the `results` dataframe containing the estimated solution in the columns `"epoch","pos_x","pos_y","pos_z","clk_b"`

**Returns:**
- `df_prx` with an additional column `residual_code`

**Test:**  
To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests/test_chapter1.py::test_code_residuals
```

> 💡**Hints:**
> - Use `DataFrame.merge` ([doc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)) to add a column with the estimated states corresponding to the same epochs. You have to specify the correct colum to use on the left and right dataframes, with the arguments `left_on` and `right_on`.
> - After merging the estimated solution and `df_prx`, you can re-use your function `gnss.obs_model_code` on `df_prx.groupby("time_of_reception_in_receiver_time")`
> - Once this function works, you can plot the residual time series for each satellite on the same plot using `helpers.plot_residuals_code`.

## 3. 📈Improve the SPP solution (optional)
One of the main sources of errors in the observations are the atmospheric errors, which are larger when the satellite elevation is low.

💡Let us try to mitigate the effect of low-elevation satellite with 2 solutions:
- applying an elevation mask
- down-weighting the observations at low elevation through the observation covariance matrix

### 3.1. Add an elevation mask
In your `prx_tools.load_prx_file` function:
- add an additional input argument  with a default value equal to 0: `mask_el_deg: float=0`.
- filter out the low elevation satellites

Test your modification (and make sure the initial test still passes) with the following command:
```bash
uv run pytest tests/test_chapter1.py::test_load_with_elevation_mask
uv run pytest tests/test_chapter0.py::test_load_prx_file
```

Run the SPP positioning algorithm with different elevation mask and choose the best option based on the score provided by `helpers.analyze_results`.

### 3.2. Use an elevation-dependent observation variance model
Add an alternative `model`, named `elevation`, to your function `gnss.obs_covariance_mat`.
A simple proposition is to model the variance as $(\sigma_{code}/sin(el))^2$.

Test your modification (and make sure the initial test still passes) with the following command:
```bash
uv run pytest tests/test_chapter1.py::test_cov_mat_elevation
uv run pytest tests/test_chapter1.py::test_cov_mat_identical
```

## 4.🏅Compete in the 2025 positioning leaderboard
Document your best solution in your report. It will be evaluated and ranked against the other student teams' solutions.

## 5. 💰Treasure hunt
Once you have a working SPP script, apply it to one of the couple of files in the folder `data/treasure_hunt`.
If the algorithm is working you should see a clue about the treasure hunt...

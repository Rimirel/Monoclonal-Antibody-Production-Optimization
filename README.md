# Monoclonal Antibody Production Optimization

This project uses machine learning to predict monoclonal antibody production and identify economically favourable bioreactor operating conditions.

## Project Description

The objective of this project is to develop a data-driven approach for optimizing monoclonal antibody (mAb) production in a 1 L bioreactor. Experimental batch data are used to train and compare regression models that predict mAb concentration based on initial culture conditions and harvest time. The selected model is then used to estimate operating conditions that maximize space-time profit while considering culture costs, production value, and prediction uncertainty.

## Highlights

- **Prepare** a dataset from 30 selected bioreactor batches.
- **Compare** three machine learning regression models for mAb concentration prediction.
- **Optimize** initial culture conditions and harvest time using the selected model.
- **Estimate** economic performance and 95% prediction intervals for the optimized conditions.

## Environment

The project was developed using the following environment:

**Python**
- Python 3.14.7

**Main packages** 
- Matplotlib 3.11.0
- NumPy 2.5.2
- Pandas 3.0.5
- Scikit-learn 1.9.0
- SciPy 1.18.0


The complete environment configuration is provided in `environment.yaml`.

## Code Design

The project is organized into separate files to keep data preparation, model development, optimization, and execution independent. The main workflow is launched through `main.py`, while reusable functions, classes, and constants are stored in the `src` folder.

### 1. Constants and project configuration

The `src/constants.py` file contains the main configuration values used throughout the project. These include file paths, dataset column names, model settings, optimization bounds, and economic parameters. Keeping these values in one location makes the code easier to maintain and modify without changing several files.

### 2. Batch data handling

The `src/classes.py` file defines the `Batch` class, which represents an individual bioreactor batch. It handles the loading and organization of batch measurements and provides methods to retrieve mAb concentration at a specified harvest time.

The class also supports the calculation of production economics. It uses the predicted or measured mAb concentration to calculate culture value, culture cost, profit, space-time profit, and space-time yield. This keeps batch-related calculations together rather than repeating them throughout the project.

### 3. Data preparation and model development

The `src/functions.py` file contains the reusable functions for preparing the dataset and developing the predictive models. The initial conditions and time-series measurements from the selected batches are loaded and combined into a structured dataset.

Each observation contains the initial glucose concentration, initial glutamine concentration, initial viable cell density (VCD), harvest time, mAb concentration, and batch ID. The batch ID is retained to identify measurements originating from the same experimental batch.

The functions also define the candidate regression models, divide the data into training, calibration, and test partitions, evaluate model performance, and prepare the results for comparison. Group-based splitting is used with batch ID so that measurements from the same batch are not distributed across different partitions.

Additional functions perform the optimization of operating conditions, calculate prediction intervals, and export tables and figures. This structure allows the same preparation and evaluation procedures to be reused consistently.

### 4. Main execution

The `main.py` file coordinates the complete workflow. It creates the required output folders, loads the initial conditions and batch data, prepares the dataset, trains and evaluates the candidate models, and selects the model used for optimization.

The selected model is then used to estimate mAb production over the defined operating ranges. The optimization routine searches for the combination of initial glucose, initial glutamine, initial VCD, and harvest time that gives the highest predicted space-time profit. Finally, the script calculates prediction intervals and saves the datasets, comparison tables, optimization results, and figures.

Only `main.py` needs to be executed to run the complete workflow.

## Analysis

### 1. Selection of experimental conditions

The original experimental space contained 1,029 possible operating conditions. Since evaluating every condition was not practical for this project, 30 batches were selected to develop the predictive model.

The selection was intended to represent variations in three initial culture parameters: glucose concentration, glutamine concentration, and initial viable cell density (VCD). The selected conditions were chosen from the available combinations to provide coverage across these variables rather than concentrating on only one region of the experimental space.

Each batch contains time-series measurements of glucose, glutamine, lactate, ammonia, VCD, total cell density (TCD), and mAb concentration. The prepared dataset contains 2,550 measurements from the 30 selected batches, with 85 measurements per batch.

This approach provides multiple observations at different culture times and allows the model to learn how mAb concentration changes throughout a batch. However, using only 30 of the 1,029 possible conditions limits the amount of experimental variation represented in the dataset. The predictions should therefore be interpreted within the context of the selected conditions and the defined optimization ranges.

### 2. Data preparation and prediction target

The prediction target is mAb concentration in mg/L. Four input variables are used: initial glucose concentration, initial glutamine concentration, initial VCD, and harvest time.

Including harvest time is important because mAb concentration changes during cultivation. A model based only on the initial conditions would not represent the effect of harvesting at different stages of the batch.

The initial lactate, ammonia, and mAb concentrations are assumed to be zero. Initial TCD is assumed to be equal to initial VCD. These assumptions provide consistent starting conditions for the batch calculations.

The data are divided into training, calibration, and test partitions using the batch ID as the grouping variable. This prevents measurements from the same batch from appearing in more than one partition. This is important because measurements collected from one batch are related to each other. Randomly splitting individual measurements could otherwise result in overly optimistic model performance.

### 3. Model selection

Three regression models were compared: Random Forest, Gradient Boosting, and Extra Trees. These models were selected because they can represent nonlinear relationships between the initial culture conditions, harvest time, and mAb concentration.

Random Forest uses multiple decision trees and averages their predictions. This reduces the influence of individual trees and allows the model to capture nonlinear patterns. The model was configured with 200 trees.

Gradient Boosting builds trees sequentially, with each new tree attempting to correct errors made by the previous trees. It was configured with 150 estimators, a learning rate of 0.05, and a maximum tree depth of 3.

Extra Trees also combines multiple decision trees, but introduces additional randomness when selecting split thresholds. It was configured with 200 trees.

The models were compared using the coefficient of determination (R²), root mean squared error (RMSE), and mean absolute error (MAE).

| Model | R² | RMSE (mg/L) | MAE (mg/L) |
|---|---:|---:|---:|
| Random Forest | 0.9146 | 0.0733 | 0.0600 |
| Gradient Boosting | 0.7590 | 0.1231 | 0.0961 |
| Extra Trees | 0.3631 | 0.2001 | 0.1522 |

Random Forest provided the best overall performance, with the highest R² and the lowest RMSE and MAE. Its R² of 0.9146 indicates that it explained a large proportion of the variation in the evaluated data. Its lower prediction errors also indicate that its predictions were closer to the measured mAb concentrations than those of the other two models.

Gradient Boosting achieved intermediate performance, while Extra Trees had the largest errors. Based on these results, Random Forest was selected for the optimization stage.

The comparison is useful for identifying the strongest candidate among the tested models. However, performance depends on the selected batches and the data partitions. The model should not be assumed to generalize equally well to conditions outside the experimental range.

### 4. Optimization approach and economic evaluation

The selected Random Forest model was used to predict mAb concentration for different combinations of initial conditions and harvest time. The optimization objective was to maximize space-time profit rather than simply maximize mAb concentration.

The economic calculations assume a 1 L bioreactor and one day of downtime between batches. Culture cost is calculated using:

**Culture cost ($/L) = 10 × glucose (mM) + 20 × glutamine (mM) + 0.0005 × initial VCD (cells/mL)**

Culture value is calculated as:

**Culture value ($/L) = 1000 × mAb concentration (mg/L)**

Profit is the difference between culture value and culture cost:

**Profit ($/L) = Culture value − Culture cost**

The downtime is included in the calculation of space-time performance:

**Space-time profit ($/L/day) = Profit / (Harvest time + 1)**

**Space-time yield (mg/L/day) = mAb concentration / (Harvest time + 1)**

The optimization was performed using differential evolution. The search ranges were 5–35 mM for initial glucose, 1–7 mM for initial glutamine, 200,000–1,400,000 cells/mL for initial VCD, and 0–28 days for harvest time.

These ranges constrain the search to the defined operating space. The optimization considers both the predicted production and the cost of the initial culture conditions. This is important because a condition that produces more mAb may not necessarily provide the highest space-time profit if it also requires substantially higher culture costs or a longer cultivation period.

### 5. Optimal operating conditions and results

The optimization identified the following combination:

| Parameter | Predicted optimum |
|---|---:|
| Initial glucose | 12.50 mM |
| Initial glutamine | 2.54 mM |
| Initial VCD | 500,488 cells/mL |
| Harvest time | 2.84 days |
| Predicted mAb concentration | 0.968 mg/L |

At these conditions, the estimated culture cost is $426.07/L and the estimated culture value is $967.70/L. This gives a predicted profit of $541.63/L.

After accounting for the one-day downtime between batches, the estimated space-time profit is **$141.08/(L·day)**. The estimated space-time yield is **0.252 mg/(L·day)**.

The relatively short predicted harvest time contributes to the space-time performance because the downtime is included in the denominator. The result also reflects the balance between mAb production and the cost of the initial glucose, glutamine, and VCD levels.

The optimization result is a model-based estimate, not an experimentally confirmed optimum. The predicted harvest time and initial conditions should be tested experimentally before being adopted as operating conditions.

### 6. Prediction uncertainty

A 95% prediction interval was calculated using the calibration data and the distribution of absolute prediction residuals. The resulting interval provides an indication of the uncertainty associated with the model predictions.

For the optimized conditions, the estimated intervals are:

| Performance measure | Predicted value or interval |
|---|---:|
| mAb concentration | 0.968 mg/L |
| mAb 95% prediction interval | 0.800–1.136 mg/L |
| Culture value 95% prediction interval | $799.75–$1,135.65/L |
| Space-time profit 95% prediction interval | $97.33–$184.82/(L·day) |
| Space-time yield 95% prediction interval | 0.208–0.296 mg/(L·day) |

The intervals show that the predicted performance is subject to uncertainty. In particular, the range for space-time profit is important because it reflects how uncertainty in mAb concentration affects the economic objective.

These intervals should not be interpreted as guaranteed ranges for future experimental batches. Their reliability depends on how representative the calibration data are and on the assumptions used in the interval calculation. Since measurements from a batch are related, a batch-level validation approach would provide a useful additional assessment of uncertainty.

### 7. Limitations and recommendations

The main limitation is the relatively small number of selected batches compared with the full experimental space. Although the selected conditions provide variation in the initial inputs, additional batches would help assess whether the model can generalize to other combinations.

The Random Forest model also does not provide a mechanistic description of the biological processes involved in mAb production. It identifies relationships from the available data, but its predictions should not be treated as proof of biological causation.

Another limitation is that the optimization searches continuous ranges using model predictions. The resulting optimum may not correspond to a condition that was directly tested in the original dataset. Experimental validation is therefore necessary, particularly around the predicted optimum.

Overall, Random Forest was the strongest of the three tested models and provided a useful basis for exploring the economic trade-off between production, culture cost, and batch duration. The predicted conditions offer a starting point for further experiments, while the prediction intervals highlight the uncertainty that should be considered before making process decisions.

### 8. Supporting results

The project exports the following files to support the analysis:

- `tables/prepared_dataset.csv` — prepared measurements from the selected batches.
- `tables/model_comparison.csv` — performance metrics for the three regression models.
- `tables/optimal_conditions.csv` — optimized operating conditions and economic results.
- `tables/prediction_intervals.csv` — calculated 95% prediction intervals.
- `figures/model_comparison.png` — visual comparison of model performance.
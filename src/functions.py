
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.classes import Batch

from src.constants import (
    DATA_DIR,
    INITIAL_CONDITIONS_FILE,
    FIGURES_DIR,
    TABLES_DIR,
    BATCH_FILE_PATTERN,
    BATCH_DATA_COLUMNS,
    INITIAL_CONDITIONS_COLUMNS,
    GLUCOSE_COST_PER_MM,
    GLUTAMINE_COST_PER_MM,
    VCD_COST_COEFFICIENT,
    MAB_VALUE_PER_MG_L,
    DOWNTIME_DAYS,
    HOURS_PER_DAY,
    GLUCOSE_BOUNDS,
    GLUTAMINE_BOUNDS,
    VCD_BOUNDS,
    HARVEST_TIME_BOUNDS_DAYS,
    RANDOM_STATE
)


# 1. DIRECTORY MANAGEMENT


def create_project_directories():
    """Create the output directories if they do not exist."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)



# 2. DATA LOADING


def load_initial_conditions(file_path=INITIAL_CONDITIONS_FILE):
    """
    Load the initial conditions of all available batches.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Initial conditions file not found: {file_path}"
        )

    conditions = pd.read_csv(file_path)

    missing_columns = [
        column for column in INITIAL_CONDITIONS_COLUMNS
        if column not in conditions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing initial condition columns: {missing_columns}"
        )

    if conditions["Batch ID"].duplicated().any():
        raise ValueError("Duplicate Batch IDs found.")

    return conditions


def load_batches(
    conditions,
    data_directory=DATA_DIR,
    batch_ids=None
):
    """
    Load available batch CSV files and create Batch objects.

    Parameters
    ----------
    conditions : DataFrame
        Initial conditions for all batches.

    data_directory : Path
        Directory containing the batch CSV files.

    batch_ids : list, optional
        Specific Batch IDs to load. If None, all available
        batch files are loaded.
    """

    data_directory = Path(data_directory)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_directory}"
        )

    if batch_ids is None:
        files = sorted(data_directory.glob("batch_*.csv"))

    else:
        files = [
            data_directory / BATCH_FILE_PATTERN.format(batch_id)
            for batch_id in batch_ids
        ]

    batches = []

    conditions_indexed = conditions.set_index("Batch ID")

    for file_path in files:

        if not file_path.exists():
            print(f"Warning: {file_path.name} was not found.")
            continue

        try:
            batch_id = int(file_path.stem.split("_")[-1])

        except ValueError:
            print(f"Skipping invalid filename: {file_path.name}")
            continue

        if batch_id not in conditions_indexed.index:
            print(f"Warning: No initial conditions for batch {batch_id}.")
            continue

        initial = conditions_indexed.loc[batch_id]

        batch = Batch(
            batch_id=batch_id,
            glucose=float(initial["Glucose [mM]"]),
            glutamine=float(initial["Glutamine [mM]"]),
            initial_vcd=float(initial["VCD [cells/mL]"])
        )

        try:
            batch.load_data(file_path)
            batches.append(batch)

        except (ValueError, FileNotFoundError) as error:
            print(f"Could not load batch {batch_id}: {error}")

    if not batches:
        raise ValueError(
            "No valid batch files were loaded. "
            "Check the datasets directory."
        )

    print(f"Successfully loaded {len(batches)} batches.")

    return batches



# 3. DATA PREPARATION


def create_training_dataset(batches):
    """
    Combine the time-series data from all batches.

    Features:
        Initial glucose concentration
        Initial glutamine concentration
        Initial VCD
        Harvest time in days

    Target:
        mAb concentration in mg/L

    Batch ID is retained for grouped model evaluation.
    """

    all_data = []

    for batch in batches:

        if batch.data.empty:
            continue

        batch_data = batch.data.copy()
        batch_data["Batch ID"] = batch.batch_id
        batch_data["Initial Glucose [mM]"] = batch.glucose
        batch_data["Initial Glutamine [mM]"] = batch.glutamine
        batch_data["Initial VCD [cells/mL]"] = batch.initial_vcd
        batch_data["Harvest Time [d]"] = (batch_data["Time [h]"] / HOURS_PER_DAY)
        all_data.append(batch_data)

    if not all_data:
        raise ValueError("No experimental data available.")

    dataset = pd.concat(all_data, ignore_index=True)

    feature_columns = [
        "Initial Glucose [mM]",
        "Initial Glutamine [mM]",
        "Initial VCD [cells/mL]",
        "Harvest Time [d]"
    ]

    required_columns = feature_columns + [
        "mAb [mg/L]",
        "Batch ID"
    ]

    dataset = dataset[required_columns].copy()

    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    dataset.dropna(inplace=True)

    if dataset.empty:
        raise ValueError("The prepared dataset is empty.")

    return dataset.reset_index(drop=True)


# ============================================================
# 4. MACHINE LEARNING MODELS
# ============================================================

def get_candidate_models():
    """
    Define the regression models to compare.
    """

    models = {

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            loss="squared_error",
            random_state=RANDOM_STATE
        )
    }

    return models


def split_data_by_batch(dataset):
    """
    Split the dataset into training, calibration and test sets.

    All observations from one batch remain in the same subset.
    """

    X = dataset.drop(columns=["mAb [mg/L]", "Batch ID"])

    y = dataset["mAb [mg/L]"]

    groups = dataset["Batch ID"]

    if groups.nunique() < 5:
        raise ValueError(
            "At least five different batches are required "
            "for grouped model evaluation."
        )

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_cal_indices, test_indices = next(
        first_split.split(X, y, groups)
    )

    X_train_cal = X.iloc[train_cal_indices]
    y_train_cal = y.iloc[train_cal_indices]
    groups_train_cal = groups.iloc[train_cal_indices]

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    train_relative, calibration_relative = next(
        second_split.split(
            X_train_cal,
            y_train_cal,
            groups_train_cal
        )
    )

    train_indices = train_cal_indices[train_relative]
    calibration_indices = train_cal_indices[calibration_relative]

    return (
        X.iloc[train_indices],
        X.iloc[calibration_indices],
        X.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[calibration_indices],
        y.iloc[test_indices]
    )


def evaluate_models(dataset):
    """
    Train and compare the candidate regression models.

    The best model is selected based on test RMSE.

    A separate calibration subset is retained for
    estimating prediction intervals.
    """

    (X_train,X_calibration,X_test,y_train,y_calibration,y_test) = split_data_by_batch(dataset)
    candidate_models = get_candidate_models()

    results = []
    fitted_models = {}

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results.append({"Model": name,"R2": r2,"RMSE [mg/L]": rmse,"MAE [mg/L]": mae})
        fitted_models[name] = model
        print(f"{name}: R2 = {r2:.4f}, "f"RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    results_df = pd.DataFrame(results)
    results_df.sort_values(
        by="RMSE [mg/L]",
        ascending=True,
        inplace=True
    )

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    # Calibration residuals for prediction intervals
    calibration_predictions = best_model.predict(X_calibration)
    calibration_errors = np.abs(
        y_calibration.to_numpy() - calibration_predictions
    )

    # Conservative empirical 95% error quantile
    prediction_error = float(
        np.quantile(calibration_errors, 0.95, method="higher")
    )

    print(f"\nSelected model: {best_model_name}")
    print(f"Calibration error: {prediction_error:.4f} mg/L")

    return {
        "model": best_model,
        "model_name": best_model_name,
        "metrics": results_df,
        "prediction_error": prediction_error,
        "X_test": X_test,
        "y_test": y_test
    }



# 5. PREDICTION INTERVALS

def predict_with_interval(model,prediction_error,glucose,glutamine,vcd,harvest_time):
    """
    Predict mAb concentration and estimate an empirical
    95% prediction interval using calibration residuals.
    """

    input_data = pd.DataFrame([{
        "Initial Glucose [mM]": glucose,
        "Initial Glutamine [mM]": glutamine,
        "Initial VCD [cells/mL]": vcd,
        "Harvest Time [d]": harvest_time
    }])

    prediction = float(model.predict(input_data)[0])

    lower = max(0.0, prediction - prediction_error)

    upper = max(0.0, prediction + prediction_error)

    return {
        "prediction": prediction,
        "lower": lower,
        "upper": upper
    }



# 6. ECONOMIC CALCULATIONS


def calculate_culture_cost(glucose, glutamine, vcd):
    """
    Calculate cell culture cost in $/L.
    """

    return (GLUCOSE_COST_PER_MM * glucose+ GLUTAMINE_COST_PER_MM * glutamine+ VCD_COST_COEFFICIENT * vcd)


def calculate_culture_value(mab_concentration):
    """
    Calculate cell culture value in $/L.
    """

    return MAB_VALUE_PER_MG_L * mab_concentration


def calculate_profit(glucose, glutamine, vcd, mab_concentration):
    """
    Calculate batch profit in $/L.
    """

    return ( calculate_culture_value(mab_concentration)- calculate_culture_cost(glucose, glutamine, vcd))

def calculate_space_time_profit(glucose,glutamine,vcd,harvest_time,mab_concentration):
    """
    Calculate space-time profit in $/(L*d).
    """

    total_cycle_time = harvest_time + DOWNTIME_DAYS

    if total_cycle_time <= 0:
        raise ValueError("Total cycle time must be positive.")

    profit = calculate_profit(glucose,glutamine,vcd,mab_concentration)
    return profit / total_cycle_time


def calculate_space_time_yield(harvest_time, mab_concentration):
    """
    Calculate mAb space-time yield in mg/(L*d).
    """

    total_cycle_time = harvest_time + DOWNTIME_DAYS

    if total_cycle_time <= 0:
        raise ValueError("Total cycle time must be positive.")

    return mab_concentration / total_cycle_time



# 7. OPTIMIZATION

def optimize_space_time_profit(model):
    """
    Find the initial culture conditions and harvest time
    that maximize predicted space-time profit.

    Differential Evolution is used for optimization.
    """

    bounds = [GLUCOSE_BOUNDS,GLUTAMINE_BOUNDS,VCD_BOUNDS,HARVEST_TIME_BOUNDS_DAYS]

    def objective(variables):

        glucose, glutamine, vcd, harvest_time = variables

        input_data = pd.DataFrame([{
            "Initial Glucose [mM]": glucose,
            "Initial Glutamine [mM]": glutamine,
            "Initial VCD [cells/mL]": vcd,
            "Harvest Time [d]": harvest_time
        }])

        predicted_mab = float(model.predict(input_data)[0])
        predicted_profit = calculate_space_time_profit(glucose,glutamine,vcd,harvest_time,predicted_mab)

        # Differential Evolution minimizes the objective
        return -predicted_profit

    result = differential_evolution(objective,bounds=bounds,seed=RANDOM_STATE,maxiter=150,popsize=12,tol=0.01,polish=True)

    glucose, glutamine, vcd, harvest_time = result.x

    predicted_mab = float(
        model.predict(pd.DataFrame([{
            "Initial Glucose [mM]": glucose,
            "Initial Glutamine [mM]": glutamine,
            "Initial VCD [cells/mL]": vcd,
            "Harvest Time [d]": harvest_time
        }]))[0]
    )

    culture_cost = calculate_culture_cost(glucose,glutamine,vcd)
    culture_value = calculate_culture_value(predicted_mab)
    profit = culture_value - culture_cost
    space_time_profit = profit / (harvest_time + DOWNTIME_DAYS)
    space_time_yield = predicted_mab / (harvest_time + DOWNTIME_DAYS)

    return {
        "Glucose [mM]": glucose,
        "Glutamine [mM]": glutamine,
        "VCD [cells/mL]": vcd,
        "Harvest Time [d]": harvest_time,
        "Predicted mAb [mg/L]": predicted_mab,
        "Culture Cost [$/L]": culture_cost,
        "Culture Value [$/L]": culture_value,
        "Profit [$/L]": profit,
        "Space-Time Profit [$/L/d]": space_time_profit,
        "Space-Time Yield [mg/L/d]": space_time_yield,
        "Optimization Success": result.success
    }



# 8. OPTIMAL CONDITION PREDICTION INTERVALS

def calculate_optimal_prediction_intervals(model,prediction_error,optimum):
    """
    Calculate 95% empirical prediction intervals for
    mAb concentration, culture value, profit,
    space-time profit and space-time yield.
    """

    glucose = optimum["Glucose [mM]"]
    glutamine = optimum["Glutamine [mM]"]
    vcd = optimum["VCD [cells/mL]"]
    harvest_time = optimum["Harvest Time [d]"]
    mab_interval = predict_with_interval(model,prediction_error,glucose,glutamine,vcd,harvest_time)
    lower_mab = mab_interval["lower"]
    upper_mab = mab_interval["upper"]
    culture_cost = optimum["Culture Cost [$/L]"]
    lower_value = calculate_culture_value(lower_mab)

    upper_value = calculate_culture_value(upper_mab)
    lower_profit = lower_value - culture_cost
    upper_profit = upper_value - culture_cost
    cycle_time = harvest_time + DOWNTIME_DAYS

    return {
        "mAb 95% PI [mg/L]": (lower_mab,upper_mab),
        "Culture Value 95% PI [$/L]": (lower_value,upper_value),
        "Space-Time Profit 95% PI [$/L/d]": (lower_profit / cycle_time,upper_profit / cycle_time),
        "Space-Time Yield 95% PI [mg/L/d]": (lower_mab / cycle_time,upper_mab / cycle_time)
    }



# 9. RESULTS EXPORT

def save_results(dataset, evaluation, optimum, intervals):
    """
    Save the prepared dataset, model comparison and
    optimization results as CSV files.
    """

    create_project_directories()
    dataset.to_csv(TABLES_DIR / "prepared_dataset.csv",index=False)
    evaluation["metrics"].to_csv(TABLES_DIR / "model_comparison.csv",index=False)
    pd.DataFrame([optimum]).to_csv(TABLES_DIR / "optimal_conditions.csv",index=False)

    interval_rows = []

    for metric, values in intervals.items():
        interval_rows.append({"Metric": metric,"Lower 95%": values[0],"Upper 95%": values[1]})

    pd.DataFrame(interval_rows).to_csv(TABLES_DIR / "prediction_intervals.csv",index=False)

    print("\nResults successfully saved.")


# ============================================================
# 10. MODEL PERFORMANCE PLOT
# ============================================================

def plot_model_comparison(metrics):
    """
    Create a bar chart comparing the test RMSE
    of the candidate models.
    """

    create_project_directories()
    plt.figure(figsize=(8, 5))
    plt.bar(metrics["Model"],metrics["RMSE [mg/L]"])
    plt.ylabel("RMSE [mg/L]")
    plt.xlabel("Regression Model")
    plt.title("Comparison of Regression Models")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png",dpi=300)

    plt.close()

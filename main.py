
import sys

from src.functions import (
    load_initial_conditions,
    load_batches,
    create_training_dataset,
    evaluate_models,
    optimize_space_time_profit,
    calculate_optimal_prediction_intervals,
    save_results,
    plot_model_comparison,
    create_project_directories
)


def main():
    """
    Main workflow for the monoclonal antibody production
    optimization project.
    """

    print("=" * 60)
    print("MONOCLONAL ANTIBODY PRODUCTION OPTIMIZATION")
    print("=" * 60)

    try:


        # 1. CREATE PROJECT DIRECTORIES


        print("\n[1/7] Preparing project directories...")
        create_project_directories()

        # 2. LOAD INITIAL CONDITIONS

        print("\n[2/7] Loading initial conditions...")
        conditions = load_initial_conditions()
        print(f"Number of available conditions: {len(conditions)}")


        # 3. LOAD EXPERIMENTAL BATCHES

        print("\n[3/7] Loading experimental batches...")
        batches = load_batches(conditions)
        print(f"Number of batches loaded: {len(batches)}")


        # 4. PREPARE TRAINING DATASET

        print("\n[4/7] Preparing the machine learning dataset...")
        dataset = create_training_dataset(batches)
        print(f"Number of observations: {len(dataset)}")
        print("\nDataset preview:")
        print(dataset.head())


        # 5. TRAIN AND EVALUATE MODELS

        print("\n[5/7] Training and comparing regression models...")
        evaluation = evaluate_models(dataset)
        print("\nModel comparison:")
        print(evaluation["metrics"].to_string(index=False))
        print(f"\nSelected model: {evaluation['model_name']}")


        # 6. OPTIMIZE CULTURE CONDITIONS

        print("\n[6/7] Optimizing culture conditions...")
        optimum = optimize_space_time_profit(evaluation["model"])
        print("\nOptimal conditions and predicted results:")

        for parameter, value in optimum.items():

            if isinstance(value, (float, int)):
                print(f"{parameter}: {value:.4f}")

            else:
                print(f"{parameter}: {value}")


        # 7. CALCULATE PREDICTION INTERVALS

        print("\n[7/7] Calculating 95% prediction intervals...")

        intervals = calculate_optimal_prediction_intervals(
            evaluation["model"],
            evaluation["prediction_error"],
            optimum
        )

        print("\nPrediction intervals:")
        for metric, values in intervals.items():

            print(f"{metric}:, "f"[{values[0]:.4f}, {values[1]:.4f}]")


        # 8. EXPORT RESULTS


        print("\nExporting results...")
        save_results(dataset,evaluation,optimum,intervals)
        plot_model_comparison(evaluation["metrics"])
        print("\n" + "=" * 60)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as error:
        print("\nERROR DURING PROJECT EXECUTION")
        print(f"Details: {error}")
        raise


if __name__ == "__main__":
    main()

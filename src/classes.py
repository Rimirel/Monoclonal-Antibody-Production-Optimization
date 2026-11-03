
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (BATCH_DATA_COLUMNS,DOWNTIME_DAYS,GLUCOSE_COST_PER_MM,GLUTAMINE_COST_PER_MM,VCD_COST_COEFFICIENT,MAB_VALUE_PER_MG_L,HOURS_PER_DAY)


@dataclass
class Batch:
    """
    Represents one cell culture batch.

    Stores the initial culture conditions and experimental
    time-series data, and calculates its economic performance.
    """

    batch_id: int
    glucose: float
    glutamine: float
    initial_vcd: float

    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        """Validate the initial batch conditions."""

        if self.batch_id <= 0:
            raise ValueError("Batch ID must be a positive integer.")

        if self.glucose < 0:
            raise ValueError("Glucose concentration cannot be negative.")

        if self.glutamine < 0:
            raise ValueError("Glutamine concentration cannot be negative.")

        if self.initial_vcd <= 0:
            raise ValueError("Initial VCD must be positive.")


    # EXPERIMENTAL DATA

    def load_data(self, file_path):
        """
        Load the experimental time-series data for this batch.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Batch file not found: {file_path}"
            )

        self.data = pd.read_csv(file_path)

        missing_columns = [
            column for column in BATCH_DATA_COLUMNS
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError( f"Missing columns in batch {self.batch_id}: "f"{missing_columns}")

        self.data = self.data.sort_values(by="Time [h]").reset_index(drop=True)

        return self.data

    def get_mab_concentration(self, harvest_time_days):
        """
        Estimate the mAb concentration at a specified
        harvest time using linear interpolation.

        Harvest time is provided in days, while experimental
        time is recorded in hours.
        """

        if self.data.empty:
            raise ValueError(
                "Experimental data must be loaded first."
            )

        harvest_time_hours = harvest_time_days * HOURS_PER_DAY

        time = self.data["Time [h]"].to_numpy(dtype=float)
        mab = self.data["mAb [mg/L]"].to_numpy(dtype=float)

        if harvest_time_hours < time.min() or harvest_time_hours > time.max():
            raise ValueError(
                "Harvest time is outside the experimental time range."
            )

        return float(
            np.interp(harvest_time_hours, time, mab)
        )

    # ECONOMIC CALCULATIONS

    def calculate_culture_cost(self):
        """
        Calculate the initial cell culture cost in $/L.
        """

        glucose_cost = GLUCOSE_COST_PER_MM * self.glucose
        glutamine_cost = GLUTAMINE_COST_PER_MM * self.glutamine
        vcd_cost = VCD_COST_COEFFICIENT * self.initial_vcd

        return glucose_cost + glutamine_cost + vcd_cost

    def calculate_culture_value(self, mab_concentration):
        """
        Calculate the monetary value of the produced mAb in $/L.
        """

        return MAB_VALUE_PER_MG_L * mab_concentration

    def calculate_profit(self, mab_concentration):
        """
        Calculate batch profit in $/L.
        """

        culture_value = self.calculate_culture_value(mab_concentration)
        culture_cost = self.calculate_culture_cost()
        return culture_value - culture_cost

    def calculate_space_time_profit(self,harvest_time_days,mab_concentration):
        """
        Calculate space-time profit in $/(L*d).
        """

        total_cycle_time = harvest_time_days + DOWNTIME_DAYS

        if total_cycle_time <= 0:
            raise ValueError("Total cycle time must be positive.")

        profit = self.calculate_profit(mab_concentration)

        return profit / total_cycle_time

    def calculate_space_time_yield(self,harvest_time_days,mab_concentration):
        """
        Calculate mAb space-time yield in mg/(L*d).
        """

        total_cycle_time = harvest_time_days + DOWNTIME_DAYS

        if total_cycle_time <= 0:
            raise ValueError("Total cycle time must be positive.")

        return mab_concentration / total_cycle_time

    # --------------------------------------------------------
    # BATCH INFORMATION
    # --------------------------------------------------------

    def display_info(self):
        """Display the initial conditions of the batch."""

        print(f"Batch ID: {self.batch_id}")
        print(f"Initial glucose: {self.glucose} mM")
        print(f"Initial glutamine: {self.glutamine} mM")
        print(f"Initial VCD: {self.initial_vcd} cells/mL")

    def get_summary(self):
        """Return the initial conditions as a dictionary."""

        return {"Batch ID": self.batch_id,"Glucose [mM]": self.glucose,"Glutamine [mM]": self.glutamine,"VCD [cells/mL]": self.initial_vcd
        }


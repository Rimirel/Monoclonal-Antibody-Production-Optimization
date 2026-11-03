
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Root directory of the project
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_DIR / "datasets"
INITIAL_CONDITIONS_FILE = DATA_DIR / "initial_conditions.csv"

# Output directories
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"


# ============================================================
# DATA FILE CONFIGURATION
# ============================================================

# Batch file naming convention
BATCH_FILE_PATTERN = "batch_{:04d}.csv"

# Column containing the batch identification number
BATCH_ID_COLUMN = "Batch ID"

# Columns in the initial conditions dataset
INITIAL_CONDITIONS_COLUMNS = [
    "Batch ID",
    "Glucose [mM]",
    "Glutamine [mM]",
    "VCD [cells/mL]"
]

# Columns in each batch time-series dataset
BATCH_DATA_COLUMNS = [
    "Time [h]",
    "Glucose [mM]",
    "Glutamine [mM]",
    "Lactate [mM]",
    "Ammonia [mM]",
    "VCD [cells/mL]",
    "TCD [cells/mL]",
    "mAb [mg/L]"
]


# ============================================================
# CULTURE PARAMETERS
# ============================================================

# Bioreactor working volume
BIOREACTOR_VOLUME_L = 1.0

# Downtime between consecutive batches
DOWNTIME_DAYS = 1.0

# Time conversion
HOURS_PER_DAY = 24.0



# ECONOMIC PARAMETERS


# Cell culture costs
GLUCOSE_COST_PER_MM = 10.0
GLUTAMINE_COST_PER_MM = 20.0
VCD_COST_COEFFICIENT = 0.0005

# Monoclonal antibody value
MAB_VALUE_PER_MG_L = 1000.0



# OPTIMIZATION PARAMETERS


# Confidence level for prediction intervals
CONFIDENCE_LEVEL = 0.95

# Bounds for the optimization variables
GLUCOSE_BOUNDS = (5.0, 35.0)
GLUTAMINE_BOUNDS = (1.0, 7.0)
VCD_BOUNDS = (200000.0, 1400000.0)
HARVEST_TIME_BOUNDS_DAYS = (0.0, 28.0)



# MACHINE LEARNING CONFIGURATION


# Reproducibility
RANDOM_STATE = 42

# Training and testing split
TEST_SIZE = 0.20

# Cross-validation
CV_FOLDS = 5


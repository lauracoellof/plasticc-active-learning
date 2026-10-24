"""Project paths, batch selection and fixed experiment settings."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


DIR_RAW = PROJECT_ROOT / "data" / "raw"
DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
for _d in (DIR_RAW, DIR_PROCESSED, MODEL_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BatchConfig:
    """Validated file paths for raw batches 1–11 or the derived combined pool (99)."""

    batch: int

    def __post_init__(self):
        batch = int(self.batch)
        if batch not in {*range(1, 12), 99}:
            raise ValueError("batch must be between 1 and 11, or 99 for the combined pool")
        object.__setattr__(self, "batch", batch)

    @property
    def suffix(self):
        return "" if self.batch == 1 else f"_b{self.batch}"

    @property
    def test_lc(self):
        return DIR_RAW / f"plasticc_test_lightcurves_{self.batch:02d}.csv.gz"

    @property
    def test_metadata_csv(self):
        return DIR_RAW / "plasticc_test_metadata.csv.gz"

    @property
    def test_clean_csv(self):
        return DIR_PROCESSED / f"test_batch{self.batch}_clean.csv"

    @property
    def test_metadata_clean_csv(self):
        return DIR_PROCESSED / f"test_batch{self.batch}_metadata_clean.csv"

    @property
    def features_test_csv(self):
        return DIR_PROCESSED / f"features_estadisticas_test{self.suffix}.csv"

    @property
    def features_gp_test_csv(self):
        return DIR_PROCESSED / f"features_gp_test{self.suffix}.csv"

    @property
    def features_gp2d_test_csv(self):
        return DIR_PROCESSED / f"features_gp2d_test{self.suffix}.csv"

    @property
    def features_extra_test_csv(self):
        return DIR_PROCESSED / f"features_extra_test{self.suffix}.csv"

    @property
    def pred_test_csv(self):
        return DIR_PROCESSED / f"predicciones_test_batch{self.batch}.csv"

    @property
    def test_results_csv(self):
        return DIR_PROCESSED / f"resultados_test_batch{self.batch}.csv"


TRAIN_CSV = DIR_RAW / "training_set.csv"
METADATA_CSV = DIR_RAW / "training_set_metadata.csv"
TRAIN_CLEAN_CSV = DIR_PROCESSED / "training_set_clean.csv"
METADATA_CLEAN_CSV = DIR_PROCESSED / "training_set_metadata_clean.csv"
FEATURES_CSV = DIR_PROCESSED / "features_estadisticas.csv"
FEATURES_GP_CSV = DIR_PROCESSED / "features_gp.csv"
FEATURES_GP2D_CSV = DIR_PROCESSED / "features_gp2d.csv"
FEATURES_EXTRA_CSV = DIR_PROCESSED / "features_extra.csv"
RESULTS_CSV = DIR_PROCESSED / "resultados_clasificacion_150.csv"
E2GP_MODEL = MODEL_DIR / "e2gp_150_calibrado.joblib"
E2GP_METADATA = MODEL_DIR / "e2gp_150_calibrado.meta.json"
CHECKSUMS_MODELOS = MODEL_DIR / "checksums.sha256"


DEFAULT_TEST_BATCH = 4
TEST_BATCH = DEFAULT_TEST_BATCH


TEST_MAX_OBJ = 32926

_BATCH_POR_DEFECTO = BatchConfig(DEFAULT_TEST_BATCH)
TEST_LC = _BATCH_POR_DEFECTO.test_lc
TEST_LC_BATCH1 = TEST_LC
TEST_METADATA_CSV = _BATCH_POR_DEFECTO.test_metadata_csv
TEST_CLEAN_CSV = _BATCH_POR_DEFECTO.test_clean_csv
TEST_METADATA_CLEAN_CSV = _BATCH_POR_DEFECTO.test_metadata_clean_csv
FEATURES_TEST_CSV = _BATCH_POR_DEFECTO.features_test_csv
FEATURES_GP_TEST_CSV = _BATCH_POR_DEFECTO.features_gp_test_csv
FEATURES_GP2D_TEST_CSV = _BATCH_POR_DEFECTO.features_gp2d_test_csv
FEATURES_EXTRA_TEST_CSV = _BATCH_POR_DEFECTO.features_extra_test_csv
PRED_TEST_CSV = _BATCH_POR_DEFECTO.pred_test_csv
TEST_RESULTS_CSV = _BATCH_POR_DEFECTO.test_results_csv


RANDOM_STATE = 42
SNR_MIN = 3
MIN_OBS = 5
PASSBAND_MAP = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}

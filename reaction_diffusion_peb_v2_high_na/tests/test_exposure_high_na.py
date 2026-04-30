import numpy as np

from reaction_diffusion_peb_v2_high_na.src.exposure_high_na import dill_acid_generation, normalize_dose


def test_normalize_dose_basic():
    assert normalize_dose(40.0, 40.0) == 1.0
    assert normalize_dose(60.0, 40.0) == 1.5


def test_dill_acid_zero_intensity_zero_acid():
    I = np.zeros((4, 4))
    H0 = dill_acid_generation(I, dose_norm=1.0, eta=1.0, Hmax=0.2)
    assert np.allclose(H0, 0.0)


def test_dill_acid_unit_intensity_below_Hmax():
    I = np.ones((4, 4))
    H0 = dill_acid_generation(I, dose_norm=1.0, eta=1.0, Hmax=0.2)
    assert np.all(H0 <= 0.2 + 1e-12)
    assert np.all(H0 > 0.0)
    # 0.2 * (1 - exp(-1)) ≈ 0.1264
    assert np.allclose(H0, 0.2 * (1 - np.exp(-1.0)))

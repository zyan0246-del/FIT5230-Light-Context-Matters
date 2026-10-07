import numpy as np
import pytest

from s2f.chromatic import roi_channel_means, spectral_features


def test_roi_channel_order_and_crop():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[2:8, 2:8, 0] = 3
    frame[2:8, 2:8, 1] = 5
    frame[2:8, 2:8, 2] = 7
    result = roi_channel_means(frame, (0.2, 0.2, 0.8, 0.8))
    assert result == {"red": 7.0, "green": 5.0, "blue": 3.0}


def test_spectral_peak_recovers_expected_frequency():
    fps = 25.0
    time = np.arange(500) / fps
    signal = 0.8 * np.sin(2 * np.pi * 1.2 * time) + 0.05 * np.sin(2 * np.pi * 2.4 * time)
    result = spectral_features(signal, fps, target_hz=1.2)
    assert result["peak_hz"] == pytest.approx(1.2, abs=result["bin_width_hz"])
    assert result["peak_to_median_ratio"] > 100
    assert result["target_band_energy_fraction"] > 0.9


@pytest.mark.parametrize("roi", [(-0.1, 0, 1, 1), (0, 0, 0, 1), (0, 0, 2, 1)])
def test_invalid_roi_rejected(roi):
    with pytest.raises(ValueError):
        roi_channel_means(np.zeros((2, 2, 3)), roi)

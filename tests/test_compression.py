import numpy as np

from convae.compression import jpeg_roundtrip, quantize


def test_quantize_error_bounded():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(1000, 16)).astype(np.float32)
    lo, hi = z.min(0), z.max(0)
    zq = quantize(z, lo, hi, bits=8)
    step = (hi - lo) / 255
    assert np.all(np.abs(zq - z) <= step / 2 + 1e-6)  # max error = half a quantization step


def test_jpeg_higher_quality_is_bigger_and_closer():
    rng = np.random.default_rng(0)
    img = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    lo, n_lo = jpeg_roundtrip(img, 10)
    hi, n_hi = jpeg_roundtrip(img, 90)
    err = lambda x: np.abs(x.astype(float) - img).mean()
    assert n_hi > n_lo and err(hi) < err(lo)

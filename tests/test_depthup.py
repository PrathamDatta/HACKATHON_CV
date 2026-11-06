"""Fast unit tests on synthetic scenes (no dataset or trained weights required)."""
import io

import numpy as np
import pytest
from PIL import Image

from depthup import classical as C, degrade, io as dio, metrics
from depthup.model import DLOnlyNet, HybridNet


def scene(h=96, w=128):
    """Two fronto-parallel planes (a step) plus a slanted floor; colour boundary aligned with the step."""
    depth = np.full((h, w), 3.0, np.float32)
    depth[:, w // 2:] = 1.5
    depth[h * 3 // 4:, :] += np.linspace(0, 1, w, dtype=np.float32)[None] * 0.0 + np.linspace(0, 0.5, h - h * 3 // 4)[:, None]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, : w // 2] = (200, 60, 60)
    rgb[:, w // 2:] = (60, 60, 200)
    return rgb, depth


def test_degrade_shapes_and_mask():
    _, d = scene()
    lr, valid = degrade.degrade(d, degrade.Degradation(scale=4, holes=0.3), np.random.default_rng(0))
    assert lr.shape == (24, 32) and valid.shape == lr.shape
    assert 0.15 < (~valid).mean() < 0.45 and (lr[~valid] == 0).all()


def test_degrade_deterministic():
    _, d = scene()
    a = degrade.degrade(d, degrade.Degradation(4, noise=0.02), np.random.default_rng(3))[0]
    b = degrade.degrade(d, degrade.Degradation(4, noise=0.02), np.random.default_rng(3))[0]
    assert np.array_equal(a, b)


def test_indivisible_scale_rejected():
    with pytest.raises(ValueError):
        degrade.downsample(np.zeros((10, 10), np.float32), 4)


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_interpolators_preserve_constant(scale):
    lr = np.full((12, 16), 2.0, np.float32)
    for f in (C.nearest_up, C.bilinear_up, C.bicubic_up):
        out = f(lr, scale)
        assert out.shape == (12 * scale, 16 * scale) and np.allclose(out, 2.0, atol=1e-4)


def test_fill_holes_completes():
    lr = np.random.default_rng(0).uniform(1, 3, (20, 20)).astype(np.float32)
    valid = np.ones_like(lr, bool); valid[5:12, 5:12] = False
    out = C.normalized_fill(np.where(valid, lr, 0), valid)
    assert np.isfinite(out).all() and (out[5:12, 5:12] > 0.5).all()


def test_spike_removed_but_step_kept():
    lr = np.full((20, 20), 2.0, np.float32); lr[:, 10:] = 3.5
    lr[5, 5] += 1.0
    out, spikes = C.remove_spikes(lr, 0.0)
    assert spikes[5, 5] and abs(out[5, 5] - 2.0) < 1e-4
    assert np.allclose(out[:, :9], out[0, 0]) and np.allclose(out[:, 11:], 3.5)   # the real step untouched


def test_detector_flags_step_not_slope():
    ramp = np.tile(np.linspace(1, 4, 40, dtype=np.float32), (30, 1))
    assert C.lr_discontinuity(ramp, 4, 0.0)[0][:, 3:-3].max() == 0
    step = np.full((30, 40), 1.0, np.float32); step[:, 20:] = 3.0
    assert C.lr_discontinuity(step, 4, 0.0)[0].max() > 0.9


def test_gate_pipeline_sharpens_step():
    rgb, d = scene()
    lr, valid = degrade.degrade(d, degrade.Degradation(scale=8), np.random.default_rng(0))
    out = C.gate_pipeline(lr, valid, rgb, 8)
    assert out.depth.shape == d.shape and np.isfinite(out.depth).all()
    bic = C.bicubic_up(lr, 8)
    err = lambda p: float(np.abs(p - d)[:, 56:72].mean())   # noqa: E731  (band around the step)
    assert err(out.depth) < err(bic)


def test_guided_filter_shape():
    rgb, d = scene()
    out = C.guided_up(C.bicubic_up(d[::4, ::4].copy(), 4)[::4, ::4], rgb.astype(np.float32) / 255, 4)
    assert out.shape == d.shape and np.isfinite(out).all()


def test_metrics_perfect_and_worse():
    rgb, d = scene()
    m0 = metrics.evaluate(d, d, rgb)
    assert m0["rmse_cm"] == 0 and m0["ssim"] > 0.999 and m0["edge_f1"] > 0.99
    m1 = metrics.evaluate(d + 0.1, d, rgb)
    assert abs(m1["rmse_cm"] - 10) < 1e-3 and m1["psnr_db"] < m0["psnr_db"]


def test_models_forward_shapes():
    x = __import__("torch").zeros(1, HybridNet.C_IN, 32, 48)
    z = x[:, :1]
    out, d_dl, w = HybridNet()(x, z, z)
    assert out.shape == (1, 1, 32, 48) and w.shape == (1, 3, 32, 48)
    assert abs(float(w.sum(1).mean()) - 1) < 1e-5
    assert DLOnlyNet()(__import__("torch").zeros(1, 5, 32, 48), z).shape == (1, 1, 32, 48)


def _png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def test_io_rejects_garbage():
    with pytest.raises(dio.InputError):
        dio.read_rgb(b"not an image")
    with pytest.raises(dio.InputError):
        dio.read_depth(b"junk", "x.png")
    with pytest.raises(dio.InputError):
        dio.read_rgb(_png(np.zeros((8, 8, 3), np.uint8)))      # too small


def test_io_depth_units():
    d16 = (np.full((10, 10), 2500)).astype(np.uint16)
    d, note = dio.read_depth(_png(d16), "d.png")
    assert np.allclose(d, 2.5) and "millimetres" in note
    d8, note = dio.read_depth(_png(np.arange(100, dtype=np.uint8).reshape(10, 10)), "d.png")
    assert 0.5 <= d8.min() and d8.max() <= 5.0 and "relative" in note
    buf = io.BytesIO(); np.save(buf, np.full((6, 6), 1.2, np.float32))
    assert np.allclose(dio.read_depth(buf.getvalue(), "d.npy")[0], 1.2)


def test_match_depth_to_rgb():
    rgb = np.zeros((96, 128, 3), np.uint8)
    r, lr, s = dio.match_depth_to_rgb(rgb, np.ones((24, 32), np.float32))
    assert s == 4 and r.shape == (96, 128, 3)
    with pytest.raises(dio.InputError):
        dio.match_depth_to_rgb(rgb, np.ones((24, 50), np.float32))


def test_fit_to_scale_crops():
    rgb, d = np.zeros((101, 135, 3), np.uint8), np.ones((101, 135), np.float32)
    r, dd = dio.fit_to_scale(rgb, 8, d)
    assert r.shape[0] % 8 == 0 and r.shape[1] % 8 == 0 and dd.shape == r.shape[:2]

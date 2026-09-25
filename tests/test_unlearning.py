import numpy as np

from eeg_unlearning.unlearning import fuzzy_weights


def test_fuzzy_weights_penalise_samples_close_to_forget_set():
    rng = np.random.default_rng(0)
    forget = rng.normal(0, 0.1, size=(30, 4))
    near = rng.normal(0, 0.1, size=(10, 4))
    far = rng.normal(10, 0.1, size=(10, 4))
    w = fuzzy_weights(np.vstack([near, far]), forget, k=5)
    assert np.all((w >= 0) & (w <= 1))
    assert w[:10].max() < w[10:].min()


def test_fuzzy_weights_handle_tiny_forget_set():
    rng = np.random.default_rng(1)
    w = fuzzy_weights(rng.random((20, 3)), rng.random((2, 3)), k=20)
    assert w.shape == (20,)

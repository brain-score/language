"""
Tests for the OASM (Orthogonal Autocorrelated Sequences Model) implementation.

Tests cover:
- Feature construction correctness
- Between-block orthogonality
- Stateful offset mechanism
- NeuroidAssembly structure (brain-score interface compliance)
- Input handling edge cases
- Equivalence to the paper's full-matrix construction
- Model registration
- Paper's key finding: shuffled CV inflates scores, contiguous CV eliminates them
"""

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.models.oasm.model import OASMSubject


def _make_model(sigma: float = 1.0, max_features: int = 20) -> OASMSubject:
    """Helper to create an OASMSubject with neural recording started."""
    model = OASMSubject(identifier='test', sigma=sigma, max_features=max_features)
    model.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI,
    )
    return model


class TestFeatureConstruction:
    """Test the core identity + Gaussian smoothing feature construction."""

    def test_sigma_zero_gives_identity(self):
        """With sigma=0, each stimulus should get a one-hot vector at its offset position."""
        model = _make_model(sigma=0.0, max_features=10)
        result = model.digest_text(['a', 'b', 'c'])['neural']

        for i in range(3):
            assert result.values[i, i] == 1.0
            non_peak = np.delete(result.values[i], i)
            np.testing.assert_array_equal(non_peak, 0.0)

    def test_smoothing_matches_scipy_exactly(self):
        """Feature values must exactly match scipy.ndimage.gaussian_filter1d on identity."""
        sigma = 2.0
        block_size = 5
        model = _make_model(sigma=sigma, max_features=10)
        result = model.digest_text([f'w{i}' for i in range(block_size)])['neural']

        expected = gaussian_filter1d(np.eye(block_size), sigma=sigma, axis=1)
        np.testing.assert_allclose(result.values[:, :block_size], expected, atol=1e-10)


class TestBetweenBlockOrthogonality:
    """Test that features from different digest_text calls occupy non-overlapping dimensions."""

    def test_two_blocks_orthogonal(self):
        """Dot product between any stimulus from block 1 and any from block 2 should be 0."""
        model = _make_model(sigma=1.0, max_features=20)
        r1 = model.digest_text(['a', 'b', 'c'])['neural']
        r2 = model.digest_text(['d', 'e'])['neural']

        for i in range(3):
            for j in range(2):
                dot = np.dot(r1.values[i], r2.values[j])
                assert dot == pytest.approx(0.0, abs=1e-10), (
                    f"Block1[{i}] . Block2[{j}] = {dot}, expected 0"
                )


class TestStatefulOffset:
    """Test the cumulative offset mechanism that places blocks in unique dimensions."""

    def test_offset_advances_by_block_size(self):
        """With sigma=0, the one-hot peak position reveals the offset."""
        model = _make_model(sigma=0.0, max_features=20)

        r1 = model.digest_text(['a', 'b', 'c'])['neural']
        assert r1.values[0, 0] == 1.0
        assert r1.values[1, 1] == 1.0
        assert r1.values[2, 2] == 1.0

        r2 = model.digest_text(['d', 'e'])['neural']
        assert r2.values[0, 3] == 1.0
        assert r2.values[1, 4] == 1.0

    def test_start_neural_recording_resets_offset(self):
        """A new recording session should reset the offset to 0."""
        model = OASMSubject(identifier='test', sigma=0.0, max_features=20)
        model.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )

        model.digest_text(['a', 'b', 'c'])
        assert model._offset == 3

        model.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )
        assert model._offset == 0

        result = model.digest_text(['x', 'y'])['neural']
        assert result.values[0, 0] == 1.0

    def test_exceeds_max_features_raises(self):
        """Should raise ValueError when cumulative stimuli exceed max_features."""
        model = _make_model(sigma=0.0, max_features=5)
        model.digest_text(['a', 'b', 'c'])

        with pytest.raises(ValueError, match="max_features"):
            model.digest_text(['d', 'e', 'f'])


class TestNeuroidAssemblyStructure:
    """Test brain-score interface compliance: coordinates, dimensions, types."""

    def test_required_neuroid_coordinates(self):
        model = _make_model(sigma=1.0, max_features=10)
        result = model.digest_text(['hello', 'world'])['neural']

        for coord in ['layer', 'neuron_number_in_layer', 'neuroid_id',
                       'recording_target', 'recording_type']:
            result[coord]  # should not raise

    def test_required_presentation_coordinates(self):
        model = _make_model(sigma=1.0, max_features=10)
        result = model.digest_text(['hello', 'world'])['neural']

        np.testing.assert_array_equal(result['stimulus'].values, ['hello', 'world'])
        np.testing.assert_array_equal(result['part_number'].values, [0, 1])

    def test_output_shape(self):
        max_features = 15
        model = _make_model(sigma=0.5, max_features=max_features)
        result = model.digest_text(['a', 'b', 'c'])['neural']

        assert result.dims == ('presentation', 'neuroid')
        assert result.shape == (3, max_features)

    def test_consistent_neuroid_dims_across_blocks(self):
        """All blocks must have the same neuroid dimension for xr.concat (as benchmarks do)."""
        model = _make_model(sigma=1.0, max_features=20)

        r1 = model.digest_text(['a', 'b', 'c'])['neural']
        r2 = model.digest_text(['d', 'e'])['neural']
        r3 = model.digest_text(['f'])['neural']

        combined = xr.concat([r1, r2, r3], dim='presentation')
        assert combined.shape == (6, 20)


class TestInputHandling:
    """Test edge cases in input normalization."""

    def test_numpy_array_input(self):
        """Benchmarks pass numpy arrays, not Python lists."""
        model = _make_model(sigma=0.5, max_features=10)
        text = np.array(['hello world', 'foo bar'])
        result = model.digest_text(text)['neural']
        assert result.shape == (2, 10)

    def test_single_string_input(self):
        model = _make_model(sigma=0.5, max_features=10)
        result = model.digest_text('hello world')['neural']
        assert result.shape == (1, 10)

    def test_no_neural_recording_raises(self):
        model = OASMSubject(identifier='test', sigma=0.5, max_features=10)
        with pytest.raises(AssertionError, match="start_neural_recording"):
            model.digest_text(['hello'])

    def test_behavioral_task_raises(self):
        model = OASMSubject(identifier='test', sigma=0.5, max_features=10)
        with pytest.raises(NotImplementedError):
            model.start_behavioral_task(ArtificialSubject.Task.next_word)


class TestEquivalenceToPaper:
    """Verify mathematical equivalence to the paper's full-matrix OASM construction."""

    def test_incremental_matches_full_matrix(self):
        """
        The stateful per-block approach must produce the same result as building
        the full N x N matrix and smoothing within blocks.

        Validates equivalence to the paper's construction:
            OASM_acts = np.eye(N)
            for block in blocks:
                OASM_acts[block, block] = gaussian_filter1d(submatrix, sigma, axis=1)
        """
        sigma = 1.5
        blocks = [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h', 'i']]
        total_n = sum(len(b) for b in blocks)

        full_matrix = np.eye(total_n)
        offset = 0
        for block in blocks:
            k = len(block)
            if k > 1:
                submatrix = full_matrix[offset:offset + k, offset:offset + k]
                full_matrix[offset:offset + k, offset:offset + k] = gaussian_filter1d(
                    submatrix, sigma=sigma, axis=1
                )
            offset += k

        model = _make_model(sigma=sigma, max_features=total_n)
        incremental_rows = []
        for block in blocks:
            result = model.digest_text(block)['neural']
            incremental_rows.append(result.values)

        incremental_matrix = np.vstack(incremental_rows)

        np.testing.assert_allclose(
            incremental_matrix, full_matrix, atol=1e-10,
            err_msg="Incremental OASM must match paper's full-matrix construction"
        )

    def test_no_linguistic_content(self):
        """
        OASM features should be identical regardless of actual text content,
        given the same block structure.
        """
        sigma = 1.0
        max_features = 10

        model1 = _make_model(sigma=sigma, max_features=max_features)
        r1 = model1.digest_text(['The cat sat on the mat', 'Dogs are great', 'Hello world'])['neural']

        model2 = _make_model(sigma=sigma, max_features=max_features)
        r2 = model2.digest_text(['xyz abc 123', 'qqq', 'asdfghjkl'])['neural']

        np.testing.assert_array_equal(
            r1.values, r2.values,
            err_msg="OASM encodes zero linguistic content: features must depend only on block structure"
        )


class TestRegistration:
    """Test model registry integration."""

    def test_boundary_sigmas_load(self):
        """All registrations are explicit literals, so load_model's static scan finds them."""
        for key in ['oasm-sigma0', 'oasm-sigma0.1', 'oasm-sigma4.8']:
            model = load_model(key)
            ident = model.identifier if isinstance(model.identifier, str) else model.identifier()
            assert ident == key

    def test_49_variants_registered(self):
        """48 sigma values (0.1-4.8) + sigma=0 = 49 total."""
        from brainscore_language import model_registry
        oasm_keys = [k for k in model_registry if k.startswith('oasm-sigma')]
        assert len(oasm_keys) == 49


class TestPaperReproduction:
    """
    Reproduce the paper's key finding on synthetic data: OASM scores high under
    shuffled CV (temporal autocorrelation leakage) but drops to ~zero under
    contiguous CV (entire blocks held out).

    Uses synthetic brain data with temporal autocorrelation -- no real data loading needed.
    Reference: Hadidi et al. (2025), Figure 1.
    """

    def test_shuffled_high_contiguous_zero(self):
        n_blocks = 6
        block_size = 4
        n_stimuli = n_blocks * block_size
        n_voxels = 50
        sigma = 1.0

        # Build OASM features: per-block identity + Gaussian smoothing
        model = _make_model(sigma=sigma, max_features=n_stimuli)
        all_features = []
        for b in range(n_blocks):
            texts = [f'block{b}_stim{i}' for i in range(block_size)]
            result = model.digest_text(texts)['neural']
            all_features.append(result.values)
        X = np.vstack(all_features)

        # Synthetic brain data: random voxel responses with temporal autocorrelation
        # (adjacent stimuli within a block have correlated responses)
        rng = np.random.default_rng(42)
        Y = np.zeros((n_stimuli, n_voxels))
        for b in range(n_blocks):
            raw = rng.standard_normal((block_size, n_voxels))
            Y[b * block_size:(b + 1) * block_size] = gaussian_filter1d(raw, sigma=1.5, axis=0)

        # Shuffled CV: random train/test split (allows within-block leakage)
        ss = ShuffleSplit(n_splits=10, train_size=0.9, random_state=1)
        shuffled_rs = []
        for train_idx, test_idx in ss.split(X):
            reg = RidgeCV(alphas=np.logspace(-3, 3, 7))
            reg.fit(X[train_idx], Y[train_idx])
            Y_pred = reg.predict(X[test_idx])
            rs = [pearsonr(Y[test_idx, j], Y_pred[:, j])[0] for j in range(n_voxels)]
            shuffled_rs.append(np.nanmedian(rs))
        shuffled_score = np.mean(shuffled_rs)

        # Contiguous CV: hold out entire blocks (no within-block leakage)
        contiguous_rs = []
        for held_out in range(n_blocks):
            test_idx = list(range(held_out * block_size, (held_out + 1) * block_size))
            train_idx = [i for i in range(n_stimuli) if i not in test_idx]
            reg = RidgeCV(alphas=np.logspace(-3, 3, 7))
            reg.fit(X[train_idx], Y[train_idx])
            Y_pred = reg.predict(X[test_idx])
            rs = []
            for j in range(n_voxels):
                if np.std(Y[test_idx, j]) < 1e-10 or np.std(Y_pred[:, j]) < 1e-10:
                    rs.append(0.0)
                else:
                    rs.append(pearsonr(Y[test_idx, j], Y_pred[:, j])[0])
            contiguous_rs.append(np.nanmedian(rs))
        contiguous_score = np.mean(contiguous_rs)

        assert shuffled_score > 0.1, (
            f"Shuffled CV should exploit temporal autocorrelation leakage: got r={shuffled_score:.4f}"
        )
        assert abs(contiguous_score) < 0.02, (
            f"Contiguous CV should eliminate leakage: got r={contiguous_score:.4f}"
        )

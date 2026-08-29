import numpy as np
import pytest
 
from src.evaluation.metrics import rmse, mape, mase, compute_all_metrics
 
ACTUAL    = np.array([100.0, 120.0, 110.0])
PREDICTED = np.array([105.0, 115.0, 115.0])
NAIVE     = np.array([ 90.0, 130.0, 100.0])
 
EXPECTED_RMSE = 5.0
 
EXPECTED_MAPE = float(np.mean([5/100, 5/120, 5/110]) * 100)
 
EXPECTED_MASE = 0.5
 
 
class TestRMSE:
 
    def test_perfect_prediction_gives_zero(self):
        actual = np.array([50.0, 100.0, 75.0])
        assert rmse(actual, actual.copy()) == pytest.approx(0.0, abs=1e-10)
 
    def test_known_answer(self):
        assert rmse(ACTUAL, PREDICTED) == pytest.approx(EXPECTED_RMSE, rel=1e-5)
 
    def test_always_nonnegative(self):
        actual    = np.random.default_rng(0).normal(100, 10, 50)
        predicted = np.random.default_rng(1).normal(100, 15, 50)
        assert rmse(actual, predicted) >= 0.0
 
    def test_single_element_array(self):
        assert rmse(np.array([10.0]), np.array([13.0])) == pytest.approx(3.0)
 
    def test_returns_python_float(self):
        result = rmse(ACTUAL, PREDICTED)
        assert isinstance(result, float)
 
    def test_symmetric(self):
        assert rmse(ACTUAL, PREDICTED) == pytest.approx(
            rmse(PREDICTED, ACTUAL), rel=1e-10
        )
 
 
class TestMAPE:
 
    def test_perfect_prediction_gives_zero(self):
        actual = np.array([50.0, 100.0, 75.0])
        assert mape(actual, actual.copy()) == pytest.approx(0.0, abs=1e-10)
 
    def test_known_answer(self):
        assert mape(ACTUAL, PREDICTED) == pytest.approx(EXPECTED_MAPE, rel=1e-4)
 
    def test_result_is_percentage_scale(self):
        actual    = np.array([100.0, 100.0, 100.0])
        predicted = np.array([110.0, 110.0, 110.0])   # 10% over each
        assert mape(actual, predicted) == pytest.approx(10.0, rel=1e-5)
 
    def test_always_nonnegative(self):
        actual    = np.random.default_rng(0).uniform(10, 100, 30)
        predicted = np.random.default_rng(1).uniform(10, 100, 30)
        assert mape(actual, predicted) >= 0.0
 
    def test_zero_actual_does_not_raise(self):
        actual    = np.array([0.0, 100.0, 50.0])
        predicted = np.array([5.0, 110.0, 45.0])
        result = mape(actual, predicted)
        assert not np.isnan(result)
        assert not np.isinf(result)
 
    def test_input_array_not_modified(self):
        actual    = np.array([0.0, 50.0, 100.0])
        original  = actual.copy()
        predicted = np.array([5.0, 55.0, 95.0])
        mape(actual, predicted)
        np.testing.assert_array_equal(actual, original)
 
    def test_returns_python_float(self):
        assert isinstance(mape(ACTUAL, PREDICTED), float)
 
 
class TestMASE:
 
    def test_perfect_prediction_gives_zero(self):
        actual = np.array([100.0, 120.0, 110.0])
        naive  = np.array([ 90.0, 130.0, 100.0])
        assert mase(actual, actual.copy(), naive) == pytest.approx(0.0, abs=1e-10)
 
    def test_model_equal_to_naive_gives_one(self):
        actual    = np.array([100.0, 120.0, 110.0])
        assert mase(actual, NAIVE.copy(), NAIVE.copy()) == pytest.approx(1.0, rel=1e-5)
 
    def test_model_better_than_naive_gives_less_than_one(self):
        result = mase(ACTUAL, PREDICTED, NAIVE)
        assert result < 1.0
 
    def test_model_worse_than_naive_gives_greater_than_one(self):
        result = mase(ACTUAL, NAIVE, PREDICTED)
        assert result > 1.0
 
    def test_known_answer(self):
        assert mase(ACTUAL, PREDICTED, NAIVE) == pytest.approx(EXPECTED_MASE, rel=1e-5)
 
    def test_zero_naive_error_does_not_raise(self):
        perfect_naive = actual.copy()   # naive perfectly predicts actual
        result = mase(actual, predicted, perfect_naive)
        assert not np.isnan(result)
        assert not np.isinf(result)
 
    def test_always_nonnegative(self):
        actual    = np.random.default_rng(0).uniform(50, 150, 30)
        predicted = np.random.default_rng(1).uniform(50, 150, 30)
        naive     = np.random.default_rng(2).uniform(50, 150, 30)
        assert mase(actual, predicted, naive) >= 0.0
 
    def test_returns_python_float(self):
        assert isinstance(mase(ACTUAL, PREDICTED, NAIVE), float)
 
 
 
class TestComputeAllMetrics:
 
    def test_returns_correct_keys(self):
        result = compute_all_metrics(ACTUAL, PREDICTED, NAIVE)
        assert set(result.keys()) == {"rmse", "mape", "mase"}
 
    def test_values_match_individual_functions(self):
        result = compute_all_metrics(ACTUAL, PREDICTED, NAIVE)
        assert result["rmse"] == pytest.approx(rmse(ACTUAL, PREDICTED), rel=1e-10)
        assert result["mape"] == pytest.approx(mape(ACTUAL, PREDICTED), rel=1e-10)
        assert result["mase"] == pytest.approx(
            mase(ACTUAL, PREDICTED, NAIVE), rel=1e-10
        )
 
    def test_all_values_are_floats(self):
        result = compute_all_metrics(ACTUAL, PREDICTED, NAIVE)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
 
    def test_perfect_prediction_all_zeros(self):
        result = compute_all_metrics(ACTUAL, ACTUAL.copy(), NAIVE)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["mape"] == pytest.approx(0.0, abs=1e-10)
        assert result["mase"] == pytest.approx(0.0, abs=1e-10)
 
 
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


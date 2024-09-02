import numpy as np
import pytest
from src.model.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as LR

from src.utility.type import ModelType
from src.utility.parameter import Parameter
from tests.util.round_parameters import round_parameters

class TestLinearRegression:

    @classmethod
    def setup_class(cls):
        """Setup state that applies to the entire class, including fitting the model."""
        cls.model = LinearRegression(alpha=1e-2, iter=20000)
        cls.test_model = LR()
        cls.X = np.array([[1], [2], [3], [4], [5]])
        cls.y = np.array([2, 4, 6, 8, 10])

        # Fit the models once
        cls.model.fit(cls.X, cls.y)
        cls.test_model.fit(cls.X, cls.y)
        cls.p = round_parameters(cls.model.parameters, 8)

    def test_fit(self):
        """Test the fit method of LinearRegression."""
        assert np.allclose(self.p[Parameter.COEFFICIENTS], self.test_model.coef_, rtol=1e-5)
        assert np.allclose(self.p[Parameter.INTERCEPTS], self.test_model.intercept_, rtol=1e-5)

    def test_predict(self):
        """Test the predict method of LinearRegression."""
        predictions = self.model.predict(self.X)
        expected_predictions = self.test_model.predict(self.X)

        assert np.allclose(predictions, expected_predictions, rtol=1e-5)

    def test_get_technical_name(self):
        """Test the get_technical_name method of LinearRegression."""
        assert self.model.get_technical_name() == ModelType.LINEAR_REGRESSION

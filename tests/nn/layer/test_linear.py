import numpy as np
from src.nn.layer.linear import Linear  # Adjust import path based on your project structure
from src.utility.parameter import Parameter
from src.utility.type import InitType


class TestLinearLayer:

    @classmethod
    def setup_class(cls):
        """Setup state that applies to the entire class, including initializing the layer."""
        cls.input_dim = 3
        cls.units = 2
        cls.layer = Linear(input_dim=cls.input_dim, units=cls.units, init=InitType.HE)

        # Manually set weights and biases for reproducibility
        cls.layer.p[Parameter.COEFFICIENTS] = np.array([[0.2, -0.4],
                                                        [0.5, 0.3],
                                                        [-0.6, 0.1]])
        cls.layer.p[Parameter.INTERCEPTS] = np.array([0.1, -0.2])

        # Create some test data
        cls.X = np.array([[1.0, -1.0, 2.0],
                          [-1.0, 2.0, -0.5]])
        cls.grad_output = np.array([[0.3, -0.5],
                                    [-0.7, 0.2]])

    def test_forward(self):
        """Test the forward method of Linear with fixed weights and biases."""
        output = self.layer.forward(self.X)

        # Manually computed expected output using the fixed weights and biases
        expected_output = np.array([[0.2 * 1.0 + 0.5 * (-1.0) - 0.6 * 2.0 + 0.1,
                                     -0.4 * 1.0 + 0.3 * (-1.0) + 0.1 * 2.0 - 0.2],
                                    [0.2 * (-1.0) + 0.5 * 2.0 - 0.6 * (-0.5) + 0.1,
                                     -0.4 * (-1.0) + 0.3 * 2.0 + 0.1 * (-0.5) - 0.2]])

        assert np.allclose(output, expected_output, rtol=1e-5), \
            "Forward pass output does not match expected output."

    def test_backward_params(self):
        """Test the backward_params method of Linear with fixed inputs and grad_output."""
        self.layer.backward_params(self.grad_output)
        gradients = self.layer._params

        # Manually computed expected gradients
        expected_grad_coeff = np.dot(self.X.T, self.grad_output)
        expected_grad_intercept = np.sum(self.grad_output, axis=0)

        assert np.allclose(gradients[Parameter.COEFFICIENTS], expected_grad_coeff, rtol=1e-5), \
            "Gradient w.r.t. coefficients does not match expected value."
        assert np.allclose(gradients[Parameter.INTERCEPTS], expected_grad_intercept, rtol=1e-5), \
            "Gradient w.r.t. intercepts does not match expected value."

    def test_backward_input(self):
        """Test the backward_input method of Linear with fixed weights and grad_output."""
        grad_input = self.layer.backward_input(self.grad_output)

        # Manually computed expected gradient with respect to inputs
        expected_grad_input = np.dot(self.grad_output, self.layer.p[Parameter.COEFFICIENTS].T)

        assert np.allclose(grad_input, expected_grad_input, rtol=1e-5), \
            "Gradient w.r.t. input does not match expected value."

    def test_xavier_initialization(self):
        """Test that the parameters are initialized correctly."""
        layer_xavier = Linear(input_dim=self.input_dim, units=self.units, init=InitType.XAVIER)
        assert layer_xavier.p[Parameter.COEFFICIENTS].shape == (self.input_dim, self.units), \
            "Weights shape does not match expected shape after Xavier initialization."
        assert layer_xavier.p[Parameter.INTERCEPTS].shape == (self.units,), \
            "Bias shape does not match expected shape after Xavier initialization."

    def test_he_initialization(self):
        """Test that the parameters are initialized correctly."""
        layer_xavier = Linear(input_dim=self.input_dim, units=self.units, init=InitType.HE)
        assert layer_xavier.p[Parameter.COEFFICIENTS].shape == (self.input_dim, self.units), \
            "Weights shape does not match expected shape after Xavier initialization."
        assert layer_xavier.p[Parameter.INTERCEPTS].shape == (self.units,), \
            "Bias shape does not match expected shape after Xavier initialization."


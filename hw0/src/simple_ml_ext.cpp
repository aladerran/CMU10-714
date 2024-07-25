#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t num_batches = (m + batch - 1) / batch;
    float *X_batch = new float[batch * n];
    unsigned char *y_batch = new unsigned char[batch];
    float *Z = new float[batch * k];
    float *softmax = new float[batch * k];
    float *grad = new float[n * k];

    for (size_t i = 0; i < num_batches; i++) {
        size_t batch_size = (i == num_batches - 1) ? (m - i * batch) : batch;
        
        // Load the current batch
        for (size_t j = 0; j < batch_size; j++) {
            std::copy(X + (i * batch + j) * n, X + (i * batch + j + 1) * n, X_batch + j * n);
            y_batch[j] = y[i * batch + j];
        }
        
        // Compute logits Z = X_batch @ theta
        for (size_t j = 0; j < batch_size; j++) {
            for (size_t l = 0; l < k; l++) {
                Z[j * k + l] = 0;
                for (size_t m = 0; m < n; m++) {
                    Z[j * k + l] += X_batch[j * n + m] * theta[m * k + l];
                }
            }
        }
        
        // Compute softmax probabilities
        for (size_t j = 0; j < batch_size; j++) {
            float sum_exp = 0;
            for (size_t l = 0; l < k; l++) {
                softmax[j * k + l] = std::exp(Z[j * k + l]);
                sum_exp += softmax[j * k + l];
            }
            for (size_t l = 0; l < k; l++) {
                softmax[j * k + l] /= sum_exp;
            }
        }
        
        // Compute the gradient
        std::fill(grad, grad + n * k, 0);
        for (size_t j = 0; j < batch_size; j++) {
            for (size_t l = 0; l < k; l++) {
                float error = softmax[j * k + l] - (l == y_batch[j] ? 1.0f : 0.0f);
                for (size_t m = 0; m < n; m++) {
                    grad[m * k + l] += X_batch[j * n + m] * error;
                }
            }
        }
        for (size_t j = 0; j < n * k; j++) {
            grad[j] /= batch_size;
        }
        
        // Update theta
        for (size_t j = 0; j < n * k; j++) {
            theta[j] -= lr * grad[j];
        }
    }
    delete[] X_batch;
    delete[] y_batch;
    delete[] Z;
    delete[] softmax;
    delete[] grad;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

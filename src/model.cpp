#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using namespace autodiff;
using namespace Eigen;

// this is user defined data structure used to pass data to the model below
struct CustomData {
    MatrixXd x;
    MatrixXd y;
    const Eigen::Index n;  // number of samples
    const Eigen::Index p;  // number of features
    const Eigen::Index m;  // number of outputs
    CustomData(MatrixXd x, MatrixXd y) : x(x), y(y), n(x.rows()), p(x.cols()), m(y.cols()) {
        if (n != y.rows()) {
            throw std::invalid_argument("x and y must have the same number of rows");
        }
    }
};

/// \brief if we want to use cross validation, we have to define how to split the data
/// \param old_data The data to split, its type is pybind11::object which wraps a pointer to the data
/// \param target_sample_index The index of the sample to be used as the target
/// \return A pybind11::object which wraps a pointer to the new target data
pybind11::object split_sample(const pybind11::object& old_data, const VectorXi& target_sample_index) {
    CustomData* old_data_ptr = old_data.cast<CustomData*>();  // unwrap the pointer
    CustomData* new_data = new CustomData(old_data_ptr->x(target_sample_index, Eigen::all),
                                          old_data_ptr->y(target_sample_index, Eigen::all));  // split the data
    return pybind11::cast(new_data);                                                          // wrap the pointer
}

/// \brief because we allocate memory in split_sample, we have to define how to free the memory
void deleter(pybind11::object const& data) { delete data.cast<CustomData*>(); }

/// @brief The loss function of linear regression model with or without intercept whose expression is ||y - x * para -
/// intercept||^2. It is a template function so that we can use it to compute the loss and its gradient, hessian.
/// @tparam T double for loss, autodiff::dual for gradient and autodiff::dual2nd for hessian
/// @param para vector of parameters, each represents a coefficient of a feature
/// @param intercept vector with one element for intercept or zero element for no intercept model
/// @param ex_data a pybind11::object type data that wraps a pointer to the actual user defined data
/// @return loss value of the model, i.e., ||y - x * para - intercept||^2 or ||y - x * para||^2 (for non-intercept
/// model)
template <class T>
T linear_model(Matrix<T, -1, 1> const& para, Matrix<T, -1, 1> const& intercept, pybind11::object const& ex_data) {
    CustomData* data = ex_data.cast<CustomData*>();  // unwrap the pointer
    return ((data->x * para - data->y).array() + (intercept.size() > 0 ? intercept[0] : 0.0))
        .square()
        .sum();  // compute the loss
}

/// @brief  The loss function of logistic regression model with or without intercept whose expression is log(1 +
/// exp(xbeta)) - y * xbeta, where xbeta = X * para + intercept. It is a template function so that we can use it to
/// compute the loss and its gradient, hessian.
/// @tparam T double for loss, autodiff::dual for gradient and autodiff::dual2nd for hessian
/// @param para vector of parameters, each represents a coefficient of a feature
/// @param intercept vector with one element for intercept or zero element for no intercept model
/// @param ex_data a pybind11::object type data that wraps a pointer to the actual user defined data
/// @return loss value of the model, i.e., log(1 + exp(xbeta)) - y * xbeta, where xbeta = X * para + intercept or X *
/// para (for non-intercept model)
template <class T>
T logistic_model(Matrix<T, -1, 1> const& para, Matrix<T, -1, 1> const& intercept, pybind11::object const& ex_data) {
    CustomData* data = ex_data.cast<CustomData*>();  // unwrap the pointer
    Eigen::Array<T, -1, 1> xbeta =
        (data->x * para).array() + (intercept.size() > 0 ? intercept[0] : 0.0);  // xbeta = x * para + intercept
    return ((xbeta.exp() + 1.0).log() - (data->y).array() * xbeta)
        .sum();  // return the negative log likelihood log(1 + exp(xbeta)) - y * xbeta
}

/// @brief The loss function of multi linear regression model with or without intercept whose expression is ||y - x *
/// Matrix(para) - intercept||^2, where Matrix(para) is the matrix version of parameters. It is a template function so
/// that we can use it to compute the loss and its gradient, hessian.
/// @tparam T double for loss, autodiff::dual for gradient and autodiff::dual2nd for hessian
/// @param para vector of parameters which is the vectorized version of the matrix of parameters. Denote the number of
/// features as p and the number of outputs as m, then the matrix of parameters B has p rows and m cols. And each row of
/// B represents coefficients of a feature, para consists of all rows of B which are positioned end to end.
/// @param intercept vector with m element for intercept or zero element for no intercept model
/// @param ex_data a pybind11::object type data that wraps a pointer to the actual user defined data
/// @return loss value of the model, i.e., ||y - x * Matrix(para) - intercept||^2 or ||y - x * Matrix(para)||^2 (for
/// non-intercept model) , where Matrix(para) is the matrix version of parameters
template <class T>
T multi_linear_model(Matrix<T, -1, 1> const& para, Matrix<T, -1, 1> const& intercept, pybind11::object const& ex_data) {
    CustomData* data = ex_data.cast<CustomData*>();    // unwrap the pointer
    T value = T(0.0);                                  // initialize the value to be returned
    bool has_intercept = intercept.size() == data->m;  // check this is non-intercept model or not
    Eigen::Map<Matrix<T, -1, 1> const, 0, InnerStride<>> beta(
        NULL, data->p, InnerStride<>(data->m));   // beta is a view of any column of matrix version of para
    for (Eigen::Index i = 0; i < data->m; i++) {  // loop over outputs
        new (&beta) Eigen::Map<Matrix<T, -1, 1> const, 0, InnerStride<>>(
            para.data() + i, data->p,
            InnerStride<>(data->m));  // update the view, let it point to the i-th column of para
        value += ((data->x * beta - data->y.col(i)).array() + (has_intercept ? intercept[i] : 0.0))
                     .square()
                     .sum();  // compute the squared error of the i-th output
    }
    return value;
}

// In the following we will export the above data structure and functions to python
PYBIND11_MODULE(scope_model, m) {
    // export the data structure CustomData and its constructor
    pybind11::class_<CustomData>(m, "CustomData").def(pybind11::init<MatrixXd, MatrixXd>());
    // export the function split_sample and corresponding deleter
    m.def("split_sample", &split_sample);
    m.def("deleter", &deleter);
    // export the loss as overload functions because we have to specify the template parameter
    // linear model
    m.def("linear_model",
          pybind11::overload_cast<Matrix<double, -1, 1> const&, Matrix<double, -1, 1> const&, pybind11::object const&>(
              &linear_model<double>));
    m.def("linear_model",
          pybind11::overload_cast<Matrix<dual, -1, 1> const&, Matrix<dual, -1, 1> const&, pybind11::object const&>(
              &linear_model<dual>));
    m.def(
        "linear_model",
        pybind11::overload_cast<Matrix<dual2nd, -1, 1> const&, Matrix<dual2nd, -1, 1> const&, pybind11::object const&>(
            &linear_model<dual2nd>));
    // logistic model
    m.def("logistic_model",
          pybind11::overload_cast<Matrix<double, -1, 1> const&, Matrix<double, -1, 1> const&, pybind11::object const&>(
              &logistic_model<double>));
    m.def("logistic_model",
          pybind11::overload_cast<Matrix<dual, -1, 1> const&, Matrix<dual, -1, 1> const&, pybind11::object const&>(
              &logistic_model<dual>));
    m.def(
        "logistic_model",
        pybind11::overload_cast<Matrix<dual2nd, -1, 1> const&, Matrix<dual2nd, -1, 1> const&, pybind11::object const&>(
            &logistic_model<dual2nd>));
    // multi linear model
    m.def("multi_linear_model",
          pybind11::overload_cast<Matrix<double, -1, 1> const&, Matrix<double, -1, 1> const&, pybind11::object const&>(
              &multi_linear_model<double>));
    m.def("multi_linear_model",
          pybind11::overload_cast<Matrix<dual, -1, 1> const&, Matrix<dual, -1, 1> const&, pybind11::object const&>(
              &multi_linear_model<dual>));
    m.def(
        "multi_linear_model",
        pybind11::overload_cast<Matrix<dual2nd, -1, 1> const&, Matrix<dual2nd, -1, 1> const&, pybind11::object const&>(
            &multi_linear_model<dual2nd>));
}

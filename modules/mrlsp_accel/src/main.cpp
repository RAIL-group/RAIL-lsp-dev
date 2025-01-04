#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <array>
#include <map>
#include "core.hpp"


namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(mrlsp_accel, m) {
    m.doc() = R"pbdoc(
        Pybind11 plugin for demonstrating C++ features
        -----------------------

        .. currentmodule:: pycpp_examples

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    py::class_<SubgoalData, std::shared_ptr<SubgoalData>>(m, "SubgoalData")
        .def(py::init<double, double, double, long, bool>(),
             py::arg("prob_feasible"),
             py::arg("delta_success_cost"),
             py::arg("exploration_cost"),
             py::arg("hash_id"),
             py::arg("is_from_last_chosen") = 0)
        .def("__hash__", &SubgoalData::get_hash);

    m.def("get_mr_ordering_cost", &get_mr_ordering_cost);
    m.def("get_mr_lowest_cost_ordering", &get_mr_lowest_cost_ordering);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

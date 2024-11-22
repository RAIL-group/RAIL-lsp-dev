#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <array>
#include <map>
#ifndef _POTLP
#define _POTLP
#include "potlp.hpp"
#endif
#include "tests.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(potlp_accel, m) {
    m.doc() = R"pbdoc(
        Pybind11 plugin for demonstrating C++ features
        -----------------------

        .. currentmodule:: pycpp_examples

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    py::class_<Action, std::shared_ptr<Action>>(m, "Action_cpp")
        .def(py::init<
                std::tuple<int, int>,
                std::tuple<int, int>,
                double,
                std::vector<std::tuple<int, int>>,
                int,
                std::vector<std::vector<int>>,
                bool,
                int64_t >(),
            py::arg("start_state"),
            py::arg("known_state"),
            py::arg("known_space_cost"),
            py::arg("node_name_path"),
            py::arg("unk_dfa_state"),
            py::arg("unk_dfa_transitions"),
            py::arg("is_terminal"),
            py::arg("hash_id"))
        .def_readonly("start_state", &Action::start_state)
        .def_readonly("known_state", &Action::known_state)
        .def_readonly("known_space_cost", &Action::known_space_cost)
        .def_readonly("node_name_path", &Action::node_name_path)
        .def_readonly("unk_dfa_state", &Action::unk_dfa_state)
        .def_readonly("unk_dfa_transitions", &Action::unk_dfa_transitions)
        .def_readonly("is_terminal", &Action::is_terminal)
        .def_readonly("hash_id", &Action::hash_id);

    m.def("compute_subgoal_props_for_action_accel", &compute_subgoal_props_for_action);
    m.def("find_best_action", &find_best_action);
    // m.def("test_potlp_state_change_accel", &test_potlp_state_change);
    // (TODO: Abhish) Combine two functions below to same function
    m.def("test_update_subgoal_prop_dict_accel", &test_update_subgoal_prop_dict);
    m.def("test_empty_history_subgoal_prop_dict_update_accel", &test_empty_history_subgoal_prop_dict_update);
    m.def("test_updated_ps_with_updated_properties_accel", &test_updated_ps_with_updated_properties);
    m.def("test_add_to_history_accel", &test_add_to_history);
    m.def("test_get_ps_rs_re_with_history_accel", &test_get_ps_rs_re_with_history);
    m.def("get_PS_per_transition_accel", &get_PS_per_transition);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

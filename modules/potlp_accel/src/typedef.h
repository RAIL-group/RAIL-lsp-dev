
#include <map>
#include <vector>
#include <tuple>

namespace structure {
    typedef std::map<int, std::vector<std::vector<double>>> SubgoalPropDict;
    typedef std::vector<std::tuple<int, int, int>> History;
    typedef std::vector<std::vector<int>> UnkDfaTransitions;
    // typedef std::map<std::tuple<int, int>, std::vector<ActionPtr>> ActionDict;
}
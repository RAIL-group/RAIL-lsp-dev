#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>


struct SubgoalData{
  double prob_feasible;
  double delta_success_cost;
  double exploration_cost;
  long hash_id;
  bool is_from_last_chosen;

  SubgoalData(double prob_feasible, double delta_success_cost,
                 double exploration_cost, long hash_id,
                 bool is_from_last_chosen)
      : prob_feasible(prob_feasible), delta_success_cost(delta_success_cost),
        exploration_cost(exploration_cost), hash_id(hash_id),
        is_from_last_chosen(is_from_last_chosen) {}

  long get_hash() const { return hash_id; }
};
typedef std::shared_ptr<SubgoalData> SubgoalDataPtr;


struct MRFstate {
  std::vector<long> subgoal_list;
  int num_robots;
  std::map<long, double> goal_distances;
  std::map<std::pair<long, long>, double> inter_distances;
  std::vector<double> progress;
  std::vector<double> cost_to_target;
  std::vector<long> prev_robot_locations;
  std::vector<long> target_robot_locations;
  std::set<long> unexplored;
  std::map<long, SubgoalDataPtr> hash_to_subgoal;
  double prob;
  double cost;

  MRFstate(const std::vector<long> robots,
           const std::map<long, double> &goal_distances,
           const std::map<std::pair<long, long>, double> &inter_distances,
           const std::vector<SubgoalDataPtr> &subgoals):
           cost(0.0), prob(1.0), num_robots(robots.size()) {
    this->goal_distances = goal_distances;
    this->inter_distances = inter_distances;
    progress.resize(num_robots, 0.0);
    cost_to_target.resize(num_robots, 0.0);
    // initialize previous locations as robots
    prev_robot_locations = robots;
    target_robot_locations = robots;
    for (auto &subgoal: subgoals) {
      unexplored.insert(subgoal->hash_id);
      hash_to_subgoal[subgoal->hash_id] = subgoal;
    }
  }

  MRFstate(const SubgoalDataPtr &new_subgoal,
           const MRFstate &old_state):
        subgoal_list(old_state.subgoal_list),
        prev_robot_locations(old_state.target_robot_locations),
        target_robot_locations(old_state.target_robot_locations),
        cost_to_target(old_state.cost_to_target),
        progress(old_state.progress),
        cost(old_state.cost),
        prob(old_state.prob) {

    num_robots = old_state.num_robots;
    hash_to_subgoal = old_state.hash_to_subgoal;
    goal_distances = old_state.goal_distances;
    inter_distances = old_state.inter_distances;

    // which robot is done?
    std::vector<double> robot_remaining_cost(num_robots);
    std::transform(old_state.cost_to_target.begin(), old_state.cost_to_target.end(),
                   old_state.progress.begin(), robot_remaining_cost.begin(),
                   std::minus<double>());
    int done_robot_id = std::min_element(robot_remaining_cost.begin(), robot_remaining_cost.end()) - robot_remaining_cost.begin();
    double delta_t = old_state.cost_to_target[done_robot_id] - old_state.progress[done_robot_id];
    long delta_fi = old_state.target_robot_locations[done_robot_id];

    // Arriving at a robot's start or at a subgoal another robot already explored reveals nothing
    bool is_newly_explored = old_state.unexplored.find(delta_fi) != old_state.unexplored.end();
    double ps = is_newly_explored ? hash_to_subgoal[delta_fi]->prob_feasible : 0.0;
    double Rs = is_newly_explored ?
      goal_distances[delta_fi] + hash_to_subgoal[delta_fi]->delta_success_cost : 0.0;
    double Re = is_newly_explored ? hash_to_subgoal[delta_fi]->exploration_cost : 0.0;
    double Q_success = Rs - std::min(Rs, Re);
    cost = old_state.cost + old_state.prob * (delta_t + ps * Q_success);
    prob = old_state.prob * (1 - ps);

    // store and update the unexplored subgoals
    for (auto &f : old_state.unexplored) {
      if (f != old_state.target_robot_locations[done_robot_id]) {
        unexplored.insert(f);
      }
    }

    std::transform(progress.begin(), progress.end(), progress.begin(),
                   [delta_t](double x) { return x + delta_t; });
    progress[done_robot_id] = 0.0;

    prev_robot_locations[done_robot_id] = target_robot_locations[done_robot_id];

    if (new_subgoal != nullptr) {
      subgoal_list.push_back(new_subgoal->hash_id);
      target_robot_locations[done_robot_id] = new_subgoal->hash_id;
      cost_to_target[done_robot_id] = get_cost_to_target(prev_robot_locations[done_robot_id], target_robot_locations[done_robot_id]);
    }
  }

  bool operator<(const MRFstate &other) const {
    return cost < other.cost;
  }

  // Subgoals the next robot to finish can be sent to. While there is a subgoal no robot is
  // heading to for every free robot, it takes one of those (a permutation of the subgoals);
  // otherwise it can join any unexplored subgoal. Empty when it explores the last subgoal.
  std::vector<SubgoalDataPtr> get_subgoal_candidates() const {
    std::vector<double> remaining_time(num_robots);
    std::transform(cost_to_target.begin(), cost_to_target.end(),
                   progress.begin(), remaining_time.begin(),
                   std::minus<double>());
    int done_robot_id = std::min_element(remaining_time.begin(), remaining_time.end()) - remaining_time.begin();
    std::set<long> remaining_unexplored(unexplored);
    remaining_unexplored.erase(target_robot_locations[done_robot_id]);

    std::vector<SubgoalDataPtr> candidates;
    if (remaining_unexplored.empty()) {
      return candidates;
    }
    long num_free_robots = std::count(remaining_time.begin(), remaining_time.end(), remaining_time[done_robot_id]);
    std::set<long> unassigned(remaining_unexplored);
    for (const auto &f : target_robot_locations) {
      unassigned.erase(f);
    }
    const std::set<long> &candidate_hashes =
      (static_cast<long>(unassigned.size()) >= num_free_robots) ? unassigned : remaining_unexplored;
    for (const auto &f : candidate_hashes) {
      candidates.push_back(hash_to_subgoal.at(f));
    }
    return candidates;
  }

  double get_cost_to_target(const long &from_frontier, const long &to_frontier) {
    auto from_frontier_it = hash_to_subgoal.find(from_frontier);
    double frontier_return_cost = (from_frontier_it != hash_to_subgoal.end()) ?
     std::min(goal_distances[from_frontier] + hash_to_subgoal[from_frontier]->delta_success_cost, hash_to_subgoal[from_frontier]->exploration_cost) : 0.0;
    double kc = inter_distances[std::pair<long, long>(from_frontier, to_frontier)];
    auto to_frontier_it = hash_to_subgoal.find(to_frontier);
    double Rs = (to_frontier_it != hash_to_subgoal.end()) ?
      goal_distances[to_frontier] + hash_to_subgoal[to_frontier]->delta_success_cost : 0.0;
    double Re = (to_frontier_it != hash_to_subgoal.end()) ?
      hash_to_subgoal[to_frontier]->exploration_cost : 0.0;
    double knowledge_time = std::min(Rs, Re);
    return frontier_return_cost + kc + knowledge_time;
  }

};

// Recursively search over the subgoal each robot is sent to whenever it is free, until all
// subgoals are explored. The first assignments follow `ordering` from `ordering_idx`.
MRFstate get_lowest_cost_state_sub(
    const MRFstate &state,
    const std::vector<SubgoalDataPtr> &ordering,
    size_t ordering_idx,
    double *bound) {
  if (state.cost > *bound) {
    return state;
  }

  std::vector<SubgoalDataPtr> candidates = state.get_subgoal_candidates();
  if (candidates.empty()) {
    MRFstate final_state(nullptr, state);
    *bound = std::min(*bound, final_state.cost);
    return final_state;
  }

  if (ordering_idx < ordering.size()) {
    candidates = {ordering[ordering_idx]};
  }
  std::vector<MRFstate> best_states;
  for (const auto &s : candidates) {
    best_states.push_back(get_lowest_cost_state_sub(MRFstate(s, state), ordering, ordering_idx + 1, bound));
  }
  return *(std::min_element(best_states.begin(), best_states.end()));
}

MRFstate get_mr_lowest_cost_state(
      const std::vector<long> robots,
      const std::vector<SubgoalDataPtr> &subgoals,
      const std::map<long, double> &goal_distances,
      const std::map<std::pair<long, long>, double> &inter_distances,
      const std::vector<SubgoalDataPtr> &ordering) {
  double bound = 1.0e10;
  MRFstate state(robots, goal_distances, inter_distances, subgoals);
  return get_lowest_cost_state_sub(state, ordering, 0, &bound);
}

// Cost of assigning subgoals in the given order, then completing exploration at lowest cost
double get_mr_ordering_cost(
      const std::vector<long> robots,
      const std::vector<SubgoalDataPtr> &subgoals,
      const std::map<long, double> &goal_distances,
      const std::map<std::pair<long, long>, double> &inter_distances) {
  return get_mr_lowest_cost_state(robots, subgoals, goal_distances, inter_distances, subgoals).cost;
}

std::pair<double, std::vector<long>> get_mr_lowest_cost_ordering(
      const std::vector<long> robots,
      const std::vector<SubgoalDataPtr> &frontiers,
      const std::map<long, double> &goal_distances,
      const std::map<std::pair<long, long>, double> &inter_distances) {
  auto sout = get_mr_lowest_cost_state(robots, frontiers, goal_distances, inter_distances, {});
  return std::make_pair(sout.cost, sout.subgoal_list);
}

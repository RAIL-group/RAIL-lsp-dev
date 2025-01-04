#include <iostream>
#include <map>
#include <memory>
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

    auto delta_fi_it = hash_to_subgoal.find(delta_fi);
    double ps = (delta_fi_it != hash_to_subgoal.end()) ? hash_to_subgoal[delta_fi]->prob_feasible : 0.0;
    double Rs = (delta_fi_it != hash_to_subgoal.end()) ?
      goal_distances[delta_fi] + hash_to_subgoal[delta_fi]->delta_success_cost : 0.0;
    double Re = (delta_fi_it != hash_to_subgoal.end()) ?
      hash_to_subgoal[delta_fi]->exploration_cost : 0.0;
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
      // Don't assign when the frontier list is equal to the overall subgoals
      // Except when the number of subgoal is less than the number of robots assign the new frontier
      if (subgoal_list.size() < hash_to_subgoal.size() ||
            subgoal_list.size() < num_robots) {
        subgoal_list.push_back(new_subgoal->hash_id);
      }
      target_robot_locations[done_robot_id] = new_subgoal->hash_id;
      cost_to_target[done_robot_id] = get_cost_to_target(prev_robot_locations[done_robot_id], target_robot_locations[done_robot_id]);
      //  If any target_robot_locations is not in unexplored, then it is already explored
      //  assign it to nearest unexplored location
      std::vector<std::pair<int, long>> already_explored_subgoals;
      for (int idx = 0; idx < num_robots; idx++) {
        long f = target_robot_locations[idx];
        if ((unexplored.find(f) == unexplored.end()) && (hash_to_subgoal.find(f) != hash_to_subgoal.end())) {
          already_explored_subgoals.push_back(std::make_pair(idx, f));
        }
      }
      for (const auto &exp_idx_f : already_explored_subgoals) {
        int idx = exp_idx_f.first;
        long exp_f =  exp_idx_f.second;
        auto nearest_unexplored = std::min_element(
          unexplored.begin(), unexplored.end(),
          [&exp_f, this](long a, long b) { // Change 4: Optimized data structure access
            return get_cost_to_target(exp_f, a) < get_cost_to_target(exp_f, b);
          }
        );
        target_robot_locations[idx] = *nearest_unexplored;
        cost_to_target[idx] += get_cost_to_target(exp_f, *nearest_unexplored);
      }
    }
  }

  bool operator<(const MRFstate &other) const {
    return cost < other.cost;
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

MRFstate mr_fstate_planner_compute_end_assignment(MRFstate mr_fstate) {
  const int remaining_robot_assignments = std::count_if(mr_fstate.target_robot_locations.begin(), mr_fstate.target_robot_locations.end(),
    [&mr_fstate](long f){ return mr_fstate.hash_to_subgoal.find(f) == mr_fstate.hash_to_subgoal.end(); });
  const int num_steps = mr_fstate.unexplored.size() + remaining_robot_assignments;
  for (int i=0; i < num_steps; i++) {
    std::vector<double> robot_remaining_cost(mr_fstate.num_robots);
    std::transform(mr_fstate.cost_to_target.begin(), mr_fstate.cost_to_target.end(),
                   mr_fstate.progress.begin(), robot_remaining_cost.begin(),
                   std::minus<double>());
    int done_robot_id = std::min_element(robot_remaining_cost.begin(), robot_remaining_cost.end()) - robot_remaining_cost.begin();
    long delta_fi = mr_fstate.target_robot_locations[done_robot_id];

    std::vector<std::pair<long, double>> remaining_subgoals;
    for (const auto &f : mr_fstate.unexplored) {
      if (f != delta_fi) {
        remaining_subgoals.push_back(std::make_pair(f, mr_fstate.get_cost_to_target(delta_fi, f)));
      }
    }
    std::sort(remaining_subgoals.begin(), remaining_subgoals.end(),
      [](const std::pair<long, double> &a, const std::pair<long, double> &b) {
        return a.second < b.second;
      });
    SubgoalDataPtr new_target = remaining_subgoals.size() > 0 ? mr_fstate.hash_to_subgoal[remaining_subgoals[0].first] : nullptr;
    mr_fstate = MRFstate(new_target, mr_fstate);
  }
  return mr_fstate;
}

double get_mr_ordering_cost(
      const std::vector<long> robots,
      const std::vector<SubgoalDataPtr> &subgoals,
      const std::map<long, double> &goal_distances,
      const std::map<std::pair<long, long>, double> &inter_distances) {
    MRFstate mr_fstate = MRFstate(robots, goal_distances, inter_distances, subgoals);
    for (auto &f: subgoals) {
      mr_fstate = MRFstate(f, mr_fstate);
    }
    mr_fstate = mr_fstate_planner_compute_end_assignment(mr_fstate);
    return mr_fstate.cost;
}

MRFstate get_lowest_cost_ordering_sub(
    const std::vector<SubgoalDataPtr> &frontiers,
    const MRFstate &prev_state,
    double *bound) {
  if (frontiers.size() == 1) {
    MRFstate state(frontiers[0], prev_state);
    state = mr_fstate_planner_compute_end_assignment(state);
    *bound = std::min(*bound, state.cost);
    return state;
  }

  if (prev_state.cost > *bound) {
    return prev_state;
  }

  std::vector<MRFstate> best_states;
  for (auto f_it = frontiers.begin(); f_it != frontiers.end(); ++f_it) {
    std::vector<SubgoalDataPtr> sub_frontiers;
    std::copy(frontiers.begin(), f_it, std::back_inserter(sub_frontiers));
    std::copy(f_it + 1, frontiers.end(), std::back_inserter(sub_frontiers));
    best_states.push_back(get_lowest_cost_ordering_sub(sub_frontiers, MRFstate(*f_it, prev_state), bound));
  }
  return *(std::min_element(best_states.begin(), best_states.end()));
}

std::pair<double, std::vector<long>> get_mr_lowest_cost_ordering(
      const std::vector<long> robots,
      const std::vector<SubgoalDataPtr> &frontiers,
      const std::map<long, double> &goal_distances,
      const std::map<std::pair<long, long>, double> &inter_distances) {
  std::vector<MRFstate> best_states;
  double bound = 1.0e10;
  MRFstate state(robots, goal_distances, inter_distances, frontiers);
  auto sout = get_lowest_cost_ordering_sub(frontiers, state, &bound);
  return std::make_pair(sout.cost, sout.subgoal_list);
}

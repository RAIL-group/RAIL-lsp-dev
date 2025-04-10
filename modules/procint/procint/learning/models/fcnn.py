import torch

import lsp_xai
import taskplan
from taskplan.core import Subgoal
from taskplan.learning.models.fcnn import Fcnn


class IntvFcnn(Fcnn):
    name = 'FCNNforIntervention'

    def __init__(self, args=None):
        super(IntvFcnn, self).__init__(args)

    def compute_subgoal_props(_, is_feasibles,
                              subgoal_data,
                              device='cpu'):
        # using the computed subgoal properties from the graph, we create
        # subgoal objects, assign their respective prob_feasible predictions
        # and finally return as a dictonary where key is the ordered index
        # of the subgoals
        subgoal_props = {}
        counter = 0
        # if graph data
        if 'is_subgoal' in subgoal_data:
            for index, is_subgoal in enumerate(subgoal_data['is_subgoal']):
                if is_subgoal == 1:
                    # print('subgoal detected')
                    subgoal = Subgoal(index)
                    subgoal.set_props(
                        prob_feasible=is_feasibles[index].cpu()
                    )
                    subgoal.id = counter
                    subgoal_props[counter] = subgoal
                    counter += 1
        else:
            for index, _ in enumerate(subgoal_data['latent_features']):
                subgoal = Subgoal(index)
                subgoal.set_props(
                    prob_feasible=is_feasibles[index].cpu()
                )
                subgoal.id = counter
                subgoal_props[counter] = subgoal
                counter += 1

        return subgoal_props

    def update_datum(self, datum, device):
        with torch.no_grad():
            # for our case this subgoal_data is input graph to
            # the gnn
            out = self.forward(datum['subgoal_data'], device)
            out_det = out.detach()
            is_feasibles = torch.nn.Sigmoid()(out_det[:, 0])
            subgoal_props = self.compute_subgoal_props(
                is_feasibles, datum['subgoal_data'], device)
            # print(subgoal_props)

            datum = lsp_xai.utils.data.update_datum_policies(
                subgoal_props, datum)

            if datum is None:
                return None

            return datum

    @classmethod
    def compute_expected_cost_for_policy(_, subgoal_props,
                                         subgoal_policy_data):
        return lsp_xai.utils.data.compute_expected_cost_for_policy(
            subgoal_props, subgoal_policy_data)

    @classmethod
    def get_net_eval_fn(_, network_file,
                        device=None, do_return_model=False):
        model = IntvFcnn()
        model.load_state_dict(torch.load(network_file,
                                         map_location=device))
        model.eval()
        model.to(device)

        def frontier_net(datum, subgoals):
            graph = taskplan.utilities.utils.preprocess_fcnn_data(datum)
            prob_feasible_dict = {}
            for idx, subgoal in enumerate(subgoals):
                sub_data = {'latent_features': graph['latent_features'][idx].unsqueeze(0)}
                with torch.no_grad():
                    out = model.forward(sub_data, device)
                    out[:, 0] = torch.sigmoid(out[:, 0])
                    out = out.detach().cpu().numpy()
                    prob_feasible_dict[subgoal] = out[0][0]
            return prob_feasible_dict

        if do_return_model:
            return frontier_net, model
        else:
            return frontier_net

    def update_model_counterfactual(self, datum, limit_num,
                                    margin, learning_rate, device, num_steps=5000):
        import copy

        datum = copy.deepcopy(datum)

        # following section is not needed unless we set a limit on the number
        # of subgoals (which we are not as of now)
        # delta_subgoal_data = self.get_subgoal_prop_impact(
        #     datum, device, delta_cost_limit=-1e10)

        # Initialize some terms for the optimization
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # Now we perfrom iterative gradient descent until the expected cost of the
        # new target subgoal is lower than that of the originally selected subgoal.
        for ii in range(5000):
            # Update datum to reflect new neural network state
            datum = self.update_datum(datum, device)

            # Compute the subgoal properties by passing images through the network.
            # (PyTorch implicitly builds a graph of these operations so that we can
            # differentiate them later.)
            nn_out = self.forward(datum['subgoal_data'], device)
            is_feasibles = torch.nn.Sigmoid()(nn_out[:, 0])
            # delta_success_costs = nn_out[:, 1]
            # exploration_costs = nn_out[:, 2]
            limited_subgoal_props = self.compute_subgoal_props(
                is_feasibles, datum['subgoal_data'], device)

            if ii == 0:
                base_subgoal_props = limited_subgoal_props

            # Compute the expected of the new target subgoal:
            q_target = self.compute_expected_cost_for_policy(
                limited_subgoal_props, datum['target_subgoal_policy'])
            # Cost of the 'backup' (formerly the agent's chosen subgoal):
            q_backup = self.compute_expected_cost_for_policy(
                limited_subgoal_props, datum['backup_subgoal_policy'])
            print(
                f"{ii:5} | Q_dif = {q_target - q_backup:6f} | Q_target = {q_target:6f} | Q_backup = {q_backup:6f}"
            )
            assert q_target > 0
            assert q_backup > 0

            # The zero-crossing of the difference between the two is the decision
            # boundary we are hoping to cross by updating the paramters of the
            # neural network via gradient descent.
            q_diff = q_target - q_backup

            if q_diff <= -margin:
                # When it's less than zero, we're done.
                break

            # Via PyTorch magic, gradient descent is easy:
            optimizer.zero_grad()
            q_diff.backward()
            optimizer.step()
        else:
            # If it never crossed the boundary, we have failed.
            raise ValueError("Decision boundary never crossed.")

        upd_subgoal_props = limited_subgoal_props

        return (datum, base_subgoal_props, upd_subgoal_props)

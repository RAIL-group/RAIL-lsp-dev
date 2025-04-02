import torch
import torch.nn as nn
import torch.nn.functional as F

import taskplan


class Fcnn(nn.Module):
    name = 'FCNNforTaskPlan'

    def __init__(self, args=None):
        super(Fcnn, self).__init__()
        torch.manual_seed(8616)
        self._args = args

        self.fc1 = nn.Linear(772 * 3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 16)
        self.fc9 = nn.Linear(16, 8)
        self.fc10 = nn.Linear(8, 4)

        self.classifier = nn.Linear(4, 1)

        self.fc1bn = nn.BatchNorm1d(2048)
        self.fc2bn = nn.BatchNorm1d(1024)
        self.fc3bn = nn.BatchNorm1d(512)
        self.fc4bn = nn.BatchNorm1d(256)
        self.fc5bn = nn.BatchNorm1d(128)
        self.fc6bn = nn.BatchNorm1d(64)
        self.fc7bn = nn.BatchNorm1d(32)
        self.fc8bn = nn.BatchNorm1d(16)
        self.fc9bn = nn.BatchNorm1d(8)
        self.fc10bn = nn.BatchNorm1d(4)

        # Following class weighting factors have
        # been calculated after observing the data
        self.pos = 1  # 9.74
        self.neg = 1  # 0.53

    def forward(self, data, device):
        h = data['latent_features'].type(torch.float).to(device)
        h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        h = F.leaky_relu(self.fc4bn(self.fc4(h)), 0.1)
        h = F.leaky_relu(self.fc5bn(self.fc5(h)), 0.1)
        h = F.leaky_relu(self.fc6bn(self.fc6(h)), 0.1)
        h = F.leaky_relu(self.fc7bn(self.fc7(h)), 0.1)
        h = F.leaky_relu(self.fc8bn(self.fc8(h)), 0.1)
        h = F.leaky_relu(self.fc9bn(self.fc9(h)), 0.1)
        h = F.leaky_relu(self.fc10bn(self.fc10(h)), 0.1)
        props = self.classifier(h)
        return props

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):
        # Separate outputs.
        is_feasible_logits = nn_out[:, 0]
        # Extract & load the data to device
        is_feasible_label = data['labels'].to(device)

        # Compute the contribution from the is_feasible_label
        is_feasible_xentropy = self.pos * \
            is_feasible_label * -F.logsigmoid(is_feasible_logits) + \
            self.neg * (1 - is_feasible_label) * -F.logsigmoid(-is_feasible_logits)
        is_feasible_xentropy = torch.mean(is_feasible_xentropy) + 1e-10

        # Sum the contributions
        loss = is_feasible_xentropy

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/is_feasible_xentropy",
                              is_feasible_xentropy.item(),
                              index)
            writer.add_scalar("Loss/total_loss",
                              loss.item(),
                              index)

        return loss

    @classmethod
    def get_net_eval_fn(_, network_file,
                        device=None):
        model = Fcnn()
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
        return frontier_net

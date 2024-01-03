import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from tqdm import tqdm

from datasets import MoleculeDataset
from transfer.learning import GInfoMinMax, ViewLearner
from transfer.model import GNN
from unsupervised.utils import initialize_edge_weight
import sys


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def compute_returns(next_value, rewards, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns



def train(args, model, view_learner_actor, view_learner_critic, device, dataset, model_optimizer, optimizerA, optimizerC):
    dataset = dataset.shuffle()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    model.train()
    reward_curve = []
    num_T = 3

    for epoch in tqdm(range(1, args.epochs)):
        logging.info('====epoch {}'.format(epoch))
        model_loss_all = 0
        actor_loss_all = 0
        critic_loss_all = 0
        flag = 0
        for batch in dataloader:
            gate_inputs = torch.ones([batch.edge_index.shape[1]])
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()
            # set up
            batch = batch.to(device)
            batch_aug_edge_weight = batch_aug_edge_weight.to(device)
            rewards = []
            values = []
            log_probs = []
            # 1. Train view to maximize contrastive loss
            if epoch % num_T == 1:
                view_learner_actor.train()
                view_learner_actor.zero_grad()
                view_learner_critic.train()
                view_learner_critic.zero_grad()
                model.eval()

                x = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
                for i in range(num_T):  # batch_aug_edge_weight is important!
                    # State
                    x_aug = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)
                    # Dist & Value
                    edge_logits_dist = view_learner_actor(x_aug, batch.batch, batch.x, batch.edge_index, batch.edge_attr)
                    edge_logits_value = view_learner_critic(x_aug, batch.batch, batch.x, batch.edge_index, batch.edge_attr)
                    values.append(edge_logits_value)
                    # Action (from Dist)
                    temperature = 1.0
                    bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                    eps = (bias - (1 - bias)) * torch.rand(edge_logits_dist.size()) + (1 - bias)
                    gate_inputs = torch.log(eps) - torch.log(1 - eps)
                    gate_inputs = gate_inputs.to(device)
                    gate_inputs = (gate_inputs + edge_logits_dist) / temperature
                    batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()
                    # Log_prob
                    log_prob  = torch.log(torch.clamp(batch_aug_edge_weight, min=1e-6)) # if not clamp, it will have 'NaN' problem
                    log_probs.append(log_prob)
                    # Reward (adversarial idea)
                    reward = model.calc_loss(x, x_aug)
                    rewards.append(reward)
                    if flag == 0:
                        reward_cpu = reward.to('cpu').detach()
                        logging.info('current reward: {}'.format(reward_cpu))
                        reward_curve.append(reward_cpu)
                        flag = 1
                
                # Next state & Next value
                x_aug = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)
                next_value = view_learner_critic(x_aug, batch.batch, batch.x, batch.edge_index, batch.edge_attr)
                # Return
                returns = compute_returns(next_value, rewards)
                returns = torch.cat(returns).detach()
                values = torch.cat(values)
                log_probs = torch.cat(log_probs)

                # Actor loss & Critic loss
                advantage = returns - values
                advantage = advantage.squeeze()
                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                actor_loss_all += actor_loss.item() * batch.num_graphs
                critic_loss_all += critic_loss.item() * batch.num_graphs
                # Gradient ascent formulation
                optimizerA.zero_grad()
                optimizerC.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                optimizerA.step()
                optimizerC.step()

            # 2. Train model to minimize contrastive loss
            model.train()
            view_learner_actor.eval()
            view_learner_critic.eval()
            model.zero_grad()
            x = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)

            edge_logits = view_learner_actor(x, batch.batch, batch.x, batch.edge_index, batch.edge_attr)
            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()
            x_aug = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            # 2.b. standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()

            # break

        logging.info('Epoch {}, Model Loss {:.4f}, Actor Loss {:.4f}, Critic Loss {:.4f}'.format(epoch, model_loss_all, actor_loss_all, critic_loss_all))

        if epoch % 1 == 0:
            torch.save(model.gnn.state_dict(), "./models_ga2c/chem_case/pretrain_ga2c_encoder_epoch_"+ str(epoch)+".pth")
            torch.save(view_learner_actor.gnn.state_dict(), "./models_ga2c/chem_case/pretrain_ga2c_actor_epoch_"+ str(epoch)+".pth")
            torch.save(view_learner_critic.gnn.state_dict(), "./models_ga2c/chem_case/pretrain_ga2c_critic_epoch_"+ str(epoch)+".pth")

    fin_model_loss = model_loss_all / len(dataloader)
    fin_actor_loss = actor_loss_all / len(dataloader)
    fin_critic_loss = critic_loss_all / len(dataloader)

    return fin_model_loss, fin_actor_loss, fin_critic_loss

def run(args):
    Path("./models_ga2c/chem").mkdir(parents=True, exist_ok=True)
    
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)
    log_file = '/home/xjc/444_ga2c/ga2c-main/ga2c_pretrain.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='a')
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    my_transforms = Compose([initialize_edge_weight])
    dataset = MoleculeDataset("original_datasets/transfer/"+args.dataset, dataset=args.dataset,
                              transform=my_transforms)
    model = GInfoMinMax(
        GNN(num_layer=args.num_gc_layers, emb_dim=args.emb_dim, JK="last", drop_ratio=args.drop_ratio, gnn_type="gin"),
        proj_hidden_dim=args.emb_dim).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    # Define actor model/optimizer and critic model/optimizer.
    view_learner_actor = ViewLearner(
        GNN(num_layer=args.num_gc_layers, emb_dim=args.emb_dim, JK="last", drop_ratio=args.drop_ratio, gnn_type="gin"),
        mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    view_learner_critic = ViewLearner(
        GNN(num_layer=args.num_gc_layers, emb_dim=args.emb_dim, JK="last", drop_ratio=args.drop_ratio, gnn_type="gin"),
        mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    optimizerA = torch.optim.Adam(view_learner_actor.parameters(), lr=args.actor_view_lr)
    optimizerC = torch.optim.Adam(view_learner_critic.parameters(), lr=args.critic_view_lr)

    model_loss, actor_loss, critic_loss = train(args, model, view_learner_actor, view_learner_critic, device, dataset, model_optimizer, optimizerA, optimizerC)




def arg_parse():
    parser = argparse.ArgumentParser(description='Transfer Learning AD-GCL Pretrain on ZINC 2M')

    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--actor_view_lr', type=float, default=0.001, help='View Learning rate.')
    parser.add_argument('--critic_view_lr', type=float, default=0.001, help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=3,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='layerwise',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=11,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=0.0, help='View Learner Edge Perturb Regularization Strength')

    parser.add_argument('--seed', type=int, default=2024)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)
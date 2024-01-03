import argparse
import logging
import random
import yaml
from yaml import SafeLoader

import numpy as np
import torch
from sklearn.svm import LinearSVC, SVC
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from datasets import TUDataset, TUEvaluator
from unsupervised.embedding_evaluation import EmbeddingEvaluation
from unsupervised.encoder import TUEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight, initialize_node_features, set_tu_dataset_y_shape
from unsupervised.view_learner import ViewLearner
from feature_expansion import FeatureExpander
from datasets_tu import get_dataset

from time import perf_counter as t
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


def run(args):
    logging.info("Running tu......")
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    evaluator = TUEvaluator()

    if args.dataset in {'MUTAG', 'COLLAB', 'REDDIT-BINARY'}:
        my_transforms = Compose([initialize_node_features, initialize_edge_weight, set_tu_dataset_y_shape])
        dataset = TUDataset("./original_datasets/", args.dataset, transform=my_transforms)

    else:
        dataset = get_dataset(args.dataset, sparse=True, feat_str='deg+odeg100', root='../data')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    n_features = dataset.num_features
    logging.info('n_features: {}'.format(n_features))

    model = GInfoMinMax(
        TUEncoder(num_dataset_features=n_features, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        args.emb_dim).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    # Define actor model/optimizer and critic model/optimizer.
    view_learner_actor = ViewLearner(TUEncoder(num_dataset_features=n_features, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    view_learner_critic = ViewLearner(TUEncoder(num_dataset_features=n_features, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    optimizerA = torch.optim.Adam(view_learner_actor.parameters(), lr=args.actor_view_lr)
    optimizerC = torch.optim.Adam(view_learner_critic.parameters(), lr=args.critic_view_lr)

    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True), evaluator, dataset.task_type, dataset.num_tasks, device, param_search=True)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type, dataset.num_tasks, device, param_search=True)

    model.eval()
    # train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)
    train_score, val_score, test_score = ee.kf_embedding_evaluation_autogcl(model.encoder, dataset)
    logging.info(
        "Before training Embedding Eval Scores: Train: {:.4f} Val: {:.4f} Test: {:.4f}".format(train_score, val_score, test_score))


    model_losses = []
    actor_losses = []
    critic_losses = []
    valid_curve = []
    test_curve = []
    train_curve = []
    reward_curve = []

    start = t()
    num_T = 3
    for epoch in range(1, args.epochs + 1):
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

                x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)
                for i in range(num_T):  # batch_aug_edge_weight is important!
                    # State
                    x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)
                    # Dist & Value
                    edge_logits_dist = view_learner_actor(x_aug, batch.batch, batch.x, batch.edge_index, None)
                    edge_logits_value = view_learner_critic(x_aug, batch.batch, batch.x, batch.edge_index, None)
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
                x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)
                next_value = view_learner_critic(x_aug, batch.batch, batch.x, batch.edge_index, None)
                # Return
                returns = compute_returns(next_value, rewards)
                returns = torch.cat(returns).detach()
                values = torch.cat(values)
                log_probs = torch.cat(log_probs)
                    
                # regularization
                # row, col = batch.edge_index
                # edge_batch = batch.batch[row]
                # edge_drop_out_prob = 1 - batch_aug_edge_weight
                # # logging.info('edge_drop_out_prob: {}'.format(torch.mean(edge_drop_out_prob)))
                # uni, edge_batch_num = edge_batch.unique(return_counts=True)
                # sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")
                # reg = []
                # for b_id in range(args.batch_size):
                #     if b_id in uni:
                #         num_edges = edge_batch_num[uni.tolist().index(b_id)]
                #         reg.append(sum_pe[b_id] / num_edges)
                #     else:
                #         # means no edges in that graph. So don't include.
                #         pass
                # num_graph_with_edges = len(reg)
                # reg = torch.stack(reg)
                # reg = reg.mean()

                # Actor loss & Critic loss
                advantage = returns - values
                advantage = advantage.squeeze()
                # actor_loss = -(log_probs * advantage.detach()).mean() + (args.reg_lambda * reg)
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
            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)

            edge_logits = view_learner_actor(x, batch.batch, batch.x, batch.edge_index, None)
            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()
            print("batch.edge_index: ", batch.edge_index)
            print("batch_aug_edge_weight: ", batch_aug_edge_weight)
            sys.exit()
            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            # 2.b. standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()


        fin_model_loss = model_loss_all / len(dataloader)
        fin_actor_loss = actor_loss_all / len(dataloader)
        fin_critic_loss = critic_loss_all / len(dataloader)
        logging.info('Epoch {}, Model Loss {:.4f}, Actor Loss {:.4f}, Critic Loss {:.4f}'.format(epoch, fin_model_loss, fin_actor_loss, fin_critic_loss))
        model_losses.append(fin_model_loss)
        actor_losses.append(fin_actor_loss)
        critic_losses.append(fin_critic_loss)
        if epoch % args.eval_interval == 0:
            model.eval()
            # train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)
            train_score, val_score, test_score = ee.kf_embedding_evaluation_autogcl(model.encoder, dataset)

            logging.info(
                "Metric: {} Train: {:.4f} Val: {:.4f} Test: {:.4f}".format(evaluator.eval_metric, train_score, val_score, test_score))
            train_curve.append(train_score)
            valid_curve.append(val_score)
            test_curve.append(test_score)

    now = t()
    print("total time: ", now-start)
    memory_stats = torch.cuda.memory_stats(device)
    peak_memory = memory_stats["allocated_bytes.all.peak"]
    print(f"Peak Memory: {peak_memory / 1e6} MB")
    logging.info('total time: {}'.format(now-start))

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    logging.info('reward curve: {}'.format(reward_curve))
    logging.info('FinishedTraining!')
    logging.info('Dataset: {}'.format(args.dataset))
    logging.info('reg_lambda: {}'.format(args.reg_lambda))
    logging.info('drop_ratio: {}'.format(args.drop_ratio))
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info('BestTrainScore: {}'.format(best_train))
    logging.info('BestValidationScore: {}'.format(valid_curve[best_val_epoch]))
    logging.info('FinalTestScore: {}'.format(test_curve[best_val_epoch]))

    return valid_curve[best_val_epoch], test_curve[best_val_epoch]


def arg_parse():
    parser = argparse.ArgumentParser(description='A2C-GCL TU')

    parser.add_argument('--dataset', type=str, default='REDDIT-MULTI-5K', help='Dataset') # NCI1, PROTEINS, MUTAG, DD, COLLAB, REDDIT-BINARY, REDDIT-MULTI-5K, IMDB-BINARY, IMDB-MULTI
    parser.add_argument('--model_lr', type=float, default=0.001, help='Model Learning rate.')
    parser.add_argument('--actor_view_lr', type=float, default=0.001, help='View Learning rate.')
    parser.add_argument('--critic_view_lr', type=float, default=0.001, help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=3, help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='layerwise', help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0, help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=60, help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=0.0, help='View Learner Edge Perturb Regularization Strength')
    parser.add_argument('--eval_interval', type=int, default=5, help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear", help="Downstream classifier is linear or non-linear")
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    data_list = ['IMDB-BINARY', 'MUTAG', 'PROTEINS', 'DD', 'NCI1', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    # data_list = ['MUTAG']   # time

    for dataset in data_list:
        if dataset == 'MUTAG':
            args.batch_size = 32

        if dataset in ['REDDIT-BINARY', 'REDDIT-MULTI-5K']:
            args.num_gc_layers = 5
            args.emb_dim = 128
            args.epochs = 150
            args.reg_lambda = 5.0
        res = []
        args.dataset = dataset
        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
        log_file = '/home/xjc/444_ga2c/ga2c-main/reg.'+args.dataset+'.log'
        logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        args.seed = 2024
        for i in range(5):
            val, test = run(args)
            res.append(test)
            args.seed += 1
        res_array = np.array(res)
        logging.info('Mean Testscore: {:.2f}±{:.2f}'.format( np.mean(res_array)*100, np.std(res_array)*100 ))
        print("dataset: ", dataset)
        print(np.mean(res_array),np.std(res_array))
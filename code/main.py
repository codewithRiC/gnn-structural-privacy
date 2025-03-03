import os
import sys
import traceback
import uuid
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch_geometric.transforms import Compose
from datasets import load_dataset
from models import NodeClassifier
from trainer import Trainer
from transforms import FeatureTransform, FeaturePerturbation, LabelPerturbation, PrivatizeStructure, TwoHopRRBaseline
from utils import print_args, WandbLogger, add_parameters_as_argument, \
    measure_runtime, from_args, str2bool, Enum, EnumAction, colored_text, bootstrap


class LogMode(Enum):
    INDIVIDUAL = 'individual'
    COLLECTIVE = 'collective'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def confidence_interval(data, func=np.mean, size=1000, ci=95, seed=12345):
    bs_replicates = bootstrap(data, func=func, n_boot=size, seed=seed)
    p = 50 - ci / 2, 50 + ci / 2
    bounds = np.nanpercentile(bs_replicates, p)
    return (bounds[1] - bounds[0]) / 2


@measure_runtime
def run(args):
    dataset = from_args(load_dataset, args)

    test_acc = []
    attack_auc = []
    run_metrics = {}
    run_id = str(uuid.uuid1())

    logger = None
    if args.log and args.log_mode == LogMode.COLLECTIVE:
        logger = WandbLogger(project=args.project_name, config=args, enabled=args.log, reinit=False, group=run_id)

    progbar = tqdm(range(args.repeats), file=sys.stdout)
    for version in progbar:

        if args.log and args.log_mode == LogMode.INDIVIDUAL:
            args.version = version
            logger = WandbLogger(project=args.project_name, config=args, enabled=args.log, group=run_id)

        try:
            non_sp_data = dataset.clone().to(args.device)

            # non-structurally private data
            non_sp_data = Compose([
                from_args(FeatureTransform, args),
                from_args(FeaturePerturbation, args),
                from_args(LabelPerturbation, args)
            ])(non_sp_data)
            
            print(f"{non_sp_data:}")

            data = dataset.clone().to(args.device)

            # # preprocess data
            data = Compose([
                from_args(FeatureTransform, args),
                from_args(FeaturePerturbation, args),
                from_args(LabelPerturbation, args),
                from_args(PrivatizeStructure, args)
                # from_args(TwoHopRRBaseline, args)
            ])(data)
            
            print(f"{data:}") 
            # exit()
            
            #
            # 
            # 
            # Ensure the node feature matrices have the same shape
            assert non_sp_data.x.shape == data.x.shape, "Node feature matrices have different shapes!"

            # Compare the feature matrices element-wise
            num_nodes, num_features = non_sp_data.x.shape
            

            for node in range(num_nodes):
                similar_count = 0
                for feature in range(num_features):
                    non_sp_value = non_sp_data.x[node, feature]
                    sp_value = data.x[node, feature]
                    # if non_sp_value != sp_value:
                    #     print(f"Node {node}, Feature {feature}: non_sp_data = {non_sp_value}, data = {sp_value}")
                    if non_sp_value == sp_value:
                        similar_count += 1

                print(f"Node {node}: Total number of similar x values = {similar_count}")
            
            different_edges = (data.adj_t.to_dense() != non_sp_data.adj_t.to_dense()).sum()
            print(f"{different_edges} edges have been changed!")
            
            # different_edges = (data.edge_index != non_sp_data.edge_index).sum().item()
            # print(f"{different_edges} edges have been changed!")
             
            # #--------------
            # #Save the privatized dataset as nodewise
            # private_structure_dir = os.path.join(args.output_dir, f"private_structure_{run_id}")
            # os.makedirs(private_structure_dir, exist_ok=True)

            # for node_id in range(data.num_nodes):
            #     # Extract node-wise private structure (edges related to the node)
            #     private_edges = data.adj_t[node_id].to_dense().nonzero(as_tuple=True)
            #     private_edges_path = os.path.join(private_structure_dir, f"node_{node_id}.pt")
            #     torch.save(private_edges, private_edges_path)

            # print(f"Node-wise private structure saved at: {private_structure_dir}")
            # #-----------------
            
            #------------------
            #Save the private dataset as whole
            # Directory to save the processed dataset
            processed_dir = os.path.join(args.output_dir, f"processed_structure_{run_id}")
            os.makedirs(processed_dir, exist_ok=True)

            # Create a dictionary to hold the dataset
            dataset_dict = {
                "node_features": data.x,          # Node features tensor
                "edge_index": data.edge_index,    # Edge index tensor
                "node_labels": data.y,            # Node labels tensor
                "train_mask": data.train_mask,    # Training mask
                "val_mask": data.val_mask,        # Validation mask
                "test_mask": data.test_mask       # Test mask
            }

            # Path to save the dataset
            dataset_path = os.path.join(processed_dir, "graph_data.pt")

            # Save the entire dataset
            torch.save(dataset_dict, dataset_path)

            print(f"Processed graph structure saved at: {dataset_path}")

            
            #-------------------
            exit()
            
            # define model
            model = from_args(NodeClassifier, args, input_dim=data.num_features, num_classes=data.num_classes)

            # train the model
            trainer = from_args(Trainer, args, logger=logger if args.log_mode == LogMode.INDIVIDUAL else None)
            best_metrics = trainer.fit(model, data)

            # attack the model for link prediction
            if args.attack:
                attack_metrics = trainer.attack(data, non_sp_data)
                # attack_metrics = trainer.baseline_attack(data, non_sp_data)
                attack_auc.append(attack_metrics)
            # print("ATTACK METRICS")
            # print(type(data))
            # print(attack_metrics)

            # process results
            for metric, value in best_metrics.items():
                run_metrics[metric] = run_metrics.get(metric, []) + [value]

            test_acc.append(best_metrics['test/acc'])
            progbar.set_postfix({'last_test_acc': test_acc[-1], 'avg_test_acc': np.mean(test_acc)})

        except Exception as e:
            error = ''.join(traceback.format_exception(Exception, e, e.__traceback__))
            logger.log_summary({'error': error})
            raise e
        finally:
            if args.log and args.log_mode == LogMode.INDIVIDUAL:
                logger.finish()

    if args.log and args.log_mode == LogMode.COLLECTIVE:
        summary = {}
        for metric, values in run_metrics.items():
            summary[metric + '_mean'] = np.mean(values)
            summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=args.seed)

        logger.log_summary(summary)

    if not args.log:
        os.makedirs(args.output_dir, exist_ok=True)
        df_results = pd.DataFrame(test_acc, columns=['test/acc']).rename_axis('version').reset_index()
        if args.attack:
            df_attack = pd.DataFrame(attack_auc)
            df_results = pd.concat([df_results, df_attack], axis=1)
        df_results['Name'] = run_id
        for arg_name, arg_val in vars(args).items():
            df_results[arg_name] = [arg_val] * len(test_acc)
        df_results.to_csv(os.path.join(args.output_dir, f'{run_id}.csv'), index=False)


def main():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    add_parameters_as_argument(load_dataset, group_dataset)

    # data transformation args
    group_perturb = init_parser.add_argument_group(f'data transformation arguments')
    add_parameters_as_argument(FeatureTransform, group_perturb)
    add_parameters_as_argument(FeaturePerturbation, group_perturb)
    add_parameters_as_argument(LabelPerturbation, group_perturb)
    add_parameters_as_argument(PrivatizeStructure, group_perturb)
    add_parameters_as_argument(TwoHopRRBaseline, group_perturb)

    # model args
    group_model = init_parser.add_argument_group(f'model arguments')
    add_parameters_as_argument(NodeClassifier, group_model)

    # trainer arguments (depends on perturbation)
    group_trainer = init_parser.add_argument_group(f'trainer arguments')
    add_parameters_as_argument(Trainer, group_trainer)
    group_trainer.add_argument('--device', help='desired device for training', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], default='cuda:0')

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-s', '--seed', type=int, default=None, help='initial random seed')
    group_expr.add_argument('-r', '--repeats', type=int, default=10, help="number of times the experiment is repeated")
    group_expr.add_argument('-o', '--output-dir', type=str, default='./output', help="directory to store the results")
    group_expr.add_argument('--log', type=str2bool, nargs='?', const=True, default=False, help='enable wandb logging')
    group_expr.add_argument('--log-mode', type=LogMode, action=EnumAction, default=LogMode.INDIVIDUAL,
                            help='wandb logging mode')
    group_expr.add_argument('--project-name', type=str, default='LPGNN', help='wandb project name')

    # attack arguments
    group_attack = init_parser.add_argument_group(f'attack arguments')
    group_attack.add_argument('--attack', type=bool, default=False, help='perform attack')

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    print_args(args)
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.seed:
        seed_everything(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print(colored_text('CUDA is not available, falling back to CPU', color='red'))
        args.device = 'cpu'

    try:
        run(args)
    except KeyboardInterrupt:
        print('Graceful Shutdown...')


if __name__ == '__main__':
    main()

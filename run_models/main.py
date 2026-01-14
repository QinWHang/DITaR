import os
import argparse
import torch

from train_models.generators.generator import Generator
from train_models.generators.bert_generator import BertGenerator
from train_models.generators.generator import GRU4RecGenerator
from train_models.trainers.seq_trainer import SeqTrainer
from train_models.utils.utils import set_seed
from train_models.utils.logger import Logger


parser = argparse.ArgumentParser()


parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., 'ml-1m'.")
parser.add_argument("--train_data_path", type=str, required=True, help="Full path to the training data file.")
parser.add_argument("--output_dir", type=str, required=True, help="The output directory for models and logs.")
parser.add_argument("--model_name", type=str, required=True, help="Model name, e.g., 'sasrec'.")

parser.add_argument("--checkpoint_dir",
                    default='./saved_model/',
                    type=str,
                    help="The directory to save checkpoints.")

parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="Whether run the test on a trained model")

parser.add_argument("--topk",
                    default=10,
                    type=int,
                    help="The number of items to recommend")

parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="The hidden size of embedding")

parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="The number of transformer layers for SASRec and BERT4Rec")

parser.add_argument("--num_heads",
                    default=2,
                    type=int,
                    help="The number of attention heads in transformer layers")

parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="The number of GRU layers for GRU4Rec")

parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="The dropout rate")

parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="The max length of input sequence")

parser.add_argument("--mask_prob",
                    type=float,
                    default=0.3,
                    help="The mask probability for training BERT4Rec model")

parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="The number of negative samples for training")

parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="The number of negative samples for testing")

parser.add_argument("--train_batch_size",
                    default=256,
                    type=int,
                    help="Total batch size for training.")

parser.add_argument("--eval_batch_size",
                     default=256, 
                     type=int, 
                     help="Total batch size for evaluation.")

parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")

parser.add_argument("--l2",
                    default=0.0,
                    type=float,
                    help='The L2 regularization')

parser.add_argument("--num_train_epochs",
                    default=200,
                    type=float,
                    help="Total number of training epochs to perform.")

parser.add_argument("--lr_dc_step",
                    default=10,
                    type=int,
                    help='Every n steps, decrease the learning rate')

parser.add_argument("--lr_dc",
                    default=0.1,
                    type=float,
                    help='Learning rate decrease factor')

parser.add_argument("--patience",
                    type=int,
                    default=10,
                    help='How many epochs to tolerate the performance decrease while training')

parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="Which metric is used to select the best model.")

parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="Random seed for reproducibility")

parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")

parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')

parser.add_argument('--num_workers',
                    default=4,
                    type=int,
                    help='The number of workers in dataloader')

parser.add_argument("--log", 
                    default=True,
                    action="store_true",
                    help="Whether create a new log file")

parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='Whether to run in demo mode')

parser.add_argument("--eval_steps",
                    default=5,
                    type=int,
                    help="Evaluate the model every n steps (epochs)")

parser.add_argument("--eval_ks",
                    default="10,20",
                    type=str,
                    help="The k values for evaluation metrics (comma-separated)")

args = parser.parse_args()
set_seed(args.seed)

def main():
    log_manager = Logger(args)
    logger, writer = log_manager.get_logger()
    args.now_str = log_manager.get_now_str()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.model_name == 'gru4rec':
        generator = GRU4RecGenerator(args, logger, device)
    elif args.model_name == 'bert4rec':
        generator = BertGenerator(args, logger, device)
    else:
        generator = Generator(args, logger, device)

    trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test:
        logger.info("Starting evaluation on test set...")
        results, _ = trainer.test()
        logger.info(f"Test results: {results}")
    else:
        logger.info("Starting training...")
        results, best_epoch = trainer.train()
        logger.info(f"Training completed. Best epoch: {best_epoch}")
        logger.info(f"Final results: {results}")

    log_manager.end_log()

if __name__ == "__main__":
    main()

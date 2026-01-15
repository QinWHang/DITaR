import os
import argparse
import torch

from train_models.generators.generator import Generator, BERT4RecGenerator, GRU4RecGenerator
from train_models.trainers.seq_trainer import SeqTrainer
from train_models.utils.utils import set_seed
from train_models.utils.logger import Logger
from DualviewIdentification.DualviewConstruct.dual_view_models import DualViewSASRec, DualViewGRU4Rec, DualViewBERT4Rec


def get_args():
    parser = argparse.ArgumentParser(description="Train a Dual-View Sequential Recommender with Contrastive Loss")
    
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'ml-1m').")
    parser.add_argument("--train_data_path", type=str, required=True, help="Full path to the training data file (.inter.txt format).")
    parser.add_argument("--output_dir", type=str, required=True, help="The base output directory for models and logs.")
    parser.add_argument("--model_name", type=str, required=True, choices=['sasrec', 'gru4rec', 'bert4rec'], help="Backbone model name.")
    
    parser.add_argument("--embedding_dir", type=str, default='./data_new/embeddings', help='Directory for pre-trained embeddings.')
    parser.add_argument("--max_len", type=int, default=200, help="Maximum sequence length.")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--train_neg", type=int, default=1, help="Number of negative samples for training.")
    parser.add_argument("--test_neg", type=int, default=100, help="Number of negative samples for testing.")

    parser.add_argument("--hidden_size", type=int, default=64, help="Model hidden size.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads (for Transformer-based models).")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers (for GRU or Transformer).")
    parser.add_argument("--trm_num", type=int, default=2, help="Alias for number of transformer layers.")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate.")
    
    parser.add_argument("--pca_dim", type=int, default=64, help="PCA embedding dimension.")
    parser.add_argument("--freeze_semantic_emb", action='store_true', help="Freeze pre-trained semantic embeddings.")
    parser.add_argument("--freeze_pca_emb", action='store_true', help="Freeze pre-trained PCA embeddings.")
    
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic view in recommendation loss.")
    parser.add_argument("--lambda_c", type=float, default=0.1, help="Weight for the contrastive loss.")

    parser.add_argument("--num_train_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization weight.")
    parser.add_argument("--lr_dc_step", type=int, default=50, help="Learning rate decay step.")
    parser.add_argument("--lr_dc", type=float, default=0.1, help="Learning rate decay factor.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--eval_steps", type=int, default=5, help="Evaluate every N epochs.")
    parser.add_argument("--eval_ks", type=str, default='10,20', help="K values for evaluation metrics (e.g., '10,20,50').")
    parser.add_argument("--watch_metric", type=str, default='NDCG@10', help="Metric to watch for early stopping.")

    parser.add_argument('--mask_prob', type=float, default=0.2, help='Masking probability for BERT4Rec.')
    
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA even if available.")
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU device ID.')
    parser.add_argument("--log", default=True, action="store_true", help="Enable logging.")
    
    args = parser.parse_args()
    return args


def create_dual_view_model(args, user_num, item_num, device):
    model_name = args.model_name
    if "sasrec" in model_name:
        return DualViewSASRec(user_num, item_num, device, args)
    elif "gru4rec" in model_name:
        return DualViewGRU4Rec(user_num, item_num, device, args)
    elif "bert4rec" in model_name:
        return DualViewBERT4Rec(user_num, item_num, device, args)
    else:
        raise ValueError(f"Unsupported backbone for dual view model: {model_name}")


def main():
    args = get_args()
    
    run_name = f"{args.dataset}_{args.model_name}_alpha{args.alpha}_lambda{args.lambda_c}"
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    log_manager = Logger(args)
    logger, writer = log_manager.get_logger()
    args.now_str = log_manager.get_now_str()
    
    logger.info(f"Full output directory: {os.path.join(args.output_dir, args.now_str)}")
    logger.info(f"Using device: {device}")
    
    logger.info('='*80)
    logger.info('Training Dual-View Model with Contrastive Loss')
    logger.info('='*80)
    for arg, value in sorted(vars(args).items()):
        logger.info(f'  {arg}: {value}')
    logger.info('='*80)
    
    logger.info('Loading data...')
    if args.model_name == 'gru4rec':
        generator = GRU4RecGenerator(args, logger, device)
    elif args.model_name == 'bert4rec':
        generator = BERT4RecGenerator(args, logger, device)
    else:
        generator = Generator(args, logger, device)

    user_num, item_num = generator.get_user_item_num()
    logger.info(f'Number of users: {user_num}, Number of items: {item_num}')
    
    args.user_num = user_num
    args.item_num = item_num
    
    logger.info(f'Creating dual-view model with backbone: {args.model_name}...')
    model = create_dual_view_model(args, user_num, item_num, device)
    logger.info(f'Model created. Total parameters: {sum(p.numel() for p in model.parameters())}')
    
    trainer = SeqTrainer(args, logger, writer, device, generator)
    trainer.model = model
    
    trainer._set_optimizer()
    trainer._set_scheduler()
    
    logger.info('***** Starting training *****')
    _, best_epoch = trainer.train()
    
    logger.info('***** Final test evaluation *****')
    test_results, _ = trainer.test()
    
    logger.info('='*80)
    logger.info('Training and evaluation finished!')
    logger.info(f'Best model saved from epoch: {best_epoch}')
    logger.info('Final test results:')
    for metric, value in sorted(test_results.items()):
        if isinstance(value, (int, float)):
            logger.info(f'  {metric}: {value:.4f}')
        else:
            logger.info(f'  {metric}: {value}')
    logger.info('='*80)
        
    log_manager.end_log()

if __name__ == '__main__':
    main()

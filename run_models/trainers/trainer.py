import os
import torch
from tqdm import trange
from train_models.utils.earlystop import EarlyStopping
from train_models.utils.utils import get_n_params,format_metrics
from train_models.models.SASRec import SASRec
from train_models.models.GRU4Rec import GRU4Rec
from train_models.models.Bert4Rec import Bert4Rec


class Trainer(object):
    """Base trainer class for training and evaluating sequential recommendation models"""

    def __init__(self, args, logger, writer, device, generator):
        self.args = args
        self.logger = logger
        self.writer = writer
        self.device = device
        self.user_num, self.item_num = generator.get_user_item_num()
        self.start_epoch = 0

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        logger.info('# of model parameters: ' + str(get_n_params(self.model)))

        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()

        self.loss_func = torch.nn.BCEWithLogitsLoss()

        self.train_loader = generator.make_trainloader()
        self.valid_loader = generator.make_evalloader()
        self.test_loader = generator.make_evalloader(test=True)
        self.generator = generator

        self.watch_metric = args.watch_metric if hasattr(args, 'watch_metric') else 'NDCG@10'

    def _create_model(self):
        """Create model"""
        if self.args.model_name == "sasrec":
            self.model = SASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "gru4rec":
            self.model = GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "bert4rec":
            self.model = Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        else:
            raise ValueError(f"Unsupported model: {self.args.model_name}")

        self.model.to(self.device)

    def _load_pretrained_model(self):
        """Load pretrained model for continued training (modified to support pt format)"""
        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.pt')

        if not os.path.exists(checkpoint_path):
            old_checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')
            if os.path.exists(old_checkpoint_path):
                self.logger.warning(f"Loading old format model from {old_checkpoint_path}")
                model_dict = self.model.state_dict()
                checkpoint = torch.load(old_checkpoint_path, map_location=self.device, weights_only=False)

                if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
                    pretrained_dict = checkpoint
                else:
                    pretrained_dict = checkpoint.get('state_dict', checkpoint)

                new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                model_dict.update(new_dict)
                self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
                self.model.load_state_dict(model_dict)
                return
            else:
                raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path} or {old_checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['state_dict']

            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
            self.model.load_state_dict(model_dict)

            if 'args' in checkpoint:
                loaded_args = checkpoint['args']
                self.logger.info("Loaded model configuration:")
                for key, value in loaded_args.items():
                    if hasattr(self.args, key):
                        current_value = getattr(self.args, key)
                        if current_value != value:
                            self.logger.warning(f"Config mismatch - {key}: current={current_value}, loaded={value}")
                    else:
                        self.logger.info(f"Loaded config - {key}: {value}")

            if 'best_epoch' in checkpoint:
                self.start_epoch = checkpoint['best_epoch']
                self.logger.info(f"Resuming from epoch {self.start_epoch}")

        else:
            model_dict = self.model.state_dict()
            new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.logger.info('Total loaded parameters: {}, update: {}'.format(len(checkpoint), len(new_dict)))
            self.model.load_state_dict(model_dict)

    def _set_optimizer(self):
        """Set optimizer"""
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.args.lr,
                                         weight_decay=self.args.l2)

    def _set_scheduler(self):
        """Set learning rate scheduler"""
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=self.args.lr_dc_step,
                                                        gamma=self.args.lr_dc)

    def _set_stopper(self):
        """Set early stopping"""
        os.makedirs(self.args.output_dir, exist_ok=True)

        self.stopper = EarlyStopping(patience=self.args.patience,
                                    verbose=True,
                                    path=self.args.output_dir,
                                    trace_func=self.logger.info)
        
    def _train_one_epoch(self, epoch):
        """Train one epoch, must be implemented in subclass"""
        raise NotImplementedError("Must be implemented in subclass")

    def _prepare_train_inputs(self, data):
        """Prepare training inputs, adapt according to model type"""
        inputs = {}
        var_names = self.generator.train_dataset.var_name

        if len(data) != len(var_names):
            self.logger.warning(f"Data length {len(data)} != variable name length {len(var_names)}!")

        for i, var_name in enumerate(var_names):
            if i < len(data):
                inputs[var_name] = data[i]

        if self.args.model_name == "bert4rec":
            required_fields = ["seq", "pos", "neg", "positions"]

            param_mapping = {
                "tokens": "seq",
                "labels": "pos",
                "neg_labels": "neg"
            }

            for old_name, new_name in param_mapping.items():
                if old_name in inputs and new_name not in inputs:
                    inputs[new_name] = inputs[old_name]

            for field in required_fields:
                if field not in inputs:
                    if field == "positions" and "seq" in inputs:
                        seq_len = inputs["seq"].size(1)
                        positions = torch.arange(1, seq_len + 1).unsqueeze(0).expand(inputs["seq"].size(0), -1).to(self.device)
                        inputs["positions"] = positions
                    else:
                        self.logger.error(f"BERT4Rec missing required parameter: {field}")

        return inputs

    def _prepare_eval_inputs(self, data):
        """Prepare evaluation inputs"""
        inputs = {}
        assert len(self.generator.eval_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.eval_dataset.var_name):
            inputs[var_name] = data[i]
        return inputs

    def eval(self, epoch=0, test=False):
        """Evaluate model, must be implemented in subclass"""
        raise NotImplementedError("Must be implemented in subclass")

    def train(self, no_eval=False):
        """Main training loop"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        self.logger.info("  Training data file: %s", self.args.train_data_path)
        
        res_list = []
        eval_epochs = []

        if isinstance(self.args.eval_ks, str):
            eval_ks = [int(k) for k in self.args.eval_ks.split(',')]
        elif isinstance(self.args.eval_ks, list):
            eval_ks = self.args.eval_ks
        else:
            raise TypeError(f"Unsupported type for eval_ks: {type(self.args.eval_ks)}. Must be str or list.")
            
        self.logger.info(f"  Evaluation k values = {eval_ks}")

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):
            self._train_one_epoch(epoch)
            
            if no_eval:
                continue

            if (epoch % self.args.eval_steps) == 0:
                metric_dict = self.eval(epoch=epoch, ks=eval_ks)
                res_list.append(metric_dict)
                eval_epochs.append(epoch)
                
                metrics_table = format_metrics(metric_dict, ks=eval_ks)
                self.logger.info(f"\nEvaluation results (Epoch {epoch}):\n{metrics_table}")
                
                user_num, item_num = self.generator.get_user_item_num()
                extra_conf = {'user_num': user_num, 'item_num': item_num}

                self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.args, extra_config=extra_conf)

                if self.stopper.early_stop:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        if no_eval:
            self.logger.info(f"Training completed after {self.args.num_train_epochs} epochs (no eval mode).")
            final_epoch = self.start_epoch + int(self.args.num_train_epochs) - 1
            user_num, item_num = self.generator.get_user_item_num()
            extra_conf = {'user_num': user_num, 'item_num': item_num}
            self.stopper.save_checkpoint(0, model_to_save, self.args, extra_config=extra_conf, is_last=True)
            return {}, final_epoch

        if not res_list:
            self.logger.warning("No evaluation results available. Training may have failed.")
            metric_dict = self.eval(epoch=self.start_epoch + int(self.args.num_train_epochs) - 1, ks=eval_ks)
            res_list.append(metric_dict)
            best_epoch = self.start_epoch + int(self.args.num_train_epochs) - 1
            self.stopper.best_epoch = best_epoch
        else:
            best_epoch = self.stopper.best_epoch
            self.logger.info(f"Best epoch: {best_epoch}, Available eval epochs: {eval_epochs}")

            if best_epoch not in eval_epochs:
                closest_idx = min(range(len(eval_epochs)), key=lambda i: abs(eval_epochs[i] - best_epoch))
                self.logger.warning(f"Best epoch {best_epoch} was not evaluated. Using closest evaluated epoch {eval_epochs[closest_idx]} instead.")
                best_epoch = eval_epochs[closest_idx]

        best_idx = eval_epochs.index(best_epoch) if best_epoch in eval_epochs else 0

        if 0 <= best_idx < len(res_list):
            best_res = res_list[best_idx]
            best_metrics_table = format_metrics(best_res, ks=eval_ks)
            self.logger.info(f"\nBest evaluation results (Epoch {best_epoch}):\n{best_metrics_table}")
        else:
            self.logger.warning(f"Could not find best results. Using last evaluation results instead.")
            best_res = res_list[-1] if res_list else {}

        self.logger.info("Running final test evaluation...")
        res = self.eval(test=True, ks=eval_ks)
        test_metrics_table = format_metrics(res, ks=eval_ks)
        self.logger.info(f"\nTest results:\n{test_metrics_table}")

        return res, best_epoch

    def test(self):
        """Test directly (modified to support pt format loading)"""
        model_path = os.path.join(self.args.output_dir, 'pytorch_model.pt')
        if os.path.exists(model_path):
            self.logger.info(f"Loading best model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                self.logger.info(f"Loaded model from epoch {checkpoint.get('best_epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
        else:
            self.logger.warning(f"No saved model found at {model_path}. Using current model state.")

        res = self.eval(test=True)
        return res, -1

    def get_model(self):
        """Get model"""
        return self.model

    def get_model_param_num(self):
        """Get model parameter count"""
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freeze_num = total_num - trainable_num
        return freeze_num, trainable_num
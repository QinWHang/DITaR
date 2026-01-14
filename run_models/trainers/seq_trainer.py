import os
import time
import torch
import numpy as np
from tqdm import tqdm
from train_models.trainers.trainer import Trainer
from train_models.utils.utils import metric_report,record_csv


class SeqTrainer(Trainer):
    """Sequential recommendation model trainer"""

    def __init__(self, args, logger, writer, device, generator):
        super().__init__(args, logger, writer, device, generator)

    def _train_one_epoch(self, epoch):
        """Train one epoch"""
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:
            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)
            loss = self.model(**inputs)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)
        self.scheduler.step()

        return np.mean(train_time)

    def eval(self, epoch=0, test=False, ks=[10, 20]):
        """Evaluate model (modified to support pt format model loading)"""
        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_path = os.path.join(self.args.output_dir, 'pytorch_model.pt')
            if os.path.exists(model_path):
                self.logger.info(f"Loading model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.logger.info(f"Loaded model from epoch {checkpoint.get('best_epoch', 'unknown')} with score {checkpoint.get('best_score', 'unknown')}")

                    if 'args' in checkpoint:
                        loaded_args = checkpoint['args']
                        self.logger.info("Verifying model configuration compatibility...")

                else:
                    self.model.load_state_dict(checkpoint)
                    self.logger.info("Loaded model in legacy format")

                self.model.to(self.device)
            else:
                old_model_path = os.path.join(self.args.output_dir, 'pytorch_model.bin')
                if os.path.exists(old_model_path):
                    self.logger.warning(f"PT file not found. Loading legacy model from {old_model_path}")
                    model_state_dict = torch.load(old_model_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(model_state_dict)
                    self.model.to(self.device)
                else:
                    self.logger.warning(f"No model file found at {model_path} or {old_model_path}. Using current model state.")

            test_loader = self.test_loader
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader

        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():
                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)
                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy(), ks=ks)

        if not test:
            for metric, value in res_dict.items():
                self.writer.add_scalar(f'Eval/{metric}', value, epoch)

        if test:
            record_csv(self.args, res_dict)

        return res_dict
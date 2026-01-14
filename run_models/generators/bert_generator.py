from train_models.generators.generator import BERT4RecGenerator
from train_models.generators.data import BertRecTrainDataset
from torch.utils.data import DataLoader, RandomSampler
from train_models.utils.utils import unzip_data, concat_data


class BertGenerator(BERT4RecGenerator):
    """BERT4Rec data generator implementation"""

    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)

    def make_trainloader(self):
        """Create BERT4Rec training data loader"""
        train_dataset = unzip_data(self.train, aug=False)
        self.train_dataset = BertRecTrainDataset(self.args, train_dataset, self.item_num, self.args.max_len)

        train_dataloader = DataLoader(self.train_dataset,
                                     sampler=RandomSampler(self.train_dataset),
                                     batch_size=self.bs,
                                     num_workers=self.num_workers)

        return train_dataloader
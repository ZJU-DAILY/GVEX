import shutil
import os

import hydra
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from gnnNets import get_gnnNets
from dataset import get_dataloader, get_dataset
from utils import check_dirs, set_seed


class TrainModel(object):
    """
    a TrainModel to train GNN_NET

    """

    def __init__(self,
                 model,
                 dataset,
                 device,
                 save_dir,
                 save_name,
                 ** kwargs
                 ):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.optimizer = None
        self.is_save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dirs(save_dir)

        dataloader_params = kwargs.get('dataloader_params')
        self.loader = get_dataloader(self.dataset, **dataloader_params)

    def __loss__(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def _train_batch(self, data, labels):
        logits = self.model(data=data)
        loss = self.__loss__(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _eval_batch(self, data, labels):
        self.model.eval()
        logits = self.model(data)
        loss = self.__loss__(logits, labels)
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        losses, accs = [], []
        for batch in self.loader['eval']:
            batch.to(self.device)
            loss, preds = self._eval_batch(batch, batch.y)
            losses.append(loss)
            accs.append(preds == batch.y)
        eval_loss = torch.tensor(losses).mean().item()
        eval_acc = torch.cat(accs, dim=-1).float().mean().item()

        return eval_loss, eval_acc

    def test(self):
        state_dict = torch.load(os.path.join(self.save_dir, f'{self.save_name}_best.pth'))['net']
        # print(os.path.join(self.save_dir, f'{self.save_name}_best.pth'))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        losses, acc, test_preds = [], [], []
        for batch in self.loader['test']:
            batch.to(self.device)
            loss, preds = self._eval_batch(batch, batch.y)
            losses.append(loss)
            test_preds.append(preds)
            acc.append(preds == batch.y)

        test_loss = torch.tensor(losses).mean().item()
        test_preds = torch.cat(test_preds, dim=-1)
        test_acc = torch.cat(acc, dim=-1).float().mean().item()

        print(f'test_loss: {test_loss:0.4f}, test_acc: {test_acc:0.4f}')
        return test_loss, test_acc, test_preds

    def train(self, train_params=None, optimizer_params=None):
        num_epochs = train_params['num_epochs']
        num_early_stop = train_params['num_early_stop']
        milestones = train_params['milestones']
        gamma = train_params['gamma']

        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters())
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params)

        if milestones is not None and gamma is not None:
            lr_schedule = MultiStepLR(self.optimizer,
                                      milestones=milestones,
                                      gamma=gamma)
        else:
            lr_schedule = None

        self.model.to(self.device)
        best_eval_acc = 0.0
        best_eval_loss = 0.0
        early_stop_counter = 0
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            losses = []
            for batch in self.loader['train']:
                batch = batch.to(self.device)
                loss = self._train_batch(batch, batch.y)
                losses.append(loss)
            train_loss = torch.FloatTensor(losses).mean().item()

            with torch.no_grad():
                eval_loss, eval_acc = self.eval()
            print(f'Epoch:{epoch}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}')
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                    break

            if lr_schedule:
                lr_schedule.step()

            if best_eval_acc <= eval_acc:
                is_best = True
                best_eval_acc = eval_acc
            recording = {'epoch': epoch, 'is_best': str(is_best)}
            if self.is_save:
                self.save_model(is_best, recording=recording)

    def save_model(self, is_best=False, recording=None):
        self.model.to('cpu')
        state = {'net': self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
       
        latest_pth_name = f'{self.save_name}_latest.pth'
        best_path_name = f'{self.save_name}_best.pth'
        ckpt_path = os.path.join(self.save_dir, latest_pth_name) 

        torch.save(state, ckpt_path)
        if is_best:
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_path_name))
        self.model.to(self.device)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    config.models.gnn_savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    config.models.params = config.models.params[config.datasets.dataset_name]

    set_seed(config.random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    dataloader_params = {
        'batch_size': config.models.params.batch_size,
        'data_split_ratio': config.datasets.data_split_ratio,
        'seed': config.datasets.seed
    }
    train_params = {
        'num_epochs': config.models.params.num_epochs,
        'num_early_stop': config.models.params.num_early_stop,
        'milestones': config.models.params.milestones,
        'gamma': config.models.params.gamma
    }
    optimizer_params = {
        'lr': config.models.params.learning_rate,
        'weight_decay': config.models.params.weight_decay
    }
    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)

    trainer = TrainModel(model=model,
                         dataset=dataset,
                         device=device,
                         save_dir=os.path.join(config.models.gnn_savedir, config.datasets.dataset_name),
                         save_name=f'{config.models.gnn_name}_{len(config.models.params.gnn_latent_dim)}l',
                         dataloader_params=dataloader_params)

    trainer.train(train_params, optimizer_params)
    _, _, _ = trainer.test()


if __name__ == '__main__':
    main()
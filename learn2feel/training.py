from __future__ import annotations

import copy
import numpy as np
import torch
import torch.optim

import typing
if typing.TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard.writer import SummaryWriter

class training_wrapper():
    """ Wrapper to perform training and evaluation of a model.
    Maintains the model, optimizer, loss criterion, device, and logger.

    :param torch.nn.Module model: pyTorch model to train or evaluate.
    :param torch.optim.optimizer.Optimizer optimizer: pyTorch optimizer object.
    :param torch.nn.Module distance_function: class for computing the distance
        between two probability distributions.
    :param torch.nn.Module criterion: class for computing the loss.
    :param torch.device device: torch device on which computation takes place.
    :param tensorboard.writer.SummaryWriter writer: tensorboard loss logger.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: Optimizer,
                 distance_function: torch.nn.Module,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 writer: SummaryWriter):

        self.model = model
        self.optimizer = optimizer
        self.distance_function = distance_function
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self._result_dict={}

    def train(self, 
              dataloader_dict: typing.Dict[str,DataLoader],
              epochs: int,
              test_interval: int = 10):
        """Perform model training for a fixed number of epochs. Evaluate 
        on the validation set at a fixed interval. Populates the property 
        result_dict with training and validation results from the highest 
        scoring validation.

        :param dataloader_dict: Dictionary containing 'train' and 
            'validation' dataloaders
        :param test_interval: How often to evaluate on the validation 
            set, defaults to 10
        """
        self._result_dict={}
        best_loss = 2
        self.model.train().to(self.device)
        
        for epoch in range(epochs):
            epoch_loss, train_detail = self.process_epoch(dataloader_dict['train'])
            self.writer.add_scalar('Training loss',epoch_loss,epoch)

            # Test the model on the validation set. 
            if (test_interval is not None) and ((epoch+1)%test_interval==0):
                val_loss, val_detail = self.evaluate(dataloader_dict['validation'])
                self.writer.add_scalar('Validation loss',val_loss,epoch)
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.populate_results('train',epoch_loss,train_detail,epoch)
                    self.populate_results('val',val_loss,val_detail)
    
    def evaluate(self, dataloader: DataLoader) -> tuple[torch.Tensor, dict]:
        """
        Evaluation wrapper to perform a single evaluation pass given
        a dataloader.
        """
        self.model.eval()
        with torch.no_grad():
            total_loss, detailed_loss = self.process_epoch(dataloader,False)
        return total_loss, detailed_loss # type: ignore

    def process_epoch(self, dataloader: DataLoader,
                      training: bool = True) -> tuple[torch.Tensor, dict]:
        """Process all data from a single dataloader. If training is 
            True, train the model. Otherwise only perform evaluation.

        :return: The loss averaged over all subjects and the subject-wise loss
        :rtype: tuple[torch.Tensor, dict]
        """
        for batch in dataloader:
            X_right,X_left,labels,subjects,_ = batch
            distances = self.run_batch(X_right,X_left)

            subj_loss = self.subject_loss(distances,labels,
                                          subjects) 
            train_loss = sum(subj_loss.values())/len(subj_loss) 
            detail_loss = {i:j.item() for i,j in subj_loss.items()}
            if training:
                self.optimizer.zero_grad()
                train_loss.backward()  # type: ignore
                self.optimizer.step()

        return train_loss, detail_loss  # type: ignore
    
    def run_batch(self,
                  X_right: torch.Tensor,
                  X_left: torch.Tensor) -> torch.Tensor:
        """Process a single batch of data and marginals.

        :param X_right: data[0] and corresponding marginals[1] from the 
            right-handed interactions 
        :type X_right: torch.Tensor
        :param X_left: data[0] and corresponding marginals[1] from the 
            left-handed interactions
        :type X_left: torch.Tensor

        :return: computed distances between the interaction pairs
        :rtype: torch.Tensor
        """
        out_right = self.model(X_right[0].to(self.device))
        out_left = self.model(X_left[0].to(self.device))
        # marginals
        mu = X_right[1].to(self.device)
        nu = X_left[1].to(self.device)
        distances_batch = self.distance_function(out_right,out_left,mu,nu)[0].to('cpu')
        return distances_batch
    
    def subject_loss(self,
                     distances: torch.Tensor,
                     labels: torch.Tensor,
                     subject_list: torch.Tensor) -> typing.Dict[str,torch.Tensor]:
        """Computes the per-subject loss.

        :param distances: computed distances between the interaction pairs
        :type distances: torch.Tensor
        :param labels: similarity rating of each trial
        :type labels: torch.Tensor
        :param subject_list: which subject performed each trial
        :type subject_list: torch.Tensor

        :return: dictionary of losses by subject
        :rtype: typing.Dict[str,torch.Tensor]
        """
        loss = {}
        for s_id in np.unique(subject_list):
            smp_ixs = torch.where(subject_list==s_id)[0]
            subject_loss = self.criterion(distances[smp_ixs],labels[smp_ixs])
            if torch.isnan(subject_loss): continue
            loss[int(s_id)] = subject_loss
        return loss

    def populate_results(self, set: str, loss: torch.Tensor, 
                         subj_loss: dict, epoch = None):
        '''
        Populate the results dictionary with averaged and subject-wise
        losses. Indicate which data split the results are from (e.g.
        'train', 'val', and 'test'). If `epoch` is specified, save the 
        epoch and the model state. 

        :param str set: Data split whose performance is being recorded.
        :param torch.Tensor loss: The loss on the designated `set`.
        :param dict subj_loss: The subject-wise loss on the designated 
            `set`.
        :param int epoch: If epoch is specified, populate the dictionary 
            with the current model state and epoch.
        '''
        if epoch is not None:
            self._result_dict['model'] = copy.deepcopy(self.model)
            self._result_dict['epoch'] = epoch
        self._result_dict[f'{set}_loss'] = loss.item()
        self._result_dict[f'{set}_subj_loss'] = subj_loss

    @property
    def result_dict(self):
        """Dictionary for storing results, best-performing model, and 
        other relevant data during training, validation, and testing. 

        Populated using calls to :py:meth:`populate_results()<learn2feel.training.training_wrapper.populate_results>`.
        """
        return self._result_dict

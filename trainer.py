#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:16:17 2021

@author: jannik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from metrics import accuracy, miou
from postprocessing import Postprocessor, plot_training, plot_confusionMatrix, compute_confusionMatrix


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 analysis: Postprocessor,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 target_accuracy: float = 1.0,
                 earlyStopCriterion: int = 0
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.analysis = analysis
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.learningRateScheduler = True if self.lr_scheduler is not None else False
        self.epochs = epochs
        self.epoch = epoch
        self.target_accuracy = target_accuracy
        self.earlyStopCrit = earlyStopCriterion

        if "DiceCeLoss" in str(self.criterion):
            self.loss = "diceCe"
        elif "DiceLoss" in str(self.criterion):
            self.loss = "dice"
        else:
            self.loss = "ce"

        self.training_accuracy = []
        self.training_loss = []
        self.training_iou = []
        self.validation_accuracy = []
        self.validation_loss = []
        self.validation_iou = []
        
        self.validation_tn = []
        self.validation_fp = []
        self.validation_fn = []
        self.validation_tp = []

    def run_trainer(self):

        from tqdm import trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter
            
            if (self.epoch <= 2) or (self.validation_accuracy[-1]/100 < self.target_accuracy):
                """Training block"""
                self._train()

                """Validation block"""
                if (self.validation_DataLoader is not None):
                    self._validate()

                """Early stopping criterion block"""
                if self.earlyStopCrit != 0:
                    if self.epoch == 1:
                        self.minValidLoss = self.validation_loss[-1]
                        self.earlyStopCounter = 0
                    else:
                        if self.validation_loss[-1] < self.minValidLoss:
                            self.minValidLoss = self.validation_loss[-1]
                            self.earlyStopCounter = 0
                        else:
                            self.earlyStopCounter += 1
                            if self.earlyStopCounter == self.earlyStopCrit:
                                break

                """Learning rate scheduler block"""          
                if (self.lr_scheduler is not None) and (self.epoch != self.epochs):
                    self.lr_scheduler.step()
                #print("Learning rate: " + str(self.lr_scheduler.get_last_lr()))
            else:
                self.epoch -= 1
                break
        
        # Evaluate loss
        plot_training(self.training_accuracy,
                      self.training_loss,
                      self.training_iou,
                      self.validation_accuracy,
                      self.validation_loss,
                      self.validation_iou)
        plt.tight_layout()
        if self.analysis.save_metrics:
            plt.savefig(self.analysis.root / ("Output/metrics_"+self.analysis.dataset_name+"_"+str(self.model.n_blocks)+"blocks_"+
                                              str(self.model.start_filters)+"filter_"+str(self.training_DataLoader.batch_size)+
                                              "batch_"+str(self.optimizer.param_groups[0]['lr'])+"lrScheduled"+str(self.learningRateScheduler)+"_"+str(self.epoch)+"epochs_"+self.model.activation+self.model.normalization+self.loss+".png"))        
            metrics_df = pd.DataFrame(list(zip(self.training_accuracy,
                                               self.training_iou,
                                               self.training_loss,
                                               self.validation_accuracy,
                                               self.validation_iou,
                                               self.validation_loss,
                                               self.validation_tn,
                                               self.validation_fp,
                                               self.validation_fn,
                                               self.validation_tp
                                               )),
                                     columns=['train_accuracy',
                                              'train_iou',
                                              'train_loss',
                                              'valid_accuracy',
                                              'valid_iou',
                                              'valid_loss',
                                              'true_negative',
                                              'false_positive',
                                              'false_negative',
                                              'true_positive'
                                              ])
            metrics_df.to_pickle(self.analysis.root / ("Output/Dataframes/metricsDf_"+self.analysis.dataset_name+"_"+str(self.model.n_blocks)+"blocks_"+
                                                       str(self.model.start_filters)+"filter_"+str(self.training_DataLoader.batch_size)+
                                                       "batch_"+str(self.optimizer.param_groups[0]['lr'])+"lrScheduled"+str(self.learningRateScheduler)+"_"+str(self.epoch)+"epochs_"+self.model.activation+self.model.normalization+self.loss+".pkl"))
        return 

    def _train(self):
        
        from tqdm import tqdm

        self.model.train()  # train mode
        train_accuracy = [] 
        train_losses = []
        train_iou = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            acc = accuracy(out, target)
            iou = miou(out, target, self.model.out_channels)
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_accuracy.append(acc)
            train_iou.append(iou)
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: loss: {loss_value:.2f}, acc: {acc:.1f}%, iou: {iou:.2f}')  # update progressbar

        self.training_accuracy.append(np.mean(train_accuracy))
        self.training_iou.append(np.mean(train_iou))
        self.training_loss.append(np.mean(train_losses))

        batch_iter.close()

    def _validate(self):

        from tqdm import tqdm

        self.model.eval()  # evaluation mode
        valid_accuracy = [] 
        valid_losses = []
        valid_iou = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        y_pred = []
        y_true = []

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                acc = accuracy(out, target)
                iou = miou(out, target, self.model.out_channels)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_accuracy.append(acc)
                valid_iou.append(iou)
                valid_losses.append(loss_value)
                   
                y_pred.extend(np.squeeze(torch.argmax(out.data, 1).cpu().numpy()).flatten())
                y_true.extend(np.squeeze(target.cpu().numpy()).flatten())

                batch_iter.set_description(f'Validation: loss: {loss_value:.2f}, acc: {acc:.1f}%, iou: {iou:.2f}')

        classes = [str(i) for i in list(range(self.model.out_channels))]
        self.validation_accuracy.append(np.mean(valid_accuracy))
        self.validation_iou.append(np.mean(valid_iou))
        self.validation_loss.append(np.mean(valid_losses))
        
        if len(classes) == 2:
            cf_matrix_df = compute_confusionMatrix(y_pred=y_pred, 
                                                   y_true=y_true, 
                                                   classes=classes)
            self.validation_tn.append(cf_matrix_df[classes[0]][classes[0]])
            self.validation_fp.append(cf_matrix_df[classes[1]][classes[0]])
            self.validation_fn.append(cf_matrix_df[classes[0]][classes[1]])
            self.validation_tp.append(cf_matrix_df[classes[1]][classes[1]])
        else:
            self.validation_tn.append(None)
            self.validation_fp.append(None)
            self.validation_fn.append(None)
            self.validation_tp.append(None)


        batch_iter.close()
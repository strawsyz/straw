#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/14 21:27
# @Author  : Shi
# @FileName: ml_experiment.py
# @Description：
from base.base_experiment import BaseExpriment


class MLExperiment(BaseExperiment):
    def __init__(self, model=None):
        if model is not None:
            self.model = model


    def train(self):
        self.prepare_data()
        assert self.X is not None
        assert self.Y is not None
        self.model.fit(self.X,self.Y)


    def prepare_data(self):
        pass

    def add_model(self,model):
        self.models.append(model)

    def get_score(self):
        return self.model.score(self.X,self.Y)

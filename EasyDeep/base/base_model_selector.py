class ScoreModel:
    def __init__(self, name, bigger_better=True, best_num=1, desciption=None):
        self.name = name
        from collections import namedtuple
        self.best_score_bean = namedtuple("BestModel",["socre", "model_path"])
        self.bigger_better = bigger_better
        self.best_num = best_num
        self.best_scores = {}
        self.description = desciption
        self.best_score = None

    def get_best_score(self, score, score_true, model_path):
        if self.best_score is None or self.best_score.score < score:
            self.best_score = self.best_score_bean(score_true, model_path)
        return self.best_score

    def is_needed_add(self, score, model_path):
        if score is None:
            raise RuntimeWarning("this socre is None")

        if not self.bigger_better:
            score = -score
            best_score = self.get_best_score(score, -score, model_path)
        else:
            best_score = self.get_best_score(score, score, model_path)

        if len(self.best_scores) < self.best_num:
            return True, best_score
        elif min([value for value in self.best_scores.values()]) < score:
            return True, best_score
        else:
            return False, best_score

    def add_score(self, score, model_path):
        if score is None:
            score = 0
        if len(self.best_scores) < self.best_num:
            pass
        else:
            worst_score = float("inf")
            path_worst_model = ""
            for model_path, value in self.best_scores.items():
                if value < worst_score:
                    worst_score = value
                    path_worst_model = model_path
            self.best_scores.pop(path_worst_model)
        if not self.bigger_better:
            score = - score
        # add new score
        self.best_scores[model_path] = score


class BaseModelSelector:
    def __init__(self, score_models: list = None, best_num=1):
        super(BaseModelSelector, self).__init__()
        self.best_num = best_num
        self.strict = False
        self.score_models = {}
        # self.best_models_path = {}
        if score_models is not None:
            for score_model in score_models:
                self.score_models[score_model.name] = score_model
                # self.best_models_path[score_model.name] = None

    def _add_record(self, scores: dict, model_path: str):
        # every score_model has a best model
        best_models = {}

        if self.strict:
            need_save_4_epoch = True
        else:
            need_save_4_epoch = False

        reason_2_need = []
        for name, score_value in scores.items():
            score_model = self.score_models[name]
            need_save_4_socre_model, best_score = score_model.is_needed_add(score_value, model_path)
            # if is_best_socre:
            best_models[name] = best_score
            if need_save_4_socre_model:
                reason_2_need.append(name)
            if self.strict:
                need_save_4_epoch = need_save_4_epoch and need_save_4_socre_model
                if not need_save_4_epoch:
                    break
            else:
                need_save_4_epoch = need_save_4_epoch or need_save_4_socre_model
        if need_save_4_epoch:
            for name, score_value in scores.items():
                score_model = self.score_models[name]
                # 给每个分数模型添加新的分数
                score_model.add_score(score_value, model_path)
        else:
            # 重新设置为空列表。防止因为逻辑的问题，导致没有保存的必要，但是依然有保存的理由
            reason_2_need = []

        return need_save_4_epoch, reason_2_need, best_models

    def regist_score_model(self, name, bigger_better=True, best_num=1, desciption=None):
        # 增加分数指标
        self.score_models[name] = ScoreModel(name, bigger_better, best_num, desciption)

    def add_record(self, recoder, model_path):
        recoder_dict = recoder.get_record_dict()
        return self._add_record(recoder_dict, model_path)

    def __str__(self):
        return str([score_model_name for score_model_name in self.score_models])

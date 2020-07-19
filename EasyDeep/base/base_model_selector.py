class ScoreModel:
    def __init__(self, name, bigger_better=True, best_num=1, desciption=None):
        self.name = name
        # 也许将最差的分数拿出来维护会比较好
        # key : path, value :score

        self.bigger_better = bigger_better
        self.best_num = best_num
        self.best_scores = {}
        self.description = desciption

    def is_needed_add(self, score):
        if len(self.best_scores) < self.best_num:
            return True
        if min([value for value in self.best_scores.values()]) < score:
            return True
        else:
            return False

    def add_score(self, score, model_path):
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
        self.best_num[model_path] = score


class BaseSelector:
    def __init__(self, score_models=None):
        # 每个分数保留前5名的模型
        self.best_num = 1
        self.score_models = {}
        self.strict = False
        if score_models is not None:
            self.score_models = score_models

    def _add_record(self, scores: dict, model_path: str):

        if self.strict:
            need_save = True
        else:
            need_save = False
        for name, score_value in scores.items():
            score_model = self.score_models[name]
            if self.strict:
                need_save = need_save and score_model.is_needed_add(score_value)
                if not need_save:
                    break
            else:
                need_save = need_save or score_model.is_needed_add(score_value)
                if need_save:
                    break
        if need_save:
            for name, score_value in scores.items():
                score_model = self.score_models[name]
                score_model.add_score(score_value, model_path)
        return need_save

    def add_score(self, name, bigger_better=True, best_num=1, desciption=None):
        # 增加分数指标
        self.score_models[name] = ScoreModel(name, bigger_better, best_num, desciption)

    def add_record(self, recoder, model_path):
        recoder_dict = recoder.get_recoder()
        self._add_record(recoder_dict, model_path)

    def __str__(self):
        return str([score_model.name for score_model in self.score_models])

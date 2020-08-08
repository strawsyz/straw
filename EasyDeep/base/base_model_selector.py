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
        if score is None:
            score = 0
        if not self.bigger_better:
            score = -score
        if len(self.best_scores) < self.best_num:
            return True
        if min([value for value in self.best_scores.values()]) < score:
            return True
        else:
            return False

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


class BaseSelector:
    def __init__(self, score_models: list = None, best_num=1):
        # 每个分数保留前5名的模型
        self.best_num = best_num
        self.strict = False
        self.score_models = {}
        self.bset_models_path = {}
        from collections import namedtuple
        if score_models is not None:
            for score_model in score_models:
                self.score_models[score_model.name] = score_model
                # 每一项指标保留最高分
                self.best_models_path[score_model.name] = None

    def _add_record(self, scores: dict, model_path: str):

        if self.strict:
            need_save = True
        else:
            need_save = False

        reason2need = []
        for name, score_value in scores.items():
            score_model = self.score_models[name]
            is_need_save = score_model.is_needed_add(score_value)
            if is_need_save:
                reason2need.append(name)
            if self.strict:
                need_save = need_save and is_need_save
                if not need_save:
                    break
            else:
                need_save = need_save or is_need_save
        if need_save:
            for name, score_value in scores.items():
                score_model = self.score_models[name]
                # 给每个分数模型添加新的分数
                score_model.add_score(score_value, model_path)
        else:
            # 重新设置为空列表。防止因为逻辑的问题，导致没有保存的必要，但是依然有保存的理由
            reason2need = []
        return need_save, reason2need

    def regist_score_model(self, name, bigger_better=True, best_num=1, desciption=None):
        # 增加分数指标
        self.score_models[name] = ScoreModel(name, bigger_better, best_num, desciption)

    def add_record(self, recoder, model_path):
        recoder_dict = recoder.get_record_dict()
        return self._add_record(recoder_dict, model_path)

    def __str__(self):
        return str([score_model_name for score_model_name in self.score_models])

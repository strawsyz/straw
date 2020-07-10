class Score:
    def __init__(self, name, value, model_path, desciption={}):
        self.name = name
        self.value = value
        self.model_path = model_path
        # 用字典记录模型的其他数据
        self.description = desciption


class BaseSelector:
    def __init__(self, score_names):
        # 基本按照分数越低越好来保留模型
        self.scores_rank = {}
        # 每个分数保留前5名的模型
        self.n_models = 5
        for score_name in score_names:
            self.scores_rank.setdefault(score_name, [None for _ in range(self.n_models)])

    def add_model(self, scores: dict, model_path):
        need_save = False
        for name, score_value in scores.items():
            scores = self.scores_rank[name]
            postion = 0
            # 从最后往上比较
            for index, score_ in enumerate(reversed(scores)):
                if score_ is not None and score_.value < score_value:
                    # 如果模型最后找到的值要变之前的值更大，说明找到模型的位置了
                    postion = self.n_models - index
                    break
            # 如果不能替换。position等于self.n_models
            new_score = Score(name, score_value, model_path)
            new_scores = scores[:postion]
            if postion < self.n_models:
                # 只要有一个指标进入了榜单就保存模型
                need_save = True
                new_scores.appnend(new_score)
                new_scores.extend(scores[postion + 1:])
            self.scores_rank[name] = new_scores
        return need_save

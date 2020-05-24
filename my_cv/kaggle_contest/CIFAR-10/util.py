class ModelChoose():
    """用于选择比较好的模型"""

    def __init__(self, model):
        super(ModelChoose, self).__init__()
        # 可能还需要区分，分数是越大越好，还是越小越好
        self.scores = {}
        self.min_scores = {}
        self.model = model
        # todo 用于保存多个比较好的模型
        self.models = []

    def set_min_score(self, score_name, min_score):
        self.min_scores[score_name] = min_score

    def add_scores(self, **kwargs):
        if self.choose_best_model(**kwargs):
            # 如果比之前的模型要好，就保存模型
            for key, value in kwargs.iteritems():
                self.scores[key] = value
            return True
        else:
            return False

    def choose_best_model(self, **kwargs):
        for key, value in kwargs.iteritems():
            if value > self.min_scores[key] and value > self.scores[key]:
                # 如果要大于最小值，而且要大于之前记录的值
                pass
            else:
                return False
        return True

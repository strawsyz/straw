# 模板模式
class Indexer:
    def process(self, text):
        text = self._normalize_text(text)
        words = self._split_text(text)
        words = self._remove_stop_words(words)
        stemmed_words = self._stem_words(words)
        return self._frequency(stemmed_words)

    def _normalize_text(self, text):
        return text.lower().strip()

    def _split_text(self, text):
        return text.split()

    def _remove_stop_words(self, text):
        raise NotImplementedError

    def _stem_words(self, words):
        raise NotImplementedError

    def _frequency(self, words):
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        return counts


class BasicIndexer(Indexer):
    _stop_words = ("he", "she", "is", "and", "or")

    def _remove_stop_words(self, words):
        return (word for word in words if word not in self._stop_words)

    def _stem_words(self, words):
        return (len(word) > 2 and word.rstrip('aeiouy') for word in words)


if __name__ == '__main__':
    indexer = BasicIndexer()
    print(indexer.process("I have a dog and you don't have"))

class StreamingConcatDataset:
    """
    複数のストリーミングシーケンス（SequenceForStreaming のインスタンス）を連結して
    ひとつのストリーミングデータセットとして扱えるようにするクラスです。
    
    各シーケンスを順にイテレーションし、全体として 1 つの連続したサンプルの流れを生成します。
    """
    def __init__(self, streaming_sequences):
        self.streaming_sequences = streaming_sequences

    def __iter__(self):
        for sequence in self.streaming_sequences:
            yield from sequence
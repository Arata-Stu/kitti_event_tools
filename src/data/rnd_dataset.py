from torch.utils.data import Dataset

# ----- 複数シーケンス統合の RandomDataset クラス -----
class RandomConcatDataset(Dataset):
    """
    複数の SequenceForRandom インスタンスを統合して、1 つのランダムアクセス用データセットとして扱えるようにします。
    返り値は各サンプルごとに
      {
         "images": [seq_len 枚の画像],
         "labels": [各画像のラベル情報リスト],
         "events": numpy 配列,
         "reset_state": bool
      }
    の辞書型となります。
    streaming の場合と同様に、個々のサンプルの構造は統一されています。
    """
    def __init__(self, sequences):
        """
        Parameters:
          - sequences: SequenceForRandom のインスタンスのリスト
        """
        self.sequences = sequences
        # 各シーケンスの長さリストを作成（各シーケンスは __len__ を実装しているものとする）
        self.sequence_lengths = [len(seq) for seq in sequences]
        # 累積長を求める（[len(seq1), len(seq1)+len(seq2), ...] のようなリスト）
        self.cumulative_lengths = []
        total = 0
        for l in self.sequence_lengths:
            total += l
            self.cumulative_lengths.append(total)

    def __len__(self):
        # 全シーケンスの合計サンプル数を返す
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # グローバルなインデックス idx から、対象となるシーケンスと
        # シーケンス内のローカルなインデックスを特定する
        seq_idx = 0
        for cum_len in self.cumulative_lengths:
            if idx < cum_len:
                break
            seq_idx += 1

        if seq_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[seq_idx - 1]
        return self.sequences[seq_idx][local_idx]

import random
from torch.utils.data import IterableDataset, get_worker_info

class ConcatStreamingDataset(IterableDataset):
    def __init__(self, sequence_datasets, batch_size, pad_last=False, pad_func=None, transform=None):
        """
        Parameters:
          sequence_datasets: 各シーケンス毎のMap-style Datasetのリスト
          batch_size: バッチ内に含めるサンプル数
          pad_last: バッチが満たない場合にpaddingを行うかどうか（Trueならpad_funcを使って不足分を埋める）
          pad_func: パディング用サンプルを生成する関数（例: datasetインスタンスのget_padding_sample()など）
          transform: 各サンプルに適用する変換
        """
        self.sequence_datasets = sequence_datasets
        self.batch_size = batch_size
        self.pad_last = pad_last
        self.pad_func = pad_func
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()
        # ワーカーごとに扱うシーケンスを全部使っても問題ない場合、シャーディングは行わず、
        # 各ワーカーで同じシーケンス集合からランダムサンプリングする。
        iterators = [self._infinite_iterator(ds) for ds in self.sequence_datasets]

        while True:
            batch = []
            # シーケンスリスト自体をランダムにシャッフルして順序に変化をつける
            random.shuffle(iterators)
            for it in iterators:
                sample = next(it)
                if self.transform is not None:
                    sample = self.transform(sample)
                batch.append(sample)
                if len(batch) == self.batch_size:
                    break
            # バッチが満たない場合の処理
            if len(batch) < self.batch_size:
                if self.pad_last and self.pad_func is not None:
                    while len(batch) < self.batch_size:
                        batch.append(self.pad_func())
                else:
                    break
            yield batch

    def _infinite_iterator(self, dataset):
        """データセットのイテレータを無限にリサイクルする"""
        while True:
            for sample in dataset:
                yield sample


class SharedStreamingDataset(IterableDataset):
    def __init__(self, sequence_datasets, batch_size, pad_last=False, pad_func=None, transform=None):
        """
        Parameters:
          sequence_datasets: 各シーケンス毎のMap-style Datasetのリスト
          batch_size: バッチ内のサンプル数
          pad_last: バッチが満たない場合にpaddingを行うかどうか
          pad_func: パディング用サンプル生成関数
          transform: 各サンプルに適用する変換
        """
        self.sequence_datasets = sequence_datasets
        self.batch_size = batch_size
        self.pad_last = pad_last
        self.pad_func = pad_func
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # シンプルな割り当て例：各シーケンスに対して、インデックスのmodでワーカーを振り分ける
        assigned_datasets = [ds for idx, ds in enumerate(self.sequence_datasets) if idx % num_workers == worker_id]

        iterators = [iter(ds) for ds in assigned_datasets]

        while True:
            batch = []
            for it in iterators:
                try:
                    sample = next(it)
                except StopIteration:
                    continue  # あるシーケンスが尽きた場合は飛ばす
                if self.transform is not None:
                    sample = self.transform(sample)
                batch.append(sample)
                if len(batch) == self.batch_size:
                    break
            if len(batch) == 0:
                break  # 全てのシーケンスが終了 → エポック終了
            if len(batch) < self.batch_size:
                if self.pad_last and self.pad_func is not None:
                    while len(batch) < self.batch_size:
                        batch.append(self.pad_func())
                else:
                    # 端数を破棄する（またはyieldする）
                    break
            yield batch

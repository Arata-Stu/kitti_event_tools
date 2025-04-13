import h5py
try:
    import hdf5plugin
except ImportError:
    pass


from .sequence_base import SequenceBase
# ----- SequenceRandom クラス -----

class SequenceForRandom(SequenceBase):
    """
    SequenceBase を継承し、ランダムアクセス用のシーケンスデータを __getitem__ により提供します。
    各サンプルは連続する seq_len フレームに対応し、画像、ラベル、イベントデータ、およびリセットフラグを含む辞書を返します。
    """
    def __getitem__(self, index):
        # 各アクセス時に h5py.File を開くことで、並列処理時のスレッドセーフ性を確保します。
        with h5py.File(str(self.event_file), 'r') as f:
            # 連続する seq_len フレームに対して画像とラベルを取得
            images = [self.load_image(i) for i in range(index, index + self.seq_len)]
            labels_seq = [self.labels.get(i, []) for i in range(index, index + self.seq_len)]
            # イベントデータも同様に連続する範囲を取得（ここではデータが "data" キーに保存されている前提）
            events = f["data"][index: index + self.seq_len]
        # randomデータセットの場合は, reset_stateは常にTrue
        reset_state =True

        outputs = {
            "images": images,
            "labels": labels_seq,
            "events": events,
            "reset_state": reset_state
        }
        # transform が指定されていれば、適用する
        if self.transform:
            outputs = self.transform(outputs)
            
        return outputs

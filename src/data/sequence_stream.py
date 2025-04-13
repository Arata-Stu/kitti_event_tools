import h5py
try:
    import hdf5plugin
except ImportError:
    pass

from .sequence_base import SequenceBase



class SequenceForStreaming(SequenceBase):
    """
    SequenceBase を利用して、シーケンス内の連続サンプルを逐次yieldするストリーミング版データセットです。
    
    1回のイテレーションで、シーケンス内の全連続した seq_len フレームのサンプルを順番に生成します。
    """
    def __iter__(self):
        # イベントデータは毎回ファイルをオープンするのではなく、1回だけオープンして使い回す
        with h5py.File(str(self.event_file), 'r') as f:
            # 各サンプルはインデックス0～length-1に対応（ランダムアクセス版の __getitem__ と同等の処理）
            for index in range(self.length):
                images = []
                labels_seq = []
                # 連続する seq_len フレーム分の画像とラベルを取得
                for i in range(index, index + self.seq_len):
                    img = self.load_image(i)
                    images.append(img)
                    # ラベルが存在しない場合は空リストになるように（もしくはそのまま）
                    labels_seq.append(self.labels.get(i, []))
                
                # h5py.File を使って、連続するイベントデータを取得
                # ※ ここでは、イベントデータがキー "events" に保存されている前提
                events = f["events"][index : index + self.seq_len]
                
                # サンプルの先頭がシーケンスの最初であれば reset_state = True とする
                reset_state = (index == 0)
                
                yield {
                    "images": images,         # 各画像は RGB の numpy 配列
                    "labels": labels_seq,       # フレームごとのラベル情報（リスト）
                    "events": events,           # 連続するイベントデータ（numpy 配列）
                    "reset_state": reset_state  # シーケンス初回を示すフラグ
                }


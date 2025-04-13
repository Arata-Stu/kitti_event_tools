from pathlib import Path
import h5py
try:
    import hdf5plugin
except ImportError:
    pass

from .sequence_base import SequenceBase

class SequenceForStreaming(SequenceBase):
    """
    SequenceBase を利用して、シーケンス内の連続サンプルを逐次 yield するストリーミング版データセットです。
    
    1回のイテレーションで、シーケンス内の全連続した seq_len フレームのサンプルを順番に生成します。
    """
    def __init__(self, data_dir: Path, sequence_name: str, ev_repr_name: str, seq_len: int, downsample: bool = False, transform=None):
        """
        コンストラクタで downsample 引数を受け取り、親クラスの初期化時に渡す。

        Parameters:
          - data_dir: データのルートディレクトリ
          - sequence_name: シーケンス名
          - ev_repr_name: イベント表現の名前
          - seq_len: シーケンス長（連続フレーム数）
          - downsample: True の場合、画像を1/2サイズにリサイズし、対応するラベルの bbox も同様に変換する
        """
        super().__init__(data_dir, sequence_name, ev_repr_name, seq_len=seq_len, downsample=downsample, transform=transform)

    def __iter__(self):
        # イベントデータは毎回ファイルをオープンするのではなく、1回だけオープンして使い回す
        with h5py.File(str(self.event_file), 'r') as f:
            # 各サンプルはインデックス 0～length-1 に対応（ランダムアクセス版の __getitem__ と同等の処理）
            for index in range(self.length):
                images = []
                labels_seq = []
                # 連続する seq_len フレーム分の画像とラベルを取得
                for i in range(index, index + self.seq_len):
                    img = self.load_image(i)
                    images.append(img)
                    # ラベルが存在しない場合は空リストになるように
                    labels_seq.append(self.labels.get(i, []))
                
                # h5py.File を使って、連続するイベントデータを取得
                # ※ ここでは、イベントデータがキー "data" に保存されている前提
                events = f["data"][index : index + self.seq_len]
                
                # サンプルの先頭がシーケンスの最初であれば reset_state = True とする
                reset_state = (index == 0)
                outputs = {
                    "images": images,         # 各画像は RGB の numpy 配列
                    "labels": labels_seq,       # 各フレームのラベル情報（リスト）
                    "events": events,           # 連続するイベントデータ（numpy 配列）
                    "reset_state": reset_state  # シーケンス初回を示すフラグ
                }
                # transform が指定されていれば、適用する
                if self.transform:
                    outputs = self.transform(outputs)

                yield outputs


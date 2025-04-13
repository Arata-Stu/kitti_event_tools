from pathlib import Path
import cv2
import h5py
try:
    import hdf5plugin
except ImportError:
    pass
from collections import defaultdict

# ----- 基底クラス SequenceBase -----
class SequenceBase:
    """
    KITTI のディレクトリ構造例:
    
    kitti
    ├── images
    │    ├── 0000/
    │         ├── 000000.png
    │         ├── 000001.png
    │         ├── ...
    │    ├── 0001/
    │    ├── <sequence_name>/
    │
    ├── labels
    │    ├── 0000.txt
    │    ├── 0001.txt
    │    ├── <sequence_name>.txt
    │
    ├── preprocessed
    │    └── <ev_repr_name>/
    │         ├── 0000.h5
    │         ├── 0001.h5
    │         ├── <sequence_name>.h5  
    """
    
    def __init__(self, data_dir: Path, sequence_name: str, ev_repr_name: str, seq_len: int):
        self.data_dir = data_dir
        self.sequence_name = sequence_name
        self.ev_repr_name = ev_repr_name
        self.seq_len = seq_len
        
        # 画像、ラベル、イベントのファイルパスを設定
        self.images_dir = self.data_dir / "images" / self.sequence_name
        self.labels_file = self.data_dir / "labels" / f"{self.sequence_name}.txt"
        self.events_dir = self.data_dir / "preprocessed" / self.ev_repr_name
        
        # 画像ファイル一覧の取得（ファイル名を数値順にソート）
        self.image_files = sorted(self.images_dir.glob("*.png"), key=lambda p: int(p.stem))
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"画像が見つかりません: {self.images_dir}")
        
        # イベントデータはシーケンス毎の 1 ファイルとする
        self.event_file = self.events_dir / f"{self.sequence_name}.h5"
        if not self.event_file.exists():
            raise FileNotFoundError(f"イベントファイルが見つかりません: {self.event_file}")
        
        # ラベルの読み込み（フレーム番号毎にグループ化）
        self.labels = self.load_labels(self.labels_file)
        
        # 画像とイベントデータの総フレーム数取得
        self.num_image_frames = len(self.image_files)
        with h5py.File(self.event_file, 'r') as f:
            # 仮定：キー "events" にイベントデータが保存され、shape は (N, ...) の形式
            self.num_event_frames = f["data"].shape[0]
        
        # 連続して seq_len フレーム取得可能なサンプル数
        self.total_frames = min(self.num_image_frames, self.num_event_frames)
        self.length = self.total_frames - self.seq_len + 1
        if self.length <= 0:
            raise ValueError("指定されたシーケンス長が利用可能なフレーム数に対して長すぎます。")
    
    def load_labels(self, file_path: Path):
        """
        ラベルファイルを読み込み、各行をパースしてフレーム番号毎にグループ化する。
        
        各行は以下の形式を仮定：
          <frame> <track_id> <type> <truncated> <occluded> <alpha> 
          <bbox_left> <bbox_top> <bbox_right> <bbox_bottom>
          <dim1> <dim2> <dim3>
          <loc1> <loc2> <loc3>
          <rotation_y>
        """
        if not file_path.exists():
            raise FileNotFoundError(f"ラベルファイルが見つかりません: {file_path}")
        
        labels_per_frame = defaultdict(list)
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if not fields:
                    continue
                try:
                    frame = int(fields[0])
                    label = {
                        "track_id": int(fields[1]),
                        "type": fields[2],
                        "truncated": float(fields[3]),
                        "occluded": int(fields[4]),
                        "alpha": float(fields[5]),
                        "bbox": list(map(float, fields[6:10])),
                        "dimensions": list(map(float, fields[10:13])),
                        "location": list(map(float, fields[13:16])),
                        "rotation_y": float(fields[16])
                    }
                    labels_per_frame[frame].append(label)
                except Exception as e:
                    raise ValueError(f"ラベル行のパースに失敗しました: {line}\nエラー: {e}")
        return labels_per_frame

    def load_image(self, index: int):
        """
        指定されたインデックスの画像を OpenCV を用いて読み込み、
        BGR→RGB に変換して返す。
        """
        if index < 0 or index >= self.num_image_frames:
            raise IndexError("画像インデックスが範囲外です")
        image_path = self.image_files[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
        # BGR→RGB 変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def __len__(self):
        """連続サンプルの総数（連続した seq_len フレームが取得可能な部分の数）を返す"""
        return self.length
    
    def __getitem__(self, index):
        """
        指定インデックスから連続する seq_len フレームの画像、ラベル、
        および対応するイベントデータを取得し、辞書型で返す。
        
        辞書のキー:
          "images": 長さ seq_len の画像リスト (各画像は RGB の numpy 配列)
          "labels": 各フレーム毎のラベル情報のリスト（ラベルが存在しない場合は空リスト）
          "events": numpy 配列 (形状: (seq_len, ...)) で連続フレーム分のイベントデータ
          "reset_state": このサンプルがシーケンスの最初かどうかを示すフラグ（最初のサンプルの場合 True）
        """
        if index < 0 or index >= self.length:
            raise IndexError("サンプルインデックスが範囲外です")
        
        images = []
        labels_seq = []
        for i in range(index, index + self.seq_len):
            img = self.load_image(i)
            images.append(img)
            labels_seq.append(self.labels.get(i, []))
        
        with h5py.File(self.event_file, 'r') as f:
            events = f["data"][index : index + self.seq_len]
        
        # サンプルの先頭がシーケンスの最初であれば reset_state True を付与
        reset_state = (index == 0)
        
        outputs = {
            "images": images,
            "labels": labels_seq,
            "events": events,
            "reset_state": reset_state
        }
        
        return outputs


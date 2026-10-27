# Physical AI 最終課題：CNNによる自律障害物回避

ROS 2 HumbleとGazebo Fortressを使用し、前方カメラ画像からCNNで障害物の位置を分類し、OpenCVによる距離推定と状態遷移制御を組み合わせて、差動二輪ロボットが赤い障害物を自律的に回避するシミュレーションです。Gazebo GUI内には、認識結果・推定距離・制御状態・Actionを表示するHUDプラグインを追加しています。

## デモ動画

[YouTubeでデモを見る](https://youtu.be/N8OKFfYinjE)

## 実装内容

- **CNNによる障害物位置分類**：`none` / `left` / `center` / `right`。収集した999枚の画像を使って学習し、検証時の正解率は96.98%でした（収集データに対する検証結果であり、未知環境での性能を保証するものではありません）。
- **OpenCVによる距離推定**：前方カメラ画像から赤色領域を抽出し、検出領域の見かけの幅から障害物までの距離を推定します。
- **状態遷移型の回避制御**：`APPROACH` → `TURN_AWAY` → `MOVE_ASIDE` → `RETURN_HEADING` → `PASS_OBSTACLE` → `STOP`。
- **遠距離からの接近**：遠方でCNNの分類信頼度が低下する場合に、距離推定を参照して低速接近する処理を追加しました。
- **Gazebo HUD**：Gazebo Transportで`/debug/status`を受信し、C++/QMLのGUIプラグインで3D画面上に推論状態を表示します。

## 動作環境（開発時）

- Ubuntu 22.04ベースのDockerコンテナ（ホストはUbuntu 24.04 / X11）
- ROS 2 Humble
- Gazebo Fortress / Ignition GUI 6.8.0 / Ignition Transport 11.4.1
- Python 3、PyTorch、OpenCV、cv_bridge
- `ros_gz_bridge`
- CMake、Qt5開発環境（HUDのビルド用）

GPUや依存ライブラリのインストール方法は環境により異なります。開発ではAI Robot BookのROS 2コンテナイメージを利用しました。

## リポジトリ構成

```text
pai-final-project/
├── README.md
├── .gitignore
├── ai_obstacle_avoidance/
│   ├── train_cnn.py
│   ├── realtime_inference.py
│   ├── distance_estimation.py
│   ├── autonomous_avoidance_v1.py
│   ├── ...
│   ├── autonomous_avoidance_v7.py  # 動画撮影で使用した最終版
│   └── models/obstacle_cnn_best.pth
├── gazebo/
│   ├── world.sdf
│   ├── LICENSE-2DRoboPrac.txt
│   └── meshes/
│       ├── 2DRoboPrac.dae
│       └── 2DRoboPrac.stl
└── gazebo_hud/
    ├── CMakeLists.txt
    ├── ObstacleHud.hh
    ├── ObstacleHud.cc
    ├── ObstacleHud.qml
    └── hud.qrc
```

## 実行方法（概要）

以下はROS 2 Humble・Gazebo Fortress・PyTorch等がインストール済みの環境を想定しています。GazeboはGUI表示に対応するX11環境から起動してください。

### 1. Gazebo HUDをビルド

```bash
cd gazebo_hud
cmake -S . -B build
cmake --build build -j2
mkdir -p ~/.ignition/gui/plugins
cp build/libObstacleHud.so ~/.ignition/gui/plugins/
```

### 2. Gazeboのリソースパスを設定して起動

```bash
# リポジトリのルートディレクトリで実行
export IGN_GAZEBO_RESOURCE_PATH="$(pwd)/gazebo${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
ign gazebo -r gazebo/world.sdf
```

### 3. ROS 2とGazeboのブリッジを起動

個別のターミナル（またはバックグラウンドプロセス）で実行します。

```bash
source /opt/ros/humble/setup.bash
ros2 run ros_gz_bridge parameter_bridge '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image'
```

```bash
source /opt/ros/humble/setup.bash
ros2 run ros_gz_bridge parameter_bridge '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist'
```

```bash
source /opt/ros/humble/setup.bash
ros2 run ros_gz_bridge parameter_bridge '/model/demoRobot/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry'
```

```bash
source /opt/ros/humble/setup.bash
ros2 run ros_gz_bridge parameter_bridge '/debug/status@std_msgs/msg/String]ignition.msgs.StringMsg'
```

### 4. 自律障害物回避プログラムを起動

```bash
source /opt/ros/humble/setup.bash
cd ai_obstacle_avoidance
python3 -u autonomous_avoidance_v7.py
```

**注意：** ソースコード内のモデル読み込みパス、動作許可フラグ（`ENABLE_MOTION_V6`）、カメラ解像度、回避距離等は配置・撮影時の最終設定と一致させてください。`autonomous_avoidance_v7.py`は以前のバージョンのクラスを継承しているため、依存する`autonomous_avoidance_v*.py`も削除しないでください。

## 評価結果と制約

- Gazebo上で障害物を回避し、停止状態まで遷移する動作を確認しました。
- 前方カメラは160×120ピクセル、約10Hzで取得するよう調整しました。
- CNNの検証正解率96.98%は、限定された学習・検証データに基づく数値です。
- 赤色の単一障害物を前提とした距離推定・状態遷移であり、任意の障害物や未知環境への汎化は未検証です。

## 参考資料・出典

- [N622のQiita記事（初期実装時に参照）](https://qiita.com/N622/items/9d89f77d85d9da0af29e) — GazeboとROS 2の連携に関する参考資料。
- [元のREADMEに記載されていたQiita記事](https://qiita.com/N622/items/6ca879f8b1f9d667e561) — 2Dロボット開発の関連資料。
- [OsawaKousei/2DRoboPrac_ws](https://github.com/OsawaKousei/2DRoboPrac_ws) — ロボットおよびGazeboモデルのベースとなった公開リポジトリ（Apache License 2.0）。`gazebo/world.sdf`にはカメラ・障害物・表示設定等の改変を加えています。
- [AI Robot Book Docker ROS 2環境](https://github.com/AI-Robot-Book/docker-ros2-desktop-ai-robot-book) — 実行環境の参考。
- [Gazebo Fortress Documentation](https://gazebosim.org/docs/fortress/) — Gazeboの設定・機能の参考。

## ライセンスと第三者著作物

ロボットモデルとそのメッシュのベースは[OsawaKousei/2DRoboPrac_ws](https://github.com/OsawaKousei/2DRoboPrac_ws)に由来し、同リポジトリのApache License 2.0に従います。上流プロジェクトのLICENSE文書は `gazebo/LICENSE-2DRoboPrac.txt` に同梱しています。上流にNOTICE文書や追加の著作権表示がある場合は、その条件にも従ってください。 本リポジトリ独自のコードについては、別途ライセンスを明示しない限り、公開されていても自由な再利用許諾を意味しません。

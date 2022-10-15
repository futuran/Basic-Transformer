from logging import (
    FileHandler,
    Logger,
    basicConfig,
    getLogger,
    StreamHandler,
    Formatter,
    DEBUG,
    INFO
)
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, fh_output_dir: str = "./", fh_output_name: str = "") -> Logger:
    """ロガーの初期設定を行う。同じ名前で二回目に呼ぶとハンドラーの設定はせずにロガーだけを返す

    Args:
        __name__ (str): ロガー名
        fh_output_name (str, optional): 出力先ファイル名. デフォルトは現在時刻.

    Returns:
        Logger: 
    """
    logger = getLogger(name)
    #
    if logger.hasHandlers():
        return logger
    logger.setLevel(DEBUG)
    format = Formatter('[%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d def:%(funcName)s] %(message)s')
    # コンソールへの出力
    sh = StreamHandler()
    sh.setLevel(INFO)
    sh.setFormatter(format)
    logger.addHandler(sh)

    # 出力先ファイルの設定
    fh_output_dir = Path('logs')
    Path.mkdir(fh_output_dir, exist_ok=True)
    if fh_output_name == "":
        fh_output_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fh = FileHandler((fh_output_dir/fh_output_name).with_suffix(".log"))
    fh.setLevel(DEBUG)
    fh.setFormatter(format)
    logger.addHandler(fh)

    logger.propagate = False
    return logger

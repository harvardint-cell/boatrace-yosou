"""GitHub Actions から呼ぶ当日ループ。usage: python scripts/run_live.py [max_minutes]"""
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from boatpred.live import Day, now_jst  # noqa: E402
from scripts.publish import publish  # noqa: E402


def main(max_minutes=340):
    if not (pathlib.Path(__file__).resolve().parent.parent / "models" / "params.json").exists():
        print("models/ がまだありません。先に train ワークフローを実行してください。")
        return
    start = now_jst()
    day = Day(start.date())
    day.run(until=start + dt.timedelta(minutes=int(max_minutes)), on_update=publish)


if __name__ == "__main__":
    main(*sys.argv[1:])

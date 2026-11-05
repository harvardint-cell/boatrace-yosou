"""夜間処理: 当日の成績・締切時オッズの取得、仮想収支の確定、翌日番組表の取得。
usage: python scripts/nightly.py [YYYY-MM-DD]"""
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from boatpred.download import fetch  # noqa: E402
from boatpred.live import SITE, now_jst, settle  # noqa: E402
from scripts.collect_odds import main as collect  # noqa: E402
from scripts.publish import publish  # noqa: E402


def main(day=None):
    # 成績(K)は翌日未明に公開されるため、既定は「昨日」
    d = dt.date.fromisoformat(day) if day else now_jst().date() - dt.timedelta(days=1)
    # 未確定の日があれば確定させる(直近7日)
    for i in range(7):
        x = d - dt.timedelta(days=i)
        if (SITE / "data" / f"{x:%Y%m%d}.json").exists():
            print(x, "settle", settle(x))
        collect(str(x), str(x), 6)
    fetch("B", d + dt.timedelta(days=1))
    publish()


if __name__ == "__main__":
    main(*sys.argv[1:])

"""公式(mbrace)の番組表(B)・競走成績(K)LZHを日付範囲で取得してテキスト化する。"""
import datetime as dt
import pathlib
import sys

import lhafile
import requests

BASE = "https://www1.mbrace.or.jp/od2/{kind}/{ym}/{k}{ymd}.lzh"
ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
UA = {"User-Agent": "boatpred/0.1 (personal research)"}


def path_for(kind: str, d: dt.date) -> pathlib.Path:
    return ROOT / kind / f"{kind}{d:%Y%m%d}.txt"


def fetch(kind: str, d: dt.date, session=None, force=False) -> pathlib.Path | None:
    """kind: 'K'(成績) or 'B'(番組表)。存在しない日(開催なし/未公開)はNone。"""
    out = path_for(kind, d)
    if out.exists() and not force:
        return out
    s = session or requests
    url = BASE.format(kind=kind, ym=f"{d:%Y%m}", k=kind.lower(), ymd=f"{d:%y%m%d}")
    r = s.get(url, headers=UA, timeout=30)
    if r.status_code != 200 or not r.content.startswith(b"\x00") and b"-lh" not in r.content[:10]:
        return None
    tmp = out.with_suffix(".lzh")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(r.content)
    try:
        lf = lhafile.Lhafile(str(tmp))
        text = "".join(lf.read(n).decode("cp932", errors="replace") for n in lf.namelist())
    finally:
        tmp.unlink(missing_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def daterange(a: dt.date, b: dt.date):
    while a <= b:
        yield a
        a += dt.timedelta(days=1)


def main(start: str, end: str, kinds="KB", workers=6):
    from concurrent.futures import ThreadPoolExecutor
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    jobs = [(k, d) for d in daterange(a, b) for k in kinds if not path_for(k, d).exists()]
    jobs.reverse()  # 新しい日付から取得
    s = requests.Session()

    def run(job):
        k, d = job
        try:
            return d, k, "ok" if fetch(k, d, s) else "none"
        except Exception as e:  # noqa: BLE001
            return d, k, f"ERR {e}"

    with ThreadPoolExecutor(int(workers)) as ex:
        for d, k, st in ex.map(run, jobs):
            print(d, k, st, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3], *(sys.argv[3:4] or ["KB"]))

"""番組表(B)・競走成績(K)テキストのパーサ。

固定長＋空白区切りが混在しているため、行の種類ごとに正規表現/位置で切り出す。
"""
import datetime as dt
import pathlib
import re
import unicodedata

import pandas as pd

NUM = r"-?\d+(?:\.\d+)?"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


# ---------------------------------------------------------------- 番組表(B)
B_RACE = re.compile(r"^\s*(\d{1,2})R\s+(.*?)\s+H(\d+)m(?:\s+電話投票締切予定(\d{1,2}):(\d{2}))?")
B_BOAT = re.compile(r"^([1-6]) (\d{4})")


def parse_b(path: pathlib.Path) -> pd.DataFrame:
    """1行=1艇。列: date,jcd,rno,race_name,distance,deadline,lane,toban,name,age,branch,weight,cls,
    nat_win,nat_2,loc_win,loc_2,motor_no,motor_2,boat_no,boat_2,series_results"""
    date = dt.datetime.strptime(path.stem[1:], "%Y%m%d").date()
    rows, jcd, race = [], None, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\r\n")
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd, race = int(m.group(1)), None
            continue
        if line.startswith("　") or line[:3].strip().endswith("Ｒ"):
            n = _nfkc(line)
            m = B_RACE.match(n)
            if m:
                race = dict(
                    rno=int(m.group(1)),
                    race_name=m.group(2).strip(),
                    distance=int(m.group(3)),
                    deadline=f"{int(m.group(4)):02d}:{m.group(5)}" if m.group(4) else None,
                )
                continue
        if race and jcd and B_BOAT.match(line) and len(line) > 30:
            head, rest = line[:18], line[18:]
            # 固定長: 全国勝率5 全国2率6 当地勝率5 当地2率6 ﾓｰﾀｰNO3 ﾓｰﾀｰ2率6 ﾎﾞｰﾄNO3 ﾎﾞｰﾄ2率6
            cuts = [0, 5, 11, 16, 22, 25, 31, 34, 40]
            v = [_f(rest[a:b]) for a, b in zip(cuts, cuts[1:])]
            rows.append(dict(
                date=date, jcd=jcd, **race,
                lane=int(head[0]), toban=int(head[2:6]), name=head[6:10].replace("　", ""),
                age=_f(head[10:12]), branch=head[12:14], weight=_f(head[14:16]), cls=head[16:18],
                nat_win=v[0], nat_2=v[1], loc_win=v[2], loc_2=v[3],
                motor_no=v[4], motor_2=v[5], boat_no=v[6], boat_2=v[7],
                series_results=rest[40:].rstrip(),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 競走成績(K)
K_RACE = re.compile(
    r"^\s+(\d{1,2})R\s+(.*?)\s+H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm"
)
K_BOAT = re.compile(r"^  (\S\S)  ([1-6]) (\d{4}) ")
K_TRI = re.compile(r"^\s+3連単\s+(\d-\d-\d)\s+(\d+)\s+人気\s+(\d+)")
K_TRI_ANY = re.compile(r"^\s+3連単\s+(.*)$")


def parse_k(path: pathlib.Path):
    """戻り値: (results, races, payouts)
    results: 1行=1艇 (finish=1..6 or NaN, status=原文着順, course, st, exhibit, motor_no, boat_no)
    races: 1行=1レース (天候・風・波)
    payouts: 3連単の的中組番・払戻(100円あたり)・人気
    """
    date = dt.datetime.strptime(path.stem[1:], "%Y%m%d").date()
    res, races, pays = [], [], []
    jcd, rno = None, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\r\n")
        m = re.match(r"^(\d{2})KBGN", line)
        if m:
            jcd, rno = int(m.group(1)), None
            continue
        if jcd is None:
            continue
        n = _nfkc(line)
        m = K_RACE.match(n)
        if m:
            rno = int(m.group(1))
            races.append(dict(
                date=date, jcd=jcd, rno=rno, race_name=m.group(2).strip(), distance=int(m.group(3)),
                weather=m.group(4), wind_dir=m.group(5), wind=int(m.group(6)), wave=int(m.group(7)),
            ))
            continue
        if rno is None:
            continue
        m = K_BOAT.match(line)
        if m:
            status = m.group(1)
            rest = line[21:].split()
            # motor boat exhibit course st racetime
            motor, boat = (_f(rest[0]), _f(rest[1])) if len(rest) >= 2 else (float("nan"),) * 2
            exhibit = _f(rest[2]) if len(rest) > 2 else float("nan")
            course = _f(rest[3]) if len(rest) > 3 else float("nan")
            st_raw = rest[4] if len(rest) > 4 else ""
            st_m = re.search(r"(\d*\.\d+)", st_raw)
            st = _f(st_m.group(1)) if st_m else float("nan")
            res.append(dict(
                date=date, jcd=jcd, rno=rno, lane=int(m.group(2)), toban=int(m.group(3)),
                status=status, finish=int(status) if status.isdigit() and 1 <= int(status) <= 6 else float("nan"),
                motor_no=motor, boat_no=boat, exhibit=exhibit, course=course, st=st,
                flying=st_raw.startswith("F"), late=st_raw.startswith("L"),
            ))
            continue
        m = K_TRI.match(n)
        if m:
            pays.append(dict(date=date, jcd=jcd, rno=rno, combo=m.group(1),
                             payout=int(m.group(2)), pop=int(m.group(3))))
            # 同着時は次行にも組番が続くが、MVPでは先頭のみ採用
    return pd.DataFrame(res), pd.DataFrame(races), pd.DataFrame(pays)

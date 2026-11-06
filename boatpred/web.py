"""boatrace.jp からオッズ・直前情報を取得する。

- odds3t: 3連単120通り(過去レースは締切時オッズ、当日は取得時点のオッズ)
- beforeinfo: 展示タイム・チルト・体重・スタート展示(進入/ST)・気象
"""
import datetime as dt
import itertools
import re
import time

import requests

BASE = "https://www.boatrace.jp/owpc/pc/race/{page}?rno={rno}&jcd={jcd:02d}&hd={hd}"
UA = {"User-Agent": "boatpred/0.1 (personal research)"}
_session = requests.Session()
_last = [0.0]
MIN_INTERVAL = 1.0  # 秒。サーバ負荷に配慮して最低1秒空ける


def _get(page: str, d: dt.date, jcd: int, rno: int) -> str:
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    url = BASE.format(page=page, rno=rno, jcd=jcd, hd=f"{d:%Y%m%d}")
    for i in range(4):
        try:
            r = _session.get(url, headers=UA, timeout=30)
            _last[0] = time.time()
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"fetch failed: {url}")


# 表の並び: 20行 × 6列(1着艇1..6)。各1着艇について2着-3着の辞書順
_PERMS = {f: [p for p in itertools.permutations([x for x in range(1, 7) if x != f], 2)] for f in range(1, 7)}
ORDER = [f"{f}-{_PERMS[f][r][0]}-{_PERMS[f][r][1]}" for r in range(20) for f in range(1, 7)]


def odds3t(d: dt.date, jcd: int, rno: int) -> dict:
    """{'1-2-3': 7.1, ...}。欠場等で数値でない組番は含めない。取得不可なら空dict。"""
    h = _get("odds3t", d, jcd, rno)
    cells = re.findall(r'class="oddsPoint[^"]*">([^<]*)<', h)
    if len(cells) != 120:
        return {}
    out = {}
    for combo, c in zip(ORDER, cells):
        try:
            out[combo] = float(c.replace(",", ""))
        except ValueError:
            pass
    m = re.search(r'class="tab4_time">([^<]*)<', h)
    out["_time"] = m.group(1).strip() if m else ""
    return out


def beforeinfo(d: dt.date, jcd: int, rno: int) -> dict:
    """{'boats': {lane: {...}}, 'weather': {...}}。展示前なら値はNone。"""
    h = re.sub(r"\s+", " ", _get("beforeinfo", d, jcd, rno))
    boats = {}
    for tb in re.findall(r'<tbody class="is-fs12 ">(.*?)</tbody>', h):
        m = re.search(r'is-boatColor(\d) is-fs14" rowspan="4">(\d)</td>', tb)
        if not m:
            continue
        lane = int(m.group(2))
        tds = re.findall(r'<td rowspan="(?:4|2)">([^<]*)</td>', tb)
        toban = re.search(r"toban=(\d{4})", tb)

        def num(s):
            m2 = re.search(r"-?\d+(?:\.\d+)?", s or "")
            return float(m2.group()) if m2 else None

        boats[lane] = dict(
            toban=int(toban.group(1)) if toban else None,
            weight=num(tds[0]) if len(tds) > 0 else None,
            exhibit=num(tds[1]) if len(tds) > 1 else None,
            tilt=num(tds[2]) if len(tds) > 2 else None,
        )
    # スタート展示: 表示順=進入コース順
    st = re.findall(
        r'table1_boatImage1Number is-type\d">(\d)</span>.*?table1_boatImage1Time[^>]*>([^<]*)<', h
    )
    for course, (lane, t) in enumerate(st, start=1):
        lane = int(lane)
        b = boats.setdefault(lane, {})
        b["ex_course"] = course
        b["ex_st_flying"] = t.strip().startswith("F")
        m = re.search(r"\.\d+", t)
        b["ex_st"] = float(m.group()) * (-1 if b["ex_st_flying"] else 1) if m else None

    def wx(title):
        m = re.search(title + r'</span> <span class="weather1_bodyUnitLabelData">([^<]*)<', h)
        if not m:
            return None
        m2 = re.search(r"-?\d+(?:\.\d+)?", m.group(1))
        return float(m2.group()) if m2 else None

    wd = re.search(r"is-wind(\d+)\"", h)
    wt = re.search(r'is-weather"> <p class="weather1_bodyUnitImage is-weather\d+"></p> <div class="weather1_bodyUnitLabel"> <span class="weather1_bodyUnitLabelTitle">([^<]*)<', h)
    weather = dict(
        air_temp=wx("気温"), wind=wx("風速"), water_temp=wx("水温"), wave=wx("波高"),
        wind_dir_code=int(wd.group(1)) if wd else None,
        weather=wt.group(1) if wt else None,
    )
    return dict(boats=boats, weather=weather)


def oddstf(d: dt.date, jcd: int, rno: int) -> dict:
    """単勝オッズ {艇番: odds}。0.0や欠場など数値として使えないものは含めない。"""
    h = _get("oddstf", d, jcd, rno)
    m = re.search(r"単勝オッズ</span>(.*?)複勝オッズ", h, re.S)
    if not m:
        return {}
    vals = re.findall(r'oddsPoint[^>]*>([^<]*)<', m.group(1))
    out = {}
    for lane, v in enumerate(vals[:6], start=1):
        try:
            f = float(v)
        except ValueError:
            continue
        if f >= 1.0:
            out[lane] = f
    return out

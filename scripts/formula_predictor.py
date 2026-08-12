# -*- coding: utf-8 -*-
"""
双色球 30 公式综合预测系统

基于用户提供的 30 个数学/物理/统计公式，对下一期双色球号码进行预测。
所有公式独立实现，最终加权集成输出推荐结果。

用法：
  python3 scripts/formula_predictor.py
  python3 scripts/formula_predictor.py --formula 5   # 仅运行第 5 号公式
  python3 scripts/formula_predictor.py --top-k 10   # 输出 Top 10 推荐组合
"""
import os
import sys
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import PROBABILITY_TXT, COMBO_ATTEND_TXT

HISTORY_PATH = os.path.join(PROJECT_ROOT, "ssq_history.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "双色球公式预测_{}.txt".format(
    datetime.now().strftime("%Y%m%d")
))

RED_RANGE = range(1, 34)
BLUE_RANGE = range(1, 17)
RED_COUNT = 6
TOTAL_RED_COMBOS = math.comb(33, 6)
TOTAL_COMBOS = TOTAL_RED_COMBOS * 16


class FormulaPredictor:
    def __init__(self, recent_window=100):
        self.recent_window = recent_window
        self.load_data()

    def load_data(self):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            self.history = json.load(f)
        self.history.sort(key=lambda x: x["issue"])
        for item in self.history:
            item["red_balls"] = [int(x) for x in item["red_balls"]]
            item["blue_ball"] = int(item["blue_ball"])
        self.total_periods = len(self.history)
        self.latest = self.history[-1]
        self.latest_issue = self.latest["issue"]

        self.recent = self.history[-self.recent_window:]
        self.red_freq_all = Counter()
        self.red_freq_recent = Counter()
        self.blue_freq_all = Counter()
        self.blue_freq_recent = Counter()
        for item in self.history:
            for r in item["red_balls"]:
                self.red_freq_all[r] += 1
            self.blue_freq_all[item["blue_ball"]] += 1
        for item in self.recent:
            for r in item["red_balls"]:
                self.red_freq_recent[r] += 1
            self.blue_freq_recent[item["blue_ball"]] += 1

        print(f"  历史期数: {self.total_periods}")
        print(f"  最近期号: {self.latest_issue} ({self.latest['date']})")
        print(f"  最近开奖: {' '.join(f'{r:02d}' for r in self.latest['red_balls'])} + {self.latest['blue_ball']:02d}")

    def red_prob(self, r, window=None):
        if window is None:
            return self.red_freq_all.get(r, 0) / self.total_periods
        return self.red_freq_recent.get(r, 0) / window

    def blue_prob(self, b, window=None):
        if window is None:
            return self.blue_freq_all.get(b, 0) / self.total_periods
        return self.blue_freq_recent.get(b, 0) / window

    def gap(self, num, ball_type="red"):
        if ball_type == "red":
            for i, item in enumerate(reversed(self.history)):
                if num in item["red_balls"]:
                    return i
        else:
            for i, item in enumerate(reversed(self.history)):
                if item["blue_ball"] == num:
                    return i
        return self.total_periods

    def red_gaps(self):
        return {r: self.gap(r, "red") for r in RED_RANGE}

    def blue_gaps(self):
        return {b: self.gap(b, "blue") for b in BLUE_RANGE}

    def _normalize(self, scores):
        total = sum(scores.values())
        if total <= 0:
            return {k: 1.0 / len(scores) for k in scores}
        return {k: v / total for k, v in scores.items()}

    def _uniform_blue(self):
        return {b: 1.0 / 16 for b in BLUE_RANGE}

    # ===================== 30 公式实现 =====================

    def formula_1(self):
        """组合投注总数 N = C(M,a)·C(N,b) → 均匀分布基准"""
        return {
            "id": 1, "name": "组合投注总数", "category": "统计基准", "meaning": "★",
            "red": {r: 1.0/33 for r in RED_RANGE},
            "blue": {b: 1.0/16 for b in BLUE_RANGE},
        }

    def formula_2(self):
        """分区选号组合叠加 → 三区高频区加权"""
        zones = [(1,11),(12,22),(23,33)]
        stats = []
        for lo, hi in zones:
            cnt = sum(sum(1 for r in item["red_balls"] if lo <= r <= hi)
                     for item in self.recent)
            avg = cnt / len(self.recent)
            exp = 6 * (hi - lo + 1) / 33
            stats.append((lo, hi, avg / exp if exp > 0 else 1))
        avg_ratio = np.mean([s[2] for s in stats])
        scores = {}
        for lo, hi, ratio in stats:
            w = ratio / avg_ratio
            for r in range(lo, hi+1):
                scores[r] = w
        return {
            "id": 2, "name": "分区选号组合", "category": "组合叠加", "meaning": "★",
            "red": self._normalize(scores), "blue": self._uniform_blue(),
            "detail": [(f"{lo}-{hi}", f"{ratio:.3f}") for lo, hi, ratio in stats],
        }

    def formula_3(self):
        """分层命中联合概率 → 多窗口联合"""
        windows = [10, 50, 100]
        p_red = {r: 1.0 for r in RED_RANGE}
        p_blue = {b: 1.0 for b in BLUE_RANGE}
        for w in windows:
            h = self.history[-w:]
            rc = Counter()
            bc = Counter()
            for it in h:
                for r in it["red_balls"]:
                    rc[r] += 1
                bc[it["blue_ball"]] += 1
            for r in RED_RANGE:
                p = rc.get(r, 0) / w
                p_red[r] *= p if p > 0 else 0.01
            for b in BLUE_RANGE:
                p = bc.get(b, 0) / w
                p_blue[b] *= p if p > 0 else 0.01
        return {
            "id": 3, "name": "分层命中联合概率", "category": "概率模型", "meaning": "★",
            "red": self._normalize(p_red), "blue": self._normalize(p_blue),
        }

    def formula_4(self):
        """长期购彩收益期望 → 负收益基准"""
        PRIZES = {"first": 5000000, "second": 200000, "third": 3000,
                   "fourth": 200, "fifth": 10, "sixth": 5}
        total = TOTAL_COMBOS
        exp = (1*PRIZES["first"] + 15*PRIZES["second"] + 162*PRIZES["third"]
               + 7695*PRIZES["fourth"] + 137475*PRIZES["fifth"]
               + 1043640*PRIZES["sixth"]) / total
        roi = (exp - 2) / 2 * 100
        return {
            "id": 4, "name": "长期购彩收益期望", "category": "收益分析", "meaning": "◇",
            "roi_percent": roi, "expected": exp, "note": f"回报率 {roi:.4f}%，长期必亏",
            "red": {r: 1.0/33 for r in RED_RANGE},
            "blue": {b: 1.0/16 for b in BLUE_RANGE},
        }

    def formula_5(self):
        """多维方差 → 位置稳定度"""
        pv = {}
        for r in RED_RANGE:
            pos = [sorted(it["red_balls"]).index(r) for it in self.history if r in it["red_balls"]]
            if len(pos) > 1:
                m = float(np.mean(pos))
                v = float(np.var(pos))
                denom = max(1e-6, m * m)
                pv[r] = v / denom
            else:
                pv[r] = 1.0
        mx = max(pv.values())
        scores = {r: max(0.01, 1.0 - pv[r] / mx) for r in RED_RANGE}
        bpv = {}
        for b in BLUE_RANGE:
            bs = [it["blue_ball"] for it in self.history if it["blue_ball"] == b]
            if len(bs) > 1:
                m = float(np.mean(bs))
                v = float(np.var(bs))
                denom = max(1e-6, m * m)
                bpv[b] = v / denom
            else:
                bpv[b] = 1.0
        mb = max(bpv.values())
        mb_safe = max(1e-9, float(mb))
        bscores = {b: max(0.01, 1.0 - bpv[b] / mb_safe) for b in BLUE_RANGE}
        return {
            "id": 5, "name": "开奖号码多维方差", "category": "统计量", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_6(self):
        """冷热演化动力学 df/dt = α(1-f) - βf → 回补预测"""
        alpha, beta = 0.05, 0.03
        heat = {}
        for r in RED_RANGE:
            cf = self.red_freq_recent.get(r,0)/self.recent_window
            tr = 6/33
            hr = cf/tr
            drift = alpha*(1-hr) - beta*hr
            heat[r] = hr + drift*10
        mn, mx = min(heat.values()), max(heat.values())
        scores = {r: (v-mn+0.01)/(mx-mn+0.01) for r,v in heat.items()}
        bheat = {}
        for b in BLUE_RANGE:
            cf = self.blue_freq_recent.get(b,0)/self.recent_window
            tr = 1/16
            hr = cf/tr
            drift = alpha*(1-hr) - beta*hr
            bheat[b] = hr + drift*10
        mn, mx = min(bheat.values()), max(bheat.values())
        bscores = {b: (v-mn+0.01)/(mx-mn+0.01) for b,v in bheat.items()}
        return {
            "id": 6, "name": "冷热号码演化动力学", "category": "动力学", "meaning": "★",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_7(self):
        """布朗运动 Xt = X0 + μt + σBt → 漂移预测"""
        scores = {}
        for r in RED_RANGE:
            fnow = self.red_freq_recent.get(r,0)/self.recent_window
            fprev = sum(1 for it in self.history[-200:-100] if r in it["red_balls"])/100
            mu = fnow - fprev
            sig = np.std([1 if r in it["red_balls"] else 0 for it in self.history[-50:]])
            scores[r] = max(0.001, fnow + mu*5 + sig*0)
        bscores = {}
        for b in BLUE_RANGE:
            fnow = self.blue_freq_recent.get(b,0)/self.recent_window
            sig = np.std([1 if it["blue_ball"]==b else 0 for it in self.history[-50:]])
            bscores[b] = max(0.001, fnow + sig*0)
        return {
            "id": 7, "name": "随机开奖布朗运动", "category": "随机过程", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_8(self):
        """遗漏衰减 L(t) = L0·e^(-λt) + Lavg → 回补预测"""
        g = self.red_gaps()
        bg = self.blue_gaps()
        avg_rg = 33/6
        avg_bg = 16
        lam = 0.1
        scores = {}
        for r in RED_RANGE:
            eg = g[r]*math.exp(-lam) + avg_rg
            scores[r] = min(1.0, max(0.01, eg/(avg_rg*5)))
        bscores = {}
        for b in BLUE_RANGE:
            eg = bg[b]*math.exp(-lam) + avg_bg
            bscores[b] = min(1.0, max(0.01, eg/(avg_bg*3)))
        return {
            "id": 8, "name": "号码遗漏值衰减回归", "category": "回补预测", "meaning": "★",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_9(self):
        """奇偶大小均衡偏离度 → 形态回归"""
        avg_o, avg_b = 3, 3
        ro = [sum(1 for r in it["red_balls"] if r%2==1) for it in self.recent]
        rb = [sum(1 for r in it["red_balls"] if r>=17) for it in self.recent]
        ca_o, ca_b = np.mean(ro), np.mean(rb)
        dev = math.sqrt((ca_o-avg_o)**2 + (ca_b-avg_b)**2)
        scores = {}
        for r in RED_RANGE:
            io, ib = r%2==1, r>=17
            ow = 1 + dev*0.3*(-1 if ca_o>avg_o else 1)*(1 if not io else -1)
            bw = 1 + dev*0.3*(-1 if ca_b>avg_b else 1)*(1 if not ib else -1)
            scores[r] = max(0.01, 0.5*ow + 0.5*bw)
        bo = np.mean([1 if it["blue_ball"]%2==1 else 0 for it in self.recent])
        bscores = {}
        for b in BLUE_RANGE:
            io = b%2==1
            if bo > 0.5:
                bscores[b] = 0.8 if not io else 1.0
            else:
                bscores[b] = 1.0 if not io else 0.8
        return {
            "id": 9, "name": "奇偶大小均衡偏离度", "category": "形态分析", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
            "deviation": dev,
        }

    def formula_10(self):
        """胆拖组合 → 胆码加权"""
        tr = sorted(RED_RANGE, key=lambda r: self.red_freq_recent.get(r,0), reverse=True)
        tb = sorted(BLUE_RANGE, key=lambda b: self.blue_freq_recent.get(b,0), reverse=True)
        dan_r = set(tr[:5])
        dan_b = set(tb[:2])
        scores = {r: 1.5 if r in dan_r else 0.5 for r in RED_RANGE}
        bscores = {b: 2.0 if b in dan_b else 1.0 for b in BLUE_RANGE}
        return {
            "id": 10, "name": "复式胆拖混合投注", "category": "投注策略", "meaning": "◆",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
            "dan_reds": sorted(dan_r), "dan_blues": sorted(dan_b),
        }

    def formula_11(self):
        """信息熵 H = -∑p·log₂p → 熵低号回补"""
        rp = [self.red_prob(r) for r in RED_RANGE]
        re = -sum(p*math.log2(p) for p in rp if p>0)
        mre = math.log2(33)
        bp = [self.blue_prob(b) for b in BLUE_RANGE]
        be = -sum(p*math.log2(p) for p in bp if p>0)
        mbe = math.log2(16)
        er = re/mre
        scores = {}
        for r in RED_RANGE:
            g = self.gap(r, "red")
            scores[r] = (0.5 + g/50) * (1 - er*0.5)
        eb = be/mbe
        bscores = {}
        for b in BLUE_RANGE:
            g = self.gap(b, "blue")
            bscores[b] = (0.5 + g/30) * (1 - eb*0.5)
        return {
            "id": 11, "name": "开奖系统信息熵", "category": "信息论", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
            "entropy_red": re, "entropy_blue": be,
        }

    def formula_12(self):
        """贝叶斯更新 → 后验概率"""
        pr = {r: 1/33 for r in RED_RANGE}
        pb = {b: 1/16 for b in BLUE_RANGE}
        lr = {r: min(1, max(0.01, self.red_prob(r)*33)) for r in RED_RANGE}
        lb = {b: min(1, max(0.01, self.blue_prob(b)*16)) for b in BLUE_RANGE}
        scores = {}
        for r in RED_RANGE:
            pyx = lr[r]
            pynx = 1 - pyx
            post = (pr[r]*pyx) / (pyx*pr[r] + pynx*(1-pr[r]))
            scores[r] = post
        bscores = {}
        for b in BLUE_RANGE:
            pyx = lb[b]
            pynx = 1 - pyx
            post = (pb[b]*pyx) / (pyx*pb[b] + pynx*(1-pb[b]))
            bscores[b] = post
        return {
            "id": 12, "name": "贝叶斯走势概率迭代", "category": "贝叶斯", "meaning": "★",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_13(self):
        """阻尼振动 S''+2γS'+ω²S=A₀sin(ωt) → 预测和值"""
        sums = [sum(it["red_balls"]) for it in self.history[-50:]]
        ms = np.mean(sums)
        scores = {}
        for r in RED_RANGE:
            dev = abs(r - ms/6) / 33
            scores[r] = max(0.01, 1 - dev)
        bscores = {}
        bm = np.mean([it["blue_ball"] for it in self.history[-50:]])
        for b in BLUE_RANGE:
            dev = abs(b - bm) / 16
            bscores[b] = max(0.01, 1 - dev)
        return {
            "id": 13, "name": "号码和值阻尼振动", "category": "动力学", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
            "predicted_sum": ms,
        }

    def formula_14(self):
        """多期未出 P_lack = ∏(1-pk) → 回补概率"""
        pr, pb = 6/33, 1/16
        scores = {}
        for r in RED_RANGE:
            g = self.gap(r, "red")
            pl = (1-pr)**g
            scores[r] = max(0.01, 1 - pl)
        bscores = {}
        for b in BLUE_RANGE:
            g = self.gap(b, "blue")
            pl = (1-pb)**g
            bscores[b] = max(0.01, 1 - pl)
        return {
            "id": 14, "name": "多期未出独立事件", "category": "概率模型", "meaning": "★",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_15(self):
        """大数定律 → 频率收敛回归"""
        tr, tb = 6/33, 1/16
        scores = {}
        for r in RED_RANGE:
            ob = self.red_prob(r)
            dev = ob - tr
            scores[r] = 1/(1+abs(dev)*10) if dev>0 else 1+abs(dev)*10
        bscores = {}
        for b in BLUE_RANGE:
            ob = self.blue_prob(b)
            dev = ob - tb
            bscores[b] = 1/(1+abs(dev)*20) if dev>0 else 1+abs(dev)*20
        return {
            "id": 15, "name": "大数定律频率收敛", "category": "统计规律", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_16(self):
        """加权预判 P_pre = ∑e^(-αi)·fi → 时间衰减"""
        alpha = 0.05
        rw = {r: 0.0 for r in RED_RANGE}
        bw = {b: 0.0 for b in BLUE_RANGE}
        tw = 0.0
        for i, it in enumerate(reversed(self.history)):
            w = math.exp(-alpha*i)
            tw += w
            for r in it["red_balls"]:
                rw[r] += w
            bw[it["blue_ball"]] += w
        scores = {r: rw[r]/tw for r in RED_RANGE}
        bscores = {b: bw[b]/tw for b in BLUE_RANGE}
        return {
            "id": 16, "name": "远近历史数据加权预判", "category": "时间衰减", "meaning": "★",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_17(self):
        """引力模型 F = K·(n₁n₂/r²)·e^(-μr) → 共现引力"""
        K, mu = 1.0, 0.1
        co = defaultdict(int)
        for it in self.history[-100:]:
            for r1, r2 in combinations(it["red_balls"], 2):
                co[(min(r1,r2), max(r1,r2))] += 1
        lr = set(self.latest["red_balls"])
        scores = {}
        for r in RED_RANGE:
            if r in lr:
                scores[r] = 1.5
            else:
                tf = sum(K*co.get((min(r,l),max(r,l)),0)/(abs(r-l)+1)**2*math.exp(-mu*abs(r-l))
                         for l in lr)
                scores[r] = 0.5 + tf
        bscores = {}
        for b in BLUE_RANGE:
            d = abs(b - self.latest["blue_ball"]) + 1
            bscores[b] = 1.0/d
        return {
            "id": 17, "name": "号码空间引力模型", "category": "物理模型", "meaning": "◆",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_18(self):
        """资金风控 → 夏普比率选号"""
        scores = {}
        for r in RED_RANGE:
            f = self.red_prob(r)
            v = f*(1-f)
            scores[r] = f/max(0.001, math.sqrt(v))
        bscores = {}
        for b in BLUE_RANGE:
            f = self.blue_prob(b)
            v = f*(1-f)
            bscores[b] = f/max(0.001, math.sqrt(v))
        return {
            "id": 18, "name": "购彩资金仓位风控", "category": "风险控制", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_19(self):
        """信号衰减 I = I₀·exp(-μd) → 近距离选号"""
        mu = 0.02
        lr = self.latest["red_balls"]
        scores = {}
        for r in RED_RANGE:
            md = min(abs(r-l) for l in lr)
            scores[r] = 0.3 + math.exp(-mu*md)*0.7
        bscores = {}
        d = {b: abs(b - self.latest["blue_ball"]) for b in BLUE_RANGE}
        bscores = {b: math.exp(-mu*d[b]) for b in BLUE_RANGE}
        return {
            "id": 19, "name": "开奖信号大气衰减", "category": "物理模型", "meaning": "◆",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_20(self):
        """最小二乘拟合 → 趋势预测"""
        scores = {}
        for r in RED_RANGE:
            y = np.array([1 if r in it["red_balls"] else 0 for it in self.history[-200:]])
            x = np.arange(len(y))
            cf = np.polyfit(x, y, 2)
            nx = len(y)
            p = cf[0]*nx**2 + cf[1]*nx + cf[2]
            scores[r] = max(0.01, min(1.0, p))
        bscores = {}
        for b in BLUE_RANGE:
            y = np.array([1 if it["blue_ball"]==b else 0 for it in self.history[-200:]])
            x = np.arange(len(y))
            cf = np.polyfit(x, y, 2)
            nx = len(y)
            p = cf[0]*nx**2 + cf[1]*nx + cf[2]
            bscores[b] = max(0.01, min(1.0, p))
        return {
            "id": 20, "name": "形态走势最小二乘拟合", "category": "回归分析", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_21(self):
        """皮尔逊相关 → 自相关预测"""
        scores = {}
        for r in RED_RANGE:
            xs = [1 if r in self.history[i-1]["red_balls"] else 0
                  for i in range(1, len(self.history))]
            ys = [1 if r in self.history[i]["red_balls"] else 0
                  for i in range(1, len(self.history))]
            if len(xs) > 10:
                xa, ya = np.array(xs, float), np.array(ys, float)
                if np.std(xa) > 0.01 and np.std(ya) > 0.01:
                    corr = np.corrcoef(xa, ya)[0,1]
                else:
                    corr = 0
                scores[r] = 0.5 + corr*0.5
            else:
                scores[r] = 0.5
        bscores = {}
        for b in BLUE_RANGE:
            xs = [1 if self.history[i-1]["blue_ball"]==b else 0
                  for i in range(1, len(self.history))]
            ys = [1 if self.history[i]["blue_ball"]==b else 0
                  for i in range(1, len(self.history))]
            if len(xs) > 10:
                xa, ya = np.array(xs, float), np.array(ys, float)
                if np.std(xa) > 0.01 and np.std(ya) > 0.01:
                    corr = np.corrcoef(xa, ya)[0,1]
                else:
                    corr = 0
                bscores[b] = 0.5 + corr*0.5
            else:
                bscores[b] = 0.5
        return {
            "id": 21, "name": "皮尔逊相关系数", "category": "相关分析", "meaning": "◆",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_22(self):
        """概率加法 → 近期出现概率"""
        K = 10
        scores = {}
        for r in RED_RANGE:
            pa = sum(1 for it in self.history[-K:] if r in it["red_balls"]) / K
            if pa == 0:
                ps = self.red_prob(r)
                scores[r] = 1 - (1-ps)**(K+1)
            else:
                scores[r] = pa
        bscores = {}
        for b in BLUE_RANGE:
            pa = sum(1 for it in self.history[-K:] if it["blue_ball"]==b) / K
            if pa == 0:
                ps = self.blue_prob(b)
                bscores[b] = 1 - (1-ps)**(K+1)
            else:
                bscores[b] = pa
        return {
            "id": 22, "name": "概率加法公式", "category": "概率模型", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_23(self):
        """稳态平衡 ∇²P-(1/τ²)P+S=0 → 稳态分布"""
        scores = {}
        for r in RED_RANGE:
            ss = self.red_prob(r)
            recent = self.red_freq_recent.get(r,0)/self.recent_window
            scores[r] = max(0.01, ss + (recent-ss)*0.3)
        bscores = {}
        for b in BLUE_RANGE:
            ss = self.blue_prob(b)
            recent = self.blue_freq_recent.get(b,0)/self.recent_window
            bscores[b] = max(0.01, ss + (recent-ss)*0.3)
        return {
            "id": 23, "name": "随机系统稳态平衡", "category": "稳态分析", "meaning": "◆",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_24(self):
        """相位同步 φ(t)=φ₀+∫ω(τ)dτ → 相位共振"""
        scores = {}
        for r in RED_RANGE:
            phase = 2*math.pi*r/33
            omega = self.red_freq_recent.get(r,0)/self.recent_window * 2*math.pi
            np_ = phase + omega
            res = 1 - abs(math.sin(np_))
            scores[r] = 0.3 + res*0.7
        bscores = {}
        for b in BLUE_RANGE:
            phase = 2*math.pi*b/16
            omega = self.blue_freq_recent.get(b,0)/self.recent_window * 2*math.pi
            np_ = phase + omega
            res = 1 - abs(math.sin(np_))
            bscores[b] = 0.3 + res*0.7
        return {
            "id": 24, "name": "号码相位周期同步", "category": "波动分析", "meaning": "◆",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_25(self):
        """AR(p) 自回归 → 预测频率"""
        scores = {}
        for r in RED_RANGE:
            fs = [1 if r in it["red_balls"] else 0 for it in self.history[-100:]]
            if len(fs) > 2:
                x = np.array(fs[:-1], float)
                y = np.array(fs[1:], float)
                if np.std(x) > 0.01:
                    phi = np.corrcoef(x,y)[0,1]*np.std(y)/np.std(x)
                    c = np.mean(y) - phi*np.mean(x)
                    p = c + phi*fs[-1]
                else:
                    p = np.mean(fs)
                scores[r] = max(0.01, min(1.0, p))
            else:
                scores[r] = 0.5
        bscores = {}
        for b in BLUE_RANGE:
            fs = [1 if it["blue_ball"]==b else 0 for it in self.history[-100:]]
            if len(fs) > 2:
                x = np.array(fs[:-1], float)
                y = np.array(fs[1:], float)
                if np.std(x) > 0.01:
                    phi = np.corrcoef(x,y)[0,1]*np.std(y)/np.std(x)
                    c = np.mean(y) - phi*np.mean(x)
                    p = c + phi*fs[-1]
                else:
                    p = np.mean(fs)
                bscores[b] = max(0.01, min(1.0, p))
            else:
                bscores[b] = 0.5
        return {
            "id": 25, "name": "自回归模型AR", "category": "时间序列", "meaning": "◆",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_26(self):
        """均线交叉 → 短长均线信号"""
        sw, lw = 10, 50
        scores = {}
        for r in RED_RANGE:
            fs = [1 if r in it["red_balls"] else 0 for it in self.history[-lw*2:]]
            if len(fs) >= lw+sw:
                mas = np.mean(fs[-sw:])
                mal = np.mean(fs[-lw:])
                ratio = mas/max(0.001, mal)
                scores[r] = min(1.0, max(0.01, 0.5*ratio))
            else:
                scores[r] = 0.5
        bscores = {}
        for b in BLUE_RANGE:
            fs = [1 if it["blue_ball"]==b else 0 for it in self.history[-lw*2:]]
            if len(fs) >= lw+sw:
                mas = np.mean(fs[-sw:])
                mal = np.mean(fs[-lw:])
                ratio = mas/max(0.001, mal)
                bscores[b] = min(1.0, max(0.01, 0.5*ratio))
            else:
                bscores[b] = 0.5
        return {
            "id": 26, "name": "均线交叉预测", "category": "技术指标", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_27(self):
        """马尔可夫链 P(Xt+1|Xt) → 转移概率"""
        tr = {i: Counter() for i in RED_RANGE}
        for idx in range(1, len(self.history)):
            pr = self.history[idx-1]["red_balls"]
            cr = self.history[idx]["red_balls"]
            for i in pr:
                for j in cr:
                    tr[i][j] += 1
        for i in RED_RANGE:
            t = sum(tr[i].values())
            if t > 0:
                for j in tr[i]:
                    tr[i][j] /= t
        lr = set(self.latest["red_balls"])
        rc = Counter()
        for lr_ in lr:
            for j, p in tr[lr_].items():
                rc[j] += p
        ts = sum(rc.values())
        scores = {r: rc.get(r,0)/max(0.001,ts) for r in RED_RANGE}
        # 蓝球
        tb = Counter()
        for idx in range(1, len(self.history)):
            if self.history[idx-1]["blue_ball"] == self.latest["blue_ball"]:
                tb[self.history[idx]["blue_ball"]] += 1
        tb_total = sum(tb.values())
        if tb_total > 0:
            bscores = {b: tb.get(b,0)/tb_total for b in BLUE_RANGE}
        else:
            bscores = self._uniform_blue()
        return {
            "id": 27, "name": "马尔可夫链转移", "category": "随机过程", "meaning": "★",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_28(self):
        """傅里叶频谱 F(ω)=∑x(t)e^(-iωt) → 周期预测"""
        scores = {}
        for r in RED_RANGE:
            s = np.array([1 if r in it["red_balls"] else 0 for it in self.history[-200:]])
            spec = np.abs(np.fft.fft(s))
            df = np.argmax(spec[1:len(spec)//2]) + 1
            pw = spec[df]/np.mean(spec)
            scores[r] = min(1.0, max(0.01, pw/10))
        bscores = {}
        for b in BLUE_RANGE:
            s = np.array([1 if it["blue_ball"]==b else 0 for it in self.history[-200:]])
            spec = np.abs(np.fft.fft(s))
            df = np.argmax(spec[1:len(spec)//2]) + 1
            pw = spec[df]/np.mean(spec)
            bscores[b] = min(1.0, max(0.01, pw/10))
        return {
            "id": 28, "name": "傅里叶频谱分析", "category": "信号处理", "meaning": "◇",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
        }

    def formula_29(self):
        """马尔可夫切换 → 状态依赖预测"""
        ra = np.mean([len(it["red_balls"]) for it in self.history[-10:]])
        hot = ra >= 6
        scores = {}
        for r in RED_RANGE:
            f = self.red_prob(r)
            scores[r] = f*2 if hot else f + self.gap(r,"red")/33
        bscores = {}
        for b in BLUE_RANGE:
            f = self.blue_prob(b)
            bscores[b] = f*2 if hot else f + self.gap(b,"blue")/16
        return {
            "id": 29, "name": "马尔可夫切换模型", "category": "状态模型", "meaning": "◆",
            "red": self._normalize(scores), "blue": self._normalize(bscores),
            "is_hot_state": hot,
        }

    def formula_30(self):
        """自适应集成权重 → 加权融合"""
        weights = {
            1:1.0, 2:1.2, 3:1.3, 4:0.5, 5:0.8, 6:1.5, 7:0.7, 8:1.4,
            9:0.9, 10:1.1, 11:0.8, 12:1.6, 13:0.7, 14:1.3, 15:0.9,
            16:1.5, 17:0.6, 18:0.8, 19:0.6, 20:0.8, 21:0.9, 22:0.9,
            23:0.6, 24:0.6, 25:1.2, 26:1.0, 27:1.3, 28:0.7, 29:0.6, 30:1.0,
        }
        return {"weights": weights}

    # ===================== 集成预测 =====================

    def run_all_formulas(self, formula_ids=None):
        """运行所有或指定公式"""
        if formula_ids is None:
            formula_ids = list(range(1, 31))
        results = []
        for fid in formula_ids:
            fn = getattr(self, f"formula_{fid}", None)
            if fn:
                try:
                    res = fn()
                    results.append(res)
                except Exception as e:
                    print(f"  ⚠ 公式 {fid} 运行失败: {e}")
        return results

    def get_fused_probabilities(self):
        """运行全部30公式并返回融合概率，供外部调用（不打印不保存）

        Returns:
            fused_red: dict {1..33: float} 红球归一化概率
            fused_blue: dict {1..16: float} 蓝球归一化概率
            results: list[dict] 各公式详细结果
        """
        results = self.run_all_formulas()
        fused_red, fused_blue = self.ensemble(results)
        return fused_red, fused_blue, results

    def ensemble(self, results):
        """加权集成所有公式的预测结果"""
        wf = self.formula_30()["weights"]
        # 加权融合
        fused_red = {r: 0.0 for r in RED_RANGE}
        fused_blue = {b: 0.0 for b in BLUE_RANGE}
        tw = 0.0
        for res in results:
            if "id" not in res or "red" not in res or "blue" not in res:
                continue
            w = wf.get(res["id"], 1.0)
            tw += w
            for r in RED_RANGE:
                fused_red[r] += res["red"][r] * w
            for b in BLUE_RANGE:
                fused_blue[b] += res["blue"][b] * w
        fused_red = {r: v/tw for r, v in fused_red.items()} if tw > 0 else {r: 1.0/33 for r in RED_RANGE}
        fused_blue = {b: v/tw for b, v in fused_blue.items()} if tw > 0 else {b: 1.0/16 for b in BLUE_RANGE}
        return fused_red, fused_blue

    def generate_recommendations(self, fused_red, fused_blue, top_k=20):
        """生成 Top K 推荐组合"""
        # 按概率排序红球和蓝球
        red_ranked = sorted(RED_RANGE, key=lambda r: fused_red[r], reverse=True)
        blue_ranked = sorted(BLUE_RANGE, key=lambda b: fused_blue[b], reverse=True)

        # 从 Top 12 红球和 Top 8 蓝球中生成组合
        red_pool = red_ranked[:12]
        blue_pool = blue_ranked[:8]

        combos = []
        for red_combo in combinations(red_pool, 6):
            for blue in blue_pool:
                prob = sum(fused_red[r] for r in red_combo) * fused_blue[blue]
                combos.append((sorted(red_combo), blue, prob))
        combos.sort(key=lambda x: -x[2])
        return combos[:top_k], red_ranked, blue_ranked

    # ===================== 报告输出 =====================

    def generate_report(self, results, fused_red, fused_blue, top_combos,
                        red_ranked, blue_ranked, single_formula_id=None):
        lines = []
        lines.append("=" * 80)
        lines.append("         双色球 30 公式综合预测系统报告")
        lines.append("=" * 80)
        lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  历史期数: {self.total_periods}")
        lines.append(f"  分析期号: {self.latest_issue} ({self.latest['date']})")
        lines.append(f"  最近开奖: {' '.join(f'{r:02d}' for r in self.latest['red_balls'])} + {self.latest['blue_ball']:02d}")
        lines.append("")

        if single_formula_id:
            lines.append(f"  仅运行公式 #{single_formula_id}")
        else:
            lines.append(f"  运行公式数: {len(results)}/30")

        lines.append("")

        # 各公式结果摘要
        lines.append("─" * 80)
        lines.append("【各公式预测摘要】")
        lines.append("─" * 80)
        lines.append(f"  {'#':>3}  {'公式名称':<20}  {'类别':<10}  {'★':>3}  {'红球Top3':<18}  {'蓝球Top3':<12}")
        lines.append(f"  {'─'*3}  {'─'*20}  {'─'*10}  {'─'*3}  {'─'*18}  {'─'*12}")
        for res in results:
            if "red" not in res or "blue" not in res:
                continue
            tr = sorted(res["red"].items(), key=lambda x: -x[1])[:3]
            tb = sorted(res["blue"].items(), key=lambda x: -x[1])[:3]
            tr_str = " ".join(f"{r:02d}({p:.3f})" for r, p in tr)
            tb_str = " ".join(f"{b:02d}({p:.3f})" for b, p in tb)
            lines.append(f"  {res['id']:>3}  {res['name']:<20}  {res['category']:<10}  {res['meaning']:>3}  {tr_str:<18}  {tb_str:<12}")

        lines.append("")

        # 集成结果
        if not single_formula_id:
            lines.append("─" * 80)
            lines.append("【加权集成预测结果】")
            lines.append("─" * 80)
            lines.append("")

        # 红球综合概率
        lines.append(f"  红球综合概率（加权集成）:")
        lines.append(f"  {'号码':>4}  {'概率':>8}  {'历史频率':>10}  {'冷热':>6}  {'遗漏':>6}")
        lines.append(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*6}")
        for r in RED_RANGE:
            p = fused_red[r]
            hp = self.red_prob(r)
            g = self.gap(r, "red")
            status = "热" if hp >= 0.212 else ("冷" if hp < 0.15 else "温")
            lines.append(f"  {r:02d}   {p:8.5f}  {hp:10.5f}  {status:>6}  {g:3d}期")
        lines.append("")

        # 蓝球综合概率
        lines.append(f"  蓝球综合概率（加权集成）:")
        lines.append(f"  {'号码':>4}  {'概率':>8}  {'历史频率':>10}  {'遗漏':>6}")
        lines.append(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*6}")
        for b in BLUE_RANGE:
            p = fused_blue[b]
            hp = self.blue_prob(b)
            g = self.gap(b, "blue")
            lines.append(f"  {b:02d}   {p:8.5f}  {hp:10.5f}  {g:3d}期")
        lines.append("")

        # Top K 推荐组合
        lines.append("─" * 80)
        lines.append(f"【Top {len(top_combos)} 推荐组合】（基于集成概率）")
        lines.append("─" * 80)
        lines.append("")
        for rank, (reds, blue, prob) in enumerate(top_combos, 1):
            red_str = " ".join(f"{r:02d}" for r in reds)
            lines.append(f"  #{rank:03d}  {red_str} + {blue:02d}  综合概率: {prob:.6f}")
        lines.append("")

        # 说明
        lines.append("═" * 80)
        lines.append("  说明:")
        lines.append("  ★ = 统计上有实际参考意义（回补预测/贝叶斯/时间衰减等）")
        lines.append("  ◇ = 物理模型类比（适用彩票但统计意义有限）")
        lines.append("  ◆ = 物理模型套用（仅作数值演示，不具备统计学意义）")
        lines.append("  ⚠️ 双色球开奖为独立随机事件，任何预测公式都无法保证中奖！")
        lines.append("     本系统仅作数学分析与娱乐参考，请理性购彩。")
        lines.append("═" * 80)

        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="双色球 30 公式综合预测")
    parser.add_argument("--formula", type=int, help="仅运行指定公式编号(1-30)")
    parser.add_argument("--top-k", type=int, default=20, help="输出 Top K 推荐组合")
    parser.add_argument("--no-save", action="store_true", help="仅打印不保存文件")
    args = parser.parse_args()

    predictor = FormulaPredictor(recent_window=100)

    if args.formula:
        print(f"\n仅运行公式 #{args.formula}...")
        results = predictor.run_all_formulas([args.formula])
        fused_red = results[0]["red"]
        fused_blue = results[0]["blue"]
    else:
        print(f"\n运行全部 30 个公式...")
        results = predictor.run_all_formulas()
        print(f"  成功运行 {len(results)} 个公式")
        fused_red, fused_blue = predictor.ensemble(results)

    top_combos, red_ranked, blue_ranked = predictor.generate_recommendations(
        fused_red, fused_blue, top_k=args.top_k
    )

    report = predictor.generate_report(
        results, fused_red, fused_blue, top_combos,
        red_ranked, blue_ranked, args.formula
    )

    print("\n" + report)

    if not args.no_save:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n报告已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBVS 精度统计分析工具

功能:
  读取 experiment_log.json，按 method (ibvs/pbvs) 分组，
  计算各指标的均值和标准差，输出表格。

用法:
  python3 analyze_error.py                        # 分析默认日志
  python3 analyze_error.py --log my_log.json      # 分析指定文件
  python3 analyze_error.py --tag "pos1"           # 只分析特定标签
"""

import json
import os
import argparse
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(SCRIPT_DIR, 'experiment_log.json')


def load_log(log_path):
    if not os.path.exists(log_path):
        print(f"❌ 未找到 {log_path}")
        exit(1)
    with open(log_path) as f:
        return json.load(f)


def analyze(records, group_key='method'):
    """按 group_key 分组统计"""
    groups = defaultdict(list)
    for r in records:
        key = r.get(group_key, 'unknown')
        groups[key].append(r)

    metrics = [
        ('err_u',              'XY-U 误差 (px)',         'px'),
        ('err_v',              'XY-V 误差 (px)',         'px'),
        ('err_xy',             'XY 欧氏距离 (px)',       'px'),
        ('err_area_ratio_pct', '面积比例误差 (%)',       '%'),
        ('err_depth_m',        '深度误差 (m)',           'm'),
    ]

    print("\n" + "=" * 72)
    print(f"  实验样本统计  (按 {group_key} 分组)")
    print("=" * 72)

    for group_name, group_records in sorted(groups.items()):
        print(f"\n  ▶ {group_name.upper()}  (共 {len(group_records)} 条记录)")
        print(f"  {'─' * 60}")
        print(f"  {'指标':<25s}  {'均值':>10s}  {'标准差':>10s}  "
              f"{'最小值':>10s}  {'最大值':>10s}")
        print(f"  {'─' * 60}")

        for metric_key, metric_name, unit in metrics:
            values = []
            for r in group_records:
                v = r.get('errors', {}).get(metric_key)
                if v is not None:
                    values.append(abs(v))  # 取绝对值统计

            if len(values) == 0:
                continue

            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            vmin = np.min(arr)
            vmax = np.max(arr)

            print(f"  {metric_name:<25s}  {mean:>9.2f}{unit}  {std:>9.2f}{unit}  "
                  f"{vmin:>9.2f}{unit}  {vmax:>9.2f}{unit}")

        print(f"  {'─' * 60}")

    # 输出论文表格格式
    print("\n" + "=" * 72)
    print("  论文表格格式输出 (表4.1)")
    print("=" * 72)

    all_group_names = sorted(groups.keys())
    header = f"  {'评估指标':<30s}"
    for gn in all_group_names:
        header += f"  {gn.upper():>15s}"
    print(header)
    print(f"  {'─' * (30 + 17 * len(all_group_names))}")

    table_metrics = [
        ('err_xy',             'XY视觉误差均值 (px)',     'mean'),
        ('err_xy',             'XY视觉误差标准差 (px)',   'std'),
        ('err_area_ratio_pct', 'Z轴面积比例误差均值 (%)', 'mean'),
        ('err_area_ratio_pct', 'Z轴面积比例误差标准差 (%)', 'std'),
    ]

    for metric_key, label, stat_type in table_metrics:
        row = f"  {label:<30s}"
        for gn in all_group_names:
            values = []
            for r in groups[gn]:
                v = r.get('errors', {}).get(metric_key)
                if v is not None:
                    values.append(abs(v))
            if values:
                arr = np.array(values)
                if stat_type == 'mean':
                    val = np.mean(arr)
                else:
                    val = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
                row += f"  {val:>15.2f}"
            else:
                row += f"  {'N/A':>15s}"
        print(row)

    print(f"  {'─' * (30 + 17 * len(all_group_names))}")
    print()


def main():
    parser = argparse.ArgumentParser(description="IBVS 精度统计分析")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG,
                        help="实验日志路径 (默认: experiment_log.json)")
    parser.add_argument("--tag", type=str, default=None,
                        help="只分析特定标签的记录")
    parser.add_argument("--group-by", type=str, default="method",
                        choices=["method", "tag"],
                        help="分组方式: method 或 tag")
    args = parser.parse_args()

    records = load_log(args.log)
    print(f"📊 加载了 {len(records)} 条实验记录")

    # 按标签过滤
    if args.tag:
        records = [r for r in records if args.tag in r.get('tag', '')]
        print(f"   过滤后: {len(records)} 条 (tag contains '{args.tag}')")

    if len(records) == 0:
        print("❌ 没有可分析的记录")
        return

    analyze(records, group_key=args.group_by)

    # 输出每条记录的详情
    print("=" * 72)
    print("  详细记录列表")
    print("=" * 72)
    for i, r in enumerate(records):
        e = r.get('errors', {})
        print(f"  [{i+1}] {r.get('timestamp', 'N/A')}  "
              f"method={r.get('method', '?')}  tag={r.get('tag', '')}  "
              f"err_xy={e.get('err_xy', 'N/A')}px  "
              f"area={e.get('err_area_ratio_pct', 'N/A')}%")
    print()


if __name__ == "__main__":
    main()

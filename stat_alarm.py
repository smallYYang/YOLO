#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 out_warn 目录下所有 JSON 文件中 alarm 字段的情况
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def count_alarm_statistics(directory="out_warn"):
    """
    统计指定目录下所有 JSON 文件的 alarm 字段
    
    Args:
        directory: JSON 文件所在的目录路径
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"错误: 目录 {directory} 不存在")
        return
    
    # 统计变量
    total_files = 0
    alarm_true_count = 0
    alarm_false_count = 0
    alarm_missing_count = 0
    
    # 详细统计：记录 alarm 为 true 和 false 的文件名
    alarm_true_files = []
    alarm_false_files = []
    alarm_missing_files = []
    
    # 遍历所有 JSON 文件
    json_files = sorted(directory_path.glob("*.json"))
    
    if not json_files:
        print(f"警告: 在 {directory} 目录下没有找到 JSON 文件")
        return
    
    print(f"正在统计 {directory} 目录下的 JSON 文件...")
    print(f"找到 {len(json_files)} 个 JSON 文件\n")
    
    for json_file in json_files:
        total_files += 1
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查顶层 alarm 字段
            if 'alarm' in data:
                if data['alarm'] is True:
                    alarm_true_count += 1
                    alarm_true_files.append(json_file.name)
                elif data['alarm'] is False:
                    alarm_false_count += 1
                    alarm_false_files.append(json_file.name)
                else:
                    # alarm 字段存在但不是布尔值
                    print(f"警告: {json_file.name} 的 alarm 字段不是布尔值: {data['alarm']}")
            else:
                alarm_missing_count += 1
                alarm_missing_files.append(json_file.name)
                print(f"警告: {json_file.name} 缺少 alarm 字段")
                
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析 {json_file.name}: {e}")
        except Exception as e:
            print(f"错误: 处理 {json_file.name} 时出错: {e}")
    
    # 输出统计结果
    print("=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总文件数: {total_files}")
    print(f"alarm = true:  {alarm_true_count} ({alarm_true_count/total_files*100:.2f}%)")
    print(f"alarm = false: {alarm_false_count} ({alarm_false_count/total_files*100:.2f}%)")
    print(f"缺少 alarm 字段: {alarm_missing_count}")
    print("=" * 60)
    
    # 可选：输出详细列表
    if alarm_true_files:
        print(f"\n{alarm_true_count} 个 alarm=true 的文件:")
        for i, filename in enumerate(alarm_true_files[:10], 1):  # 只显示前10个
            print(f"  {i}. {filename}")
        if len(alarm_true_files) > 10:
            print(f"  ... 还有 {len(alarm_true_files) - 10} 个文件")
    
    if alarm_false_files:
        print(f"\n{alarm_false_count} 个 alarm=false 的文件:")
        for i, filename in enumerate(alarm_false_files[:10], 1):  # 只显示前10个
            print(f"  {i}. {filename}")
        if len(alarm_false_files) > 10:
            print(f"  ... 还有 {len(alarm_false_files) - 10} 个文件")
    
    if alarm_missing_files:
        print(f"\n缺少 alarm 字段的文件:")
        for filename in alarm_missing_files:
            print(f"  - {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统计指定目录下所有 JSON 文件中 alarm 字段的情况"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="out_warn",
        help="要统计的目录路径（默认为 out_warn）"
    )
    
    args = parser.parse_args()
    count_alarm_statistics(args.directory)


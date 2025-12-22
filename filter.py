"""
filter_equipment_labels.py
过滤YOLO标注文件：只保留equipment标签并重编号
labels文件夹下只有txt文件
"""

import os
import sys
from pathlib import Path

def process_single_file(input_file, output_file):
    """
    处理单个标注文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    Returns:
        (equipment_count, total_lines): 处理的equipment数量和原始总行数
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        equipment_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                class_id = parts[0]
                
                # 只保留class_id为"2"的行
                if class_id == "2":
                    # 将class_id改为"0"
                    parts[0] = "0"
                    equipment_lines.append(' '.join(parts))
        
        # 写入输出文件（即使为空也创建文件）
        with open(output_file, 'w', encoding='utf-8') as f:
            if equipment_lines:
                f.write('\n'.join(equipment_lines))
        
        return len(equipment_lines), total_lines
    
    except Exception as e:
        print(f"  错误: {e}")
        return 0, 0

def process_folder(input_folder, output_folder):
    """
    处理整个文件夹
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_files = 0
    files_with_equipment = 0
    total_equipments = 0
    total_original_annotations = 0
    empty_files = 0
    
    print(f"处理文件夹: {input_folder}")
    print(f"输出到: {output_folder}")
    print("-" * 50)
    
    # 遍历所有txt文件
    for txt_file in input_path.glob("*.txt"):
        total_files += 1
        
        # 构建输出文件路径
        output_file = output_path / txt_file.name
        
        # 处理文件
        equipment_count, original_count = process_single_file(txt_file, output_file)
        
        total_equipments += equipment_count
        total_original_annotations += original_count
        
        # 更新统计
        if equipment_count > 0:
            files_with_equipment += 1
            print(f"✓ {txt_file.name}: 保留 {equipment_count}/{original_count} 个标注")
        else:
            empty_files += 1
            print(f"○ {txt_file.name}: 无equipment标注 ({original_count} 个原始标注)")
    
    print("-" * 50)
    print("处理完成！")
    print(f"总文件数: {total_files}")
    print(f"包含equipment的文件数: {files_with_equipment}")
    print(f"空的标注文件数: {empty_files}")
    print(f"原始标注总数: {total_original_annotations}")
    print(f"equipment标注总数: {total_equipments}")
    
    if files_with_equipment > 0:
        avg_per_file = total_equipments / files_with_equipment
        print(f"平均每个文件的equipment数量: {avg_per_file:.2f}")
    
    # 保存统计信息
    stats_file = output_path / "processing_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("标注文件处理统计\n")
        f.write("=" * 40 + "\n")
        f.write(f"输入文件夹: {input_folder}\n")
        f.write(f"输出文件夹: {output_folder}\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"包含equipment的文件数: {files_with_equipment}\n")
        f.write(f"空的标注文件数: {empty_files}\n")
        f.write(f"原始标注总数: {total_original_annotations}\n")
        f.write(f"equipment标注总数: {total_equipments}\n")
        if files_with_equipment > 0:
            f.write(f"平均每个文件的equipment数量: {avg_per_file:.2f}\n")
    
    print(f"\n统计信息已保存到: {stats_file}")

def main():
    """
    主函数：处理命令行参数
    """
    if len(sys.argv) != 3:
        print("使用方法:")
        print("  python filter_equipment_labels.py <输入文件夹> <输出文件夹>")
        print("\n示例:")
        print("  python filter_equipment_labels.py data/labels data/equipment_labels")
        print("\n注意:")
        print("  1. 输入文件夹应包含YOLO格式的txt标注文件")
        print("  2. 程序会删除class_id为0和1的行")
        print("  3. 只保留class_id为2的行，并重编号为0")
        print("  4. 空的文件也会被创建（内容为空）")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在！")
        sys.exit(1)
    
    # 检查输入文件夹是否包含txt文件
    txt_files = list(Path(input_folder).glob("*.txt"))
    if not txt_files:
        print(f"警告: 输入文件夹 '{input_folder}' 中没有找到txt文件！")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # 开始处理
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
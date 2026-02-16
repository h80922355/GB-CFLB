"""
基于Gnuplot的高质量准确率可视化模块 - 期刊级专业风格
生成带粒球事件标记的准确率曲线，支持PNG、PDF、EPS多格式输出
采用半透明背景区分训练阶段，事件线段清晰标记
完全替代matplotlib实现
改进：增大所有字体，优化布局，完全匹配matplotlib视觉效果
修复：图例显示在最上层，不被标识线遮挡
"""
import os
import subprocess
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from gbcfl.utils.logger import ensure_output_dir


class AccuracyPlotterGnuplot:
    """基于Gnuplot的专业准确率绘图器"""

    def __init__(self, output_dir='outputs'):
        """
        初始化绘图器

        参数:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 专业配色方案（参考期刊标准）
        self.colors = {
            'accuracy_curve': '#1E90FF',      # 道奇蓝 - 主曲线
            'standard_phase': '#D5D5D5',      # 浅灰色 - 标准FL阶段
            'granular_phase': '#B3D9FF',      # 浅蓝色 - 粒球聚类阶段
            'split_event': '#FF4444',         # 红色 - 分裂事件
            'merge_event': '#22AA22',         # 绿色 - 合并事件
            'grid': '#E0E0E0',                # 网格线颜色
            'border': '#333333',              # 边框颜色
        }

        # 透明度设置
        self.alpha = {
            'standard_phase': 0.3,
            'granular_phase': 0.3,
        }

    def generate_data_files(self, rounds: np.ndarray, accuracies: np.ndarray,
                           events: List[Dict], first_split_round: int,
                           communication_rounds: int, temp_dir: str) -> Dict[str, str]:
        """
        生成所有Gnuplot所需的.dat数据文件

        参数:
            rounds: 轮次数组
            accuracies: 准确率数组
            events: 事件列表
            first_split_round: 首次分裂轮次
            communication_rounds: 总通信轮数
            temp_dir: 临时目录

        返回:
            数据文件路径字典
        """
        data_files = {}

        # 1. 主准确率数据文件
        accuracy_file = os.path.join(temp_dir, 'accuracy_data.dat')
        with open(accuracy_file, 'w') as f:
            f.write("# Round  Accuracy\n")
            for r, a in zip(rounds, accuracies):
                f.write(f"{r:.1f}  {a:.6f}\n")
        data_files['accuracy'] = accuracy_file

        # 2. 阶段区域数据（用于背景色）
        if first_split_round > 0:
            # 标准FL阶段区域
            standard_phase_file = os.path.join(temp_dir, 'standard_phase.dat')
            with open(standard_phase_file, 'w') as f:
                f.write("# X_start  Y_start  X_end  Y_end\n")
                f.write(f"0  0  {first_split_round}  1.0\n")
            data_files['standard_phase'] = standard_phase_file

            # 粒球聚类阶段区域
            granular_phase_file = os.path.join(temp_dir, 'granular_phase.dat')
            with open(granular_phase_file, 'w') as f:
                f.write("# X_start  Y_start  X_end  Y_end\n")
                f.write(f"{first_split_round}  0  {communication_rounds}  1.0\n")
            data_files['granular_phase'] = granular_phase_file

        # 3. 分裂事件数据
        split_events = [e for e in events if e.get('type') == 'Split']
        if split_events:
            split_file = os.path.join(temp_dir, 'split_events.dat')
            with open(split_file, 'w') as f:
                f.write("# Round  Y_start  Y_end\n")
                for event in split_events:
                    r = event['round']
                    f.write(f"{r}  0  1.0\n")
            data_files['split_events'] = split_file

        # 4. 合并事件数据
        merge_events = [e for e in events if e.get('type') == 'Merge']
        if merge_events:
            merge_file = os.path.join(temp_dir, 'merge_events.dat')
            with open(merge_file, 'w') as f:
                f.write("# Round  Y_start  Y_end\n")
                for event in merge_events:
                    r = event['round']
                    f.write(f"{r}  0  1.0\n")
            data_files['merge_events'] = merge_file

        return data_files

    def generate_gnuplot_script(self, data_files: Dict[str, str],
                                output_file: str, format_type: str,
                                first_split_round: int, communication_rounds: int,
                                events: List[Dict]) -> str:
        """
        生成Gnuplot脚本

        参数:
            data_files: 数据文件路径字典
            output_file: 输出文件路径
            format_type: 输出格式 ('png', 'pdf', 'eps')
            first_split_round: 首次分裂轮次
            communication_rounds: 总通信轮数
            events: 事件列表

        返回:
            Gnuplot脚本内容
        """
        # 根据格式类型设置终端和字体大小（匹配matplotlib效果）
        if format_type == 'png':
            # PNG: 更大的尺寸以匹配matplotlib输出
            terminal = "set terminal pngcairo enhanced size 1600,960 font 'Arial,20'"
        elif format_type == 'pdf':
            # PDF: 矢量格式
            terminal = "set terminal pdfcairo enhanced size 10in,6in font 'Arial,20'"
        elif format_type == 'eps':
            # EPS: PostScript格式
            terminal = "set terminal postscript eps enhanced color size 10in,6in font 'Arial,20'"
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")

        script = f"""# Gnuplot script for accuracy with events - Professional Journal Style
# Optimized to match matplotlib visual quality
# Fixed: Legend now appears on top of all elements

# ========== Terminal and Output ==========
{terminal}
set output '{output_file}'

# ========== Global Style Settings ==========
set encoding utf8
set style line 1 lc rgb '{self.colors['accuracy_curve']}' lt 1 lw 5
set style line 2 lc rgb '{self.colors['split_event']}' lt 1 lw 4
set style line 3 lc rgb '{self.colors['merge_event']}' lt 1 lw 4
set style fill solid 1.0 border -1

# ========== Axes and Labels (Optimized for Readability) ==========
set xlabel 'Communication Rounds' font 'Arial,24' offset 0,-0.8
set ylabel 'Accuracy' font 'Arial,24' offset -2.0,0
set xrange [0:{communication_rounds}]
set yrange [0:1.05]

# ========== Grid Settings ==========
set grid xtics ytics
set grid linewidth 1.5 linetype 0 linecolor rgb '{self.colors['grid']}'

# ========== Border Settings ==========
set border linewidth 2.5 linecolor rgb '{self.colors['border']}'

# ========== Tics Settings (Larger Numbers) ==========
set xtics font 'Arial,20' offset 0,-0.4
set ytics font 'Arial,20' offset -0.5,0
set xtics 0,10,{communication_rounds}
set ytics 0,0.2,1.0
set format x '%g'
set format y '%.1f'

# ========== Margins (Optimized Layout) ==========
set lmargin 12
set rmargin 7
set tmargin 3
set bmargin 5

# ========== Legend Settings (Matching matplotlib style - on top layer) ==========
set key bottom right
set key box linewidth 2.5 linecolor rgb '{self.colors['border']}'
set key spacing 2.0
set key samplen 4.5
set key width 0.5
set key height 0.8
set key font 'Arial,18'
set key opaque
set key maxrows 6
set key vertical

"""

        # 添加背景区域（使用graph坐标系确保完全覆盖）
        if first_split_round > 0:
            script += f"""# ========== Phase Background Regions (Full Coverage) ==========
# Standard FL Phase (semi-transparent)
set obj 1 rect from graph 0, graph 0 to first {first_split_round}, graph 1 \\
    fc rgb '{self.colors['standard_phase']}' \\
    fs solid {self.alpha['standard_phase']} behind

# Granular Ball Clustering Phase (semi-transparent)
set obj 2 rect from first {first_split_round}, graph 0 to graph 1, graph 1 \\
    fc rgb '{self.colors['granular_phase']}' \\
    fs solid {self.alpha['granular_phase']} behind

"""

        # 添加事件竖线（改为behind，确保不遮挡图例）
        split_events = [e for e in events if e.get('type') == 'Split']
        merge_events = [e for e in events if e.get('type') == 'Merge']

        arrow_id = 3  # 从3开始（1,2用于背景矩形）

        if split_events:
            script += "# ========== Split Event Lines (Above curve, below legend) ==========\n"
            for event in split_events:
                r = event['round']
                script += f"set arrow {arrow_id} from {r}, graph 0 to {r}, graph 1 nohead \\\n"
                script += f"    lc rgb '{self.colors['split_event']}' lw 3.5\n"
                arrow_id += 1
            script += "\n"

        if merge_events:
            script += "# ========== Merge Event Lines (Above curve, below legend) ==========\n"
            for event in merge_events:
                r = event['round']
                script += f"set arrow {arrow_id} from {r}, graph 0 to {r}, graph 1 nohead \\\n"
                script += f"    lc rgb '{self.colors['merge_event']}' lw 3.5\n"
                arrow_id += 1
            script += "\n"

        # 构建绘图命令
        plot_commands = []

        # 主准确率曲线
        plot_commands.append(
            f"'{data_files['accuracy']}' using 1:2 with lines "
            f"lc rgb '{self.colors['accuracy_curve']}' lw 5 "
            f"title 'Accuracy'"
        )

        # 添加图例项（使用虚拟数据点）
        if first_split_round > 0:
            # 标准FL阶段图例
            plot_commands.append(
                f"1/0 with boxes "
                f"fc rgb '{self.colors['standard_phase']}' "
                f"fs solid {self.alpha['standard_phase']} "
                f"title 'Standard FL Phase'"
            )
            # 粒球聚类阶段图例
            plot_commands.append(
                f"1/0 with boxes "
                f"fc rgb '{self.colors['granular_phase']}' "
                f"fs solid {self.alpha['granular_phase']} "
                f"title 'GB Clustering Phase'"
            )

        # 事件图例
        if split_events:
            plot_commands.append(
                f"1/0 with lines "
                f"lc rgb '{self.colors['split_event']}' lw 4 "
                f"title 'Granular Ball Split'"
            )

        if merge_events:
            plot_commands.append(
                f"1/0 with lines "
                f"lc rgb '{self.colors['merge_event']}' lw 4 "
                f"title 'Granular Ball Merge'"
            )

        # 组合绘图命令
        script += "# ========== Plot Command ==========\n"
        script += "plot " + ", \\\n     ".join(plot_commands) + "\n"

        return script

    def plot_accuracy_with_events(self, rounds: np.ndarray, accuracies: np.ndarray,
                                  events: List[Dict], first_split_round: int,
                                  communication_rounds: int,
                                  filename: str = "accuracy_with_events",
                                  generate_png: bool = True,
                                  generate_pdf: bool = True,
                                  generate_eps: bool = True) -> Dict[str, str]:
        """
        绘制带事件标记的准确率曲线（多格式输出）

        参数:
            rounds: 轮次数组
            accuracies: 准确率数组
            events: 事件列表
            first_split_round: 首次分裂轮次
            communication_rounds: 总通信轮数
            filename: 输出文件名（不含扩展名）
            generate_png: 是否生成PNG
            generate_pdf: 是否生成PDF
            generate_eps: 是否生成EPS

        返回:
            生成的文件路径字典
        """
        # 确保数据从(0,0)开始
        if rounds[0] != 0:
            rounds = np.concatenate([[0], rounds])
            accuracies = np.concatenate([[0.0], accuracies])

        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='gnuplot_accuracy_')
        generated_files = {}

        try:
            # 生成数据文件
            data_files = self.generate_data_files(
                rounds, accuracies, events,
                first_split_round, communication_rounds, temp_dir
            )

            # 检查gnuplot是否可用
            try:
                subprocess.run(['gnuplot', '--version'],
                             capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("错误：未找到gnuplot命令，请确保已安装gnuplot")
                return generated_files

            # 生成各种格式的图表
            formats_to_generate = []
            if generate_png:
                formats_to_generate.append(('png', f"{filename}.png"))
            if generate_pdf:
                formats_to_generate.append(('pdf', f"{filename}.pdf"))
            if generate_eps:
                formats_to_generate.append(('eps', f"{filename}.eps"))

            for format_type, output_filename in formats_to_generate:
                output_file = self.output_dir / output_filename

                # 生成Gnuplot脚本
                script_content = self.generate_gnuplot_script(
                    data_files, str(output_file), format_type,
                    first_split_round, communication_rounds, events
                )

                # 写入脚本文件
                script_file = os.path.join(temp_dir, f'plot_script_{format_type}.gp')
                with open(script_file, 'w', encoding='utf-8') as f:
                    f.write(script_content)

                # 执行Gnuplot
                try:
                    result = subprocess.run(
                        ['gnuplot', script_file],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if result.returncode != 0:
                        print(f"Gnuplot警告 ({format_type.upper()}): {result.stderr}")

                    # 验证文件是否生成
                    if output_file.exists():
                        generated_files[format_type] = str(output_file)
                        file_size = output_file.stat().st_size / 1024  # KB
                        print(f"✓ {format_type.upper()}图表已生成: {output_file} ({file_size:.1f} KB)")
                    else:
                        print(f"✗ {format_type.upper()}图表生成失败")

                except subprocess.TimeoutExpired:
                    print(f"✗ Gnuplot执行超时 ({format_type.upper()})")
                except Exception as e:
                    print(f"✗ 生成{format_type.upper()}时出错: {e}")

            # 保存数据文件到输出目录（可选）
            self._save_data_files_to_output(data_files, filename)

        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"清理临时文件时出错: {e}")

        return generated_files

    def _save_data_files_to_output(self, data_files: Dict[str, str], base_filename: str):
        """
        保存数据文件到输出目录（用于后续分析或手动绘图）

        参数:
            data_files: 数据文件路径字典
            base_filename: 基础文件名
        """
        data_output_dir = self.output_dir / 'plot_data'
        data_output_dir.mkdir(exist_ok=True)

        for data_type, data_file in data_files.items():
            if os.path.exists(data_file):
                output_name = f"{base_filename}_{data_type}.dat"
                output_path = data_output_dir / output_name
                try:
                    shutil.copy2(data_file, output_path)
                except Exception as e:
                    print(f"保存数据文件 {data_type} 时出错: {e}")


# ========== 便捷函数接口（兼容原有API） ==========

def plot_accuracy_with_events(cfl_stats, first_split_round: int, communication_rounds: int,
                              output_dir: str = "outputs", filename: str = "accuracy_with_events.png",
                              events: Optional[List[Dict]] = None, use_honest_only: bool = True,
                              generate_pdf: bool = True, generate_eps: bool = True,
                              silent_export: bool = False, **kwargs):
    """
    使用Gnuplot绘制带粒球事件标记的准确率曲线（兼容原接口）

    参数:
        cfl_stats: 包含训练统计信息的ExperimentLogger对象
        first_split_round: 首次分裂轮次
        communication_rounds: 总通信轮数
        output_dir: 输出目录
        filename: 输出文件名
        events: 事件列表
        use_honest_only: 是否只使用诚实节点准确率
        generate_pdf: 是否生成PDF文件
        generate_eps: 是否生成EPS文件
        silent_export: 静默模式（忽略）
        **kwargs: 其他兼容参数
    """
    # 确保输出目录存在
    ensure_output_dir(output_dir)

    # 检查数据可用性
    if not hasattr(cfl_stats, "rounds") or len(cfl_stats.rounds) == 0:
        print(f"警告：没有足够的数据绘制准确率曲线。")
        return False

    # 提取准确率数据
    rounds = np.array(cfl_stats.rounds)

    # 根据参数选择使用诚实节点准确率还是全体准确率
    if use_honest_only and hasattr(cfl_stats, 'honest_acc_clients') and cfl_stats.honest_acc_clients:
        # 使用诚实节点准确率
        acc_data = cfl_stats.honest_acc_clients
        acc_mean = np.array([np.mean(round_acc) if round_acc else 0.0
                             for round_acc in acc_data])
    elif hasattr(cfl_stats, 'honest_acc_clients') and cfl_stats.honest_acc_clients:
        # 优先使用诚实节点准确率
        acc_data = cfl_stats.honest_acc_clients
        acc_mean = np.array([np.mean(round_acc) if round_acc else 0.0
                             for round_acc in acc_data])
    else:
        # 后备：使用全体客户端准确率（过滤NaN）
        acc_data = cfl_stats.acc_clients
        acc_mean = []
        for round_acc in acc_data:
            if round_acc:
                valid_acc = [a for a in round_acc if not np.isnan(a)]
                if valid_acc:
                    acc_mean.append(np.mean(valid_acc))
                else:
                    acc_mean.append(0.0)
            else:
                acc_mean.append(0.0)
        acc_mean = np.array(acc_mean)

    # 提取事件（如果未提供）
    if events is None:
        events = []
        if hasattr(cfl_stats, "ball_events"):
            if isinstance(cfl_stats.ball_events, (list, tuple)) and len(cfl_stats.ball_events) > 0:
                for event in cfl_stats.ball_events:
                    if isinstance(event, dict) and 'type' in event and 'round' in event:
                        events.append({
                            'round': event['round'],
                            'type': event['type']
                        })

    # 创建绘图器
    plotter = AccuracyPlotterGnuplot(output_dir=output_dir)

    # 提取文件名（不含扩展名）
    base_filename = filename.replace('.png', '').replace('.pdf', '').replace('.eps', '')

    # 生成图表
    generated_files = plotter.plot_accuracy_with_events(
        rounds, acc_mean, events,
        first_split_round, communication_rounds,
        filename=base_filename,
        generate_png=True,
        generate_pdf=generate_pdf,
        generate_eps=generate_eps
    )

    # 返回是否成功生成至少一个文件
    return len(generated_files) > 0


if __name__ == "__main__":
    # 测试代码
    print("测试Gnuplot准确率绘图模块（完全优化版 - 匹配matplotlib质量）...")

    # 创建模拟数据
    rounds = np.arange(0, 101, 1)
    accuracies = 0.3 + 0.6 * (1 - np.exp(-rounds / 30))
    accuracies = np.minimum(accuracies, 0.95)

    # 模拟事件
    events = [
        {'round': 50, 'type': 'Split'},
        {'round': 55, 'type': 'Split'},
        {'round': 70, 'type': 'Merge'},
        {'round': 80, 'type': 'Split'},
    ]

    # 创建绘图器
    plotter = AccuracyPlotterGnuplot(output_dir='test_outputs')

    # 生成图表
    result = plotter.plot_accuracy_with_events(
        rounds, accuracies, events,
        first_split_round=50,
        communication_rounds=100,
        filename="test_accuracy_with_events",
        generate_png=True,
        generate_pdf=True,
        generate_eps=True
    )

    print(f"\n生成的文件: {result}")
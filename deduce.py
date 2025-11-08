#!/usr/bin/env python3
"""
YOLOv8分割模型推理脚本
功能：对输入图像进行损伤检测和分割，用中文标签标注结果
使用方法：python inference.py --image 图片路径
"""

import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class DamageDetector:
    def __init__(self, model_path, output_dir="inference_results"):
        """
        初始化损伤检测器
        
        Args:
            model_path: 训练好的模型权重路径
            output_dir: 结果保存目录
        """
        # 加载模型
        self.model = YOLO(model_path)
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 中文标签映射
        self.chinese_labels = {
            0: "完好",
            1: "轻度损毁", 
            2: "重度损毁",
            3: "倒塌",
            4: "状况未知"
        }
        
        # 颜色映射（BGR格式）
        self.color_map = {
            0: (0, 255, 0),      # 绿色 - 完好
            1: (255, 255, 0),    # 青色 - 轻度损毁
            2: (0, 165, 255),    # 橙色 - 重度损毁  
            3: (0, 0, 255),      # 红色 - 倒塌
            4: (128, 128, 128)   # 灰色 - 状况未知
        }
        
        print(f"✅ 模型加载成功: {model_path}")
        print(f"✅ 结果将保存到: {self.output_dir.absolute()}")
    
    def load_chinese_font(self, font_size=20):
        """
        尝试加载中文字体
        """
        # 尝试多种常见中文字体路径
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/System/Library/Fonts/PingFang.ttc",               # macOS
            "C:/Windows/Fonts/simhei.ttf",                      # Windows
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux备用
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except:
                    continue
        
        # 如果找不到中文字体，使用默认字体（可能不支持中文）
        print("⚠️ 未找到中文字体，使用默认字体（可能无法显示中文）")
        return ImageFont.load_default()
    
    def process_image(self, image_path, conf_threshold=0.25):
        """
        处理单张图像
        
        Args:
            image_path: 输入图像路径
            conf_threshold: 置信度阈值
            
        Returns:
            result_image: 处理后的图像路径
        """
        # 检查输入文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 未找到图像文件: {image_path}")
        
        print(f"🔍 开始处理图像: {image_path}")
        
        # 使用YOLO模型进行推理
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            imgsz=1024,  # 使用训练时的尺寸
            save=False,   # 我们不使用内置保存功能，自己控制输出
            verbose=False # 减少输出
        )
        
        if len(results) == 0:
            print("⚠️ 未检测到任何目标")
            return self._save_no_detection_result(image_path)
        
        # 获取第一个结果（单张图像）
        result = results[0]
        
        if result.masks is None or len(result.boxes) == 0:
            print("⚠️ 未检测到分割目标")
            return self._save_no_detection_result(image_path)
        
        # 读取原始图像
        original_image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 使用PIL进行图像处理（更好的文本支持）
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        font = self.load_chinese_font(24)
        
        # 获取检测信息
        boxes = result.boxes
        masks = result.masks
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        print(f"🎯 检测到 {len(boxes)} 个目标")
        
        # 绘制每个检测结果
        for i, (class_id, confidence, mask) in enumerate(zip(class_ids, confidences, masks.data)):
            # 获取类别信息
            damage_type = int(class_id)
            chinese_label = self.chinese_labels.get(damage_type, "未知")
            color = self.color_map.get(damage_type, (255, 255, 255))
            
            # 处理分割掩码
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]))
            
            # 创建彩色掩码
            color_mask = np.zeros_like(original_image)
            color_mask[mask_resized > 0.5] = color[::-1]  # RGB转BGR
            
            # 将掩码叠加到原图（透明度混合）
            alpha = 0.3
            original_image = cv2.addWeighted(original_image, 1, color_mask, alpha, 0)
            
            # 获取边界框坐标（用于放置标签）
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            
            # 在PIL图像上绘制文本（更好的中文支持）
            label_text = f"{chinese_label} {confidence:.2f}"
            draw.text((x1, y1 - 30), label_text, fill=color, font=font)
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            print(f"  - 目标 {i+1}: {chinese_label} (置信度: {confidence:.3f})")
        
        # 转换回OpenCV格式并保存
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 生成输出文件名
        input_path = Path(image_path)
        output_filename = f"result_{input_path.stem}.png"
        output_path = self.output_dir / output_filename
        
        # 保存结果
        cv2.imwrite(str(output_path), result_image)
        
        # 创建检测结果统计图
        stats_image_path = self._create_statistics_chart(class_ids, confidences, output_path.stem)
        
        print(f"✅ 结果已保存: {output_path}")
        print(f"📊 统计图表: {stats_image_path}")
        
        return str(output_path), stats_image_path
    
    def _save_no_detection_result(self, image_path):
        """处理无检测结果的情况"""
        original_image = cv2.imread(image_path)
        
        # 在图像上添加"未检测到目标"的文本
        text = "未检测到损伤目标"
        font_scale = 2
        thickness = 3
        color = (0, 0, 255)  # 红色
        
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # 计算文本位置（居中）
        x = (original_image.shape[1] - text_width) // 2
        y = (original_image.shape[0] + text_height) // 2
        
        # 添加文本
        cv2.putText(original_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # 保存结果
        input_path = Path(image_path)
        output_path = self.output_dir / f"no_detection_{input_path.stem}.png"
        cv2.imwrite(str(output_path), original_image)
        
        return str(output_path), None
    
    def _create_statistics_chart(self, class_ids, confidences, image_name):
        """创建检测结果统计图表"""
        # 统计每个类别的数量
        unique, counts = np.unique(class_ids, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # 准备数据
        labels = [self.chinese_labels.get(i, "未知") for i in unique]
        sizes = [class_counts.get(i, 0) for i in unique]
        colors = [tuple(c/255 for c in self.color_map[i][::-1]) for i in unique]  # 转换颜色格式
        
        # 创建饼图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 饼图 - 各类别数量分布
        if len(sizes) > 0:
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('损伤类型分布')
        
        # 柱状图 - 置信度分布
        if len(confidences) > 0:
            confidence_ranges = [0.2, 0.4, 0.6, 0.8, 1.0]
            confidence_counts = [np.sum((confidences >= low) & (confidences < high)) 
                               for low, high in zip([0.0] + confidence_ranges[:-1], confidence_ranges)]
            
            ax2.bar(['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], confidence_counts, 
                   color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
            ax2.set_title('置信度分布')
            ax2.set_ylabel('目标数量')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.output_dir / f"stats_{image_name}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def process_directory(self, directory_path, conf_threshold=0.25):
        """
        处理目录中的所有图像
        
        Args:
            directory_path: 包含图像的目录路径
            conf_threshold: 置信度阈值
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"❌ 目录不存在: {directory_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in directory.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"⚠️ 在目录中未找到图像文件: {directory_path}")
            return
        
        print(f"📁 发现 {len(image_files)} 个图像文件")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n--- 处理第 {i}/{len(image_files)} 张图像: {image_file.name} ---")
            try:
                result_path, stats_path = self.process_image(str(image_file), conf_threshold)
                results.append((str(image_file), result_path, stats_path))
            except Exception as e:
                print(f"❌ 处理图像失败 {image_file}: {e}")
                results.append((str(image_file), None, None))
        
        # 生成处理报告
        self._generate_report(results, directory_path)
        
        return results
    
    def _generate_report(self, results, source_path):
        """生成处理报告"""
        report_path = self.output_dir / "inference_report.txt"
        
        successful = sum(1 for _, result_path, _ in results if result_path is not None)
        failed = len(results) - successful
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLOv8损伤检测推理报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"源路径: {source_path}\n")
            f.write(f"处理时间: {np.datetime64('now')}\n")
            f.write(f"总图像数: {len(results)}\n")
            f.write(f"成功处理: {successful}\n")
            f.write(f"处理失败: {failed}\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 50 + "\n")
            for input_path, result_path, stats_path in results:
                f.write(f"输入: {Path(input_path).name}\n")
                f.write(f"结果: {Path(result_path).name if result_path else '失败'}\n")
                f.write(f"统计: {Path(stats_path).name if stats_path else '无'}\n")
                f.write("-" * 30 + "\n")
        
        print(f"📄 处理报告已生成: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8损伤检测推理脚本')
    parser.add_argument('--image', type=str, required=True, 
                       help='输入图像路径')
    parser.add_argument('--model', type=str, default='trained_model/yolov8n-seg-1024/weights/best.pt',
                       help='模型权重路径 (默认: trained_model/yolov8n-seg-1024/weights/best.pt)')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='输出目录 (默认: inference_results)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--dir', type=str, 
                       help='处理整个目录而不是单张图像')
    
    args = parser.parse_args()
    
    try:
        # 创建检测器实例
        detector = DamageDetector(args.model, args.output)
        
        if args.dir:
            # 处理整个目录
            print(f"📁 开始处理目录: {args.dir}")
            detector.process_directory(args.dir, args.conf)
        else:
            # 处理单张图像
            result_path, stats_path = detector.process_image(args.image, args.conf)
            
            print("\n" + "="*50)
            print("🎉 处理完成!")
            print(f"📁 结果图像: {result_path}")
            if stats_path:
                print(f"📊 统计图表: {stats_path}")
            print("="*50)
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 如果直接运行脚本，显示使用说明
    if len(os.sys.argv) == 1:
        print("""
YOLOv8损伤检测推理工具
使用方法:
    
1. 检测单张图像:
   python inference.py --image 图片路径.jpg
    
2. 检测目录中的所有图像:
   python inference.py --dir 图片目录路径
    
3. 使用自定义模型和参数:
   python inference.py --image 图片路径.jpg --model 模型路径.pt --conf 0.3
    
4. 查看所有参数:
   python inference.py --help

示例:
   python inference.py --image test_image.jpg
   python inference.py --dir ./test_images --conf 0.3
        """)
    else:
        exit(main())

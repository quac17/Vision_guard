import os
import sys
import time
import csv
import psutil
import threading
from datetime import datetime

# Thêm thư mục ai-recognition vào đường dẫn để import được FaceRecognizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-recognition")))

from recognizer import FaceRecognizer

class PerformanceMonitor:
    def __init__(self):
        self.keep_measuring = False
        self.cpu_usages = []
        self.ram_usages = []
        self.thread = None
        self.process = psutil.Process(os.getpid())

    def _measure_loop(self):
        # Thiết lập base CPU
        self.process.cpu_percent(interval=None)
        while self.keep_measuring:
            try:
                cpu = self.process.cpu_percent(interval=0.1)
                ram_mb = self.process.memory_info().rss / (1024 * 1024)
                self.cpu_usages.append(cpu)
                self.ram_usages.append(ram_mb)
            except:
                pass

    def start(self):
        self.keep_measuring = True
        self.cpu_usages = []
        self.ram_usages = []
        self.thread = threading.Thread(target=self._measure_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.keep_measuring = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
        avg_cpu = sum(self.cpu_usages) / len(self.cpu_usages) if self.cpu_usages else 0.0
        max_cpu = max(self.cpu_usages) if self.cpu_usages else 0.0
        avg_ram = sum(self.ram_usages) / len(self.ram_usages) if self.ram_usages else 0.0
        max_ram = max(self.ram_usages) if self.ram_usages else 0.0
        
        return {
            "avg_cpu_percent": round(avg_cpu, 2),
            "max_cpu_percent": round(max_cpu, 2),
            "avg_ram_mb": round(avg_ram, 2),
            "max_ram_mb": round(max_ram, 2)
        }

def run_test(test_data_dir, model_path, db_path, output_dir, threshold=1.0):
    print("==============================================")
    print(f"BẮT ĐẦU CHẠY TEST BỘ DỮ LIỆU: {test_data_dir}")
    print("==============================================")
    
    if not os.path.exists(test_data_dir):
        print(f"LỖI: Không tìm thấy thư mục {test_data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_csv_path = os.path.join(output_dir, f"accuracy_{today_str}.csv")
    metrics_csv_path = os.path.join(output_dir, f"metrics_{today_str}.csv")

    try:
        # Khởi tạo recognizer, tắt call API để tránh rác database thật
        recognizer = FaceRecognizer(model_path, db_path, threshold=threshold)
        # Override hàm call API bằng một hàm trống để test an toàn
        recognizer.submit_attendance = lambda student_code: print(f"[MÔ PHỎNG] Đã gọi API điểm danh cho mã: {student_code}")
        
    except Exception as e:
        print(f"Lỗi khởi tạo Recognizer: {e}")
        return

    total_subjects = 0
    correct_identifications = 0
    unknown_identifications = 0
    wrong_identifications = 0

    accuracy_records = []
    metrics_records = []
    
    # Khởi tạo Object đo lường hiệu năng
    monitor = PerformanceMonitor()

    # Mở sẵn file CSV để ghi
    with open(accuracy_csv_path, mode='w', newline='', encoding='utf-8') as acc_file, \
         open(metrics_csv_path, mode='w', newline='', encoding='utf-8') as met_file:
         
        # Ẩn việc ghi file CSV từng dòng, thay bằng gom dữ liệu
        all_time_taken = []
        all_avg_cpu = []
        all_max_cpu = []
        all_avg_ram = []
        all_max_ram = []

        # Lặp qua từng người (thư mục con) trong test_data
        for subject_name in sorted(os.listdir(test_data_dir)):
            subject_dir = os.path.join(test_data_dir, subject_name)
            
            if not os.path.isdir(subject_dir):
                continue
                
            total_subjects += 1
            print(f"\n--- Đang test thư mục: {subject_name} ---")
            
            # Ghi nhận thời gian và bật theo dõi CPU/RAM
            start_time = time.time()
            monitor.start()
            
            # Chạy hàm nhận diện
            identified_name, distance = recognizer.recognize_batch(subject_dir)
            
            # Tắt theo dõi
            perf_data = monitor.stop()
            time_taken = round(time.time() - start_time, 4)
            
            if "Unknown" in identified_name:
                status = "UNKNOWN"
                unknown_identifications += 1
            elif identified_name == subject_name:
                status = "CORRECT"
                correct_identifications += 1
            else:
                status = "WRONG"
                wrong_identifications += 1
                
            print(f"Kết quả AI: {identified_name}")
            print(f"Khoảng cách L2: {distance:.4f}")
            print(f"Đánh giá: {status} | Thời gian xử lý: {time_taken}s")
            print(f"Hiệu năng: CPU trung bình {perf_data['avg_cpu_percent']}% | RAM tối đa {perf_data['max_ram_mb']}MB")
            
            # Gom dữ liệu để lưu vào tổng kết
            all_time_taken.append(time_taken)
            all_avg_cpu.append(perf_data['avg_cpu_percent'])
            all_max_cpu.append(perf_data['max_cpu_percent'])
            all_avg_ram.append(perf_data['avg_ram_mb'])
            all_max_ram.append(perf_data['max_ram_mb'])
            
        # ---------------------------------------------
        # TỔNG HỢP VÀ GHI VÀO FILE METRICS TỔNG
        # ---------------------------------------------
        total_time_taken = sum(all_time_taken)
        avg_time_per_subject = total_time_taken / max(1, len(all_time_taken))
        
        overall_avg_cpu = sum(all_avg_cpu) / max(1, len(all_avg_cpu))
        overall_max_cpu = max(all_max_cpu) if all_max_cpu else 0
        overall_avg_ram = sum(all_avg_ram) / max(1, len(all_avg_ram))
        overall_max_ram = max(all_max_ram) if all_max_ram else 0
        
        met_writer = csv.writer(met_file)
        met_writer.writerow([
            "Total_Subjects_Tested", 
            "Total_Time_Taken_Sec", 
            "Avg_Time_Per_Subject_Sec",
            "Overall_Avg_CPU_%", 
            "Overall_Max_CPU_%", 
            "Overall_Avg_RAM_MB", 
            "Overall_Max_RAM_MB"
        ])
        met_writer.writerow([
            total_subjects,
            round(total_time_taken, 4),
            round(avg_time_per_subject, 4),
            round(overall_avg_cpu, 2),
            round(overall_max_cpu, 2),
            round(overall_avg_ram, 2),
            round(overall_max_ram, 2)
        ])

        # ---------------------------------------------
        # TỔNG HỢP VÀ GHI VÀO FILE ACCURACY TỔNG
        # ---------------------------------------------
        acc_writer = csv.writer(acc_file)
        acc_writer.writerow([
            "Total_Subjects_Tested", 
            "Correct_Identifications", 
            "Correct_Rate_%", 
            "Wrong_Identifications", 
            "Wrong_Rate_%", 
            "Unknown_Identifications", 
            "Unknown_Rate_%"
        ])
        
        correct_rate = correct_identifications / max(1, total_subjects) * 100
        wrong_rate = wrong_identifications / max(1, total_subjects) * 100
        unknown_rate = unknown_identifications / max(1, total_subjects) * 100
        
        acc_writer.writerow([
            total_subjects,
            correct_identifications,
            round(correct_rate, 2),
            wrong_identifications,
            round(wrong_rate, 2),
            unknown_identifications,
            round(unknown_rate, 2)
        ])
            # Chặn hành vi xóa ảnh gốc của hàm cleanup trong recognizer khi đang chạy test đã được auto boqua vì đây là copy dir
            
    print("\n==============================================")
    print("BÁO CÁO KẾT QUẢ TEST:")
    print(f"Tổng số đối tượng test: {total_subjects}")
    print(f"Nhận diện ĐÚNG: {correct_identifications} ({correct_identifications/max(1, total_subjects)*100:.2f}%)")
    print(f"Nhận diện NHẦM: {wrong_identifications} ({wrong_identifications/max(1, total_subjects)*100:.2f}%)")
    print(f"KHÔNG nhận diện được: {unknown_identifications} ({unknown_identifications/max(1, total_subjects)*100:.2f}%)")
    print("\nFile Export Thành công:")
    print(f"- {accuracy_csv_path}")
    print(f"- {metrics_csv_path}")
    print("==============================================")


if __name__ == "__main__":
    # Đường dẫn tương đối dựa trên cấu trúc dự án từ thư mục gốc
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    test_dir = os.path.join(base_dir, "test_data")
    model_path = os.path.join(base_dir, "ai-recognition", "models", "mobilefacenet.tflite")
    db_path = os.path.join(base_dir, "ai-recognition", "local_db", "face_embeddings.json")
    output_dir = os.path.join(base_dir, "test_reports")
    
    # Bạn có thể điều chỉnh threshold ở đây để tinh chỉnh độ chính xác
    # Threshold 0.45 là phù hợp cho độ đo Cosine Distance (1 - Cosine Similarity) mới
    current_threshold = 0.45
    
    run_test(test_dir, model_path, db_path, output_dir, threshold=current_threshold)

import sys
import os
import subprocess
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
import threading
from pathlib import Path
import json
import cv2
from PIL import Image, ImageTk
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}

class ReviewDialog(tk.Toplevel):
    def __init__(self, parent, frames_dir):
        super().__init__(parent)
        self.title(f"Review: {Path(frames_dir).name}")
        self.geometry("900x700")
        self.configure(bg="#1e1e2e")
        self.frames_dir = frames_dir
        self.local_clusters_path = os.path.join(frames_dir, "local_clusters.json")
        self.global_clusters_path = os.path.join("temp", "clusters_data.json")
        
        self.local_data = {}
        self.global_clusters = []
        self.items = [] # Храним виджеты строк
        
        if not self._load_data():
            self.destroy()
            return
            
        self._build_ui()
    
    def _load_data(self):
        try:
            with open(self.local_clusters_path, "r", encoding="utf-8") as f:
                self.local_data = json.load(f)
            if os.path.exists(self.global_clusters_path):
                with open(self.global_clusters_path, "r", encoding="utf-8") as f:
                    self.global_clusters = json.load(f)
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")
            return False

    def _get_thumbnail(self, filename, location):
        path = os.path.join(self.frames_dir, filename)
        img = cv2.imread(path)
        if img is None: return None
        
        t, r, b, l = location
        h, w = img.shape[:2]
        pad = int((b-t)*0.2)
        t, b, l, r = max(0, t-pad), min(h, b+pad), max(0, l-pad), min(w, r+pad)
        
        face = img[t:b, l:r]
        if face.size == 0: return None
        face = cv2.resize(face, (100, 100))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(face))

    def _build_ui(self):
        # Заголовок
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(header, text=f"Проверка лиц в {Path(self.frames_dir).name}", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        
        # Скроллируемая область
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(container, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Список кластеров
        medoids = self.local_data.get("medoids", {})
        frames = self.local_data.get("frames", {})
        
        # Для каждого локального кластера найдем пример (медоид)
        unique_local = sorted([int(k) for k in medoids.keys()])
        
        # Подготовим список глобальных имен для выпадающего списка
        self.global_names = [f"{gc['label']} ({gc.get('name', 'Unknown')})" for gc in self.global_clusters]
        self.global_names.insert(0, "🆕 Создать нового человека")

        for local_id in unique_local:
            # Нам нужно найти кадр, где этот человек виден.
            found_face = None
            found_filename = None
            for filename, faces in frames.items():
                for face in faces:
                    if face.get("local_label") == local_id:
                        found_face = face
                        found_filename = filename
                        break
                if found_face:
                    break
            
            if not found_face:
                continue
            
            row = ttk.Frame(self.scrollable_frame, padding=5)
            row.pack(fill=tk.X, pady=2)
            
            # Миниатюра
            thumb = self._get_thumbnail(found_filename, found_face["location"])
            if thumb:
                img_label = tk.Label(row, image=thumb, bg="#313244")
                img_label.image = thumb # prevent GC
                img_label.pack(side=tk.LEFT, padx=5)
            
            info_frame = ttk.Frame(row)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
            
            ttk.Label(info_frame, text=f"Локальный ID: {local_id}", font=("Segoe UI", 9, "italic")).pack(anchor="w")
            
            # Текущее назначение
            current_gid = found_face.get("global_label", -1)
            current_gc = next((g for g in self.global_clusters if g["label"] == f"person_{current_gid}"), None)
            
            assigned_text = f"Назначен: person_{current_gid}"
            if current_gc and "name" in current_gc:
                assigned_text += f" ({current_gc['name']})"
            
            ttk.Label(info_frame, text=assigned_text, foreground="#a6adc8").pack(anchor="w")
            
            # Выбор нового назначения
            action_var = tk.StringVar()
            if current_gc:
                action_var.set(f"{current_gc['label']} ({current_gc.get('name', 'Unknown')})")
            else:
                action_var.set(self.global_names[0])
                
            combo = ttk.Combobox(info_frame, textvariable=action_var, values=self.global_names, width=40)
            combo.pack(anchor="w", pady=2)
            
            # Поле для ввода имени (если нужно)
            name_var = tk.StringVar()
            if current_gc: name_var.set(current_gc.get("name", ""))
            
            name_entry = tk.Entry(info_frame, textvariable=name_var, bg="#313244", fg="#cdd6f4", relief=tk.FLAT, insertbackground="#cdd6f4")
            name_entry.pack(anchor="w", fill=tk.X, pady=2)
            
            self.items.append({
                "local_id": local_id,
                "action_var": action_var,
                "name_var": name_var,
                "current_gid": current_gid
            })

        # Кнопки сохранения
        footer = ttk.Frame(self)
        footer.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(footer, text="💾 Сохранить изменения", command=self._save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(footer, text="❌ Отмена", command=self.destroy).pack(side=tk.RIGHT)

    def _save(self):
        # 1. Обновляем имена в глобальной базе
        # 2. Переназначаем лейблы если нужно
        # 3. Сохраняем clusters_data.json
        # 4. Обновляем local_clusters.json и перерисовываем аннотации (опционально)
        
        new_global_clusters = list(self.global_clusters)
        local_to_global = {-1: -1}
        
        for item in self.items:
            action = item["action_var"].get()
            new_name = item["name_var"].get().strip()
            
            if action == "🆕 Создать нового человека":
                # Создаем нового в базе
                max_id = max([int(g["label"].split("_")[1]) for g in new_global_clusters], default=-1)
                new_id = max_id + 1
                new_label = f"person_{new_id}"
                
                # Берем медоид из локальных данных
                medoid = self.local_data["medoids"][str(item["local_id"])]
                
                new_global_clusters.append({
                    "label": new_label,
                    "name": new_name if new_name else new_label,
                    "encodings": [medoid]
                })
                local_to_global[item["local_id"]] = new_id
            else:
                # Находим существующего
                label = action.split(" ")[0]
                target_gc = next((g for g in new_global_clusters if g["label"] == label), None)
                if target_gc:
                    target_gc["name"] = new_name
                    gid = int(label.split("_")[1])
                    local_to_global[item["local_id"]] = gid

        # Сохраняем глобальную базу
        try:
            with open(self.global_clusters_path, "w", encoding="utf-8") as f:
                json.dump(new_global_clusters, f, indent=4, ensure_ascii=False)
            
            # Обновляем локальные лейблы
            for filename, faces in self.local_data.get("frames", {}).items():
                for face in faces:
                    face["global_label"] = local_to_global.get(face["local_label"], -1)
            
            with open(self.local_clusters_path, "w", encoding="utf-8") as f:
                json.dump(self.local_data, f, indent=4, ensure_ascii=False)
                
            # Перерисовываем аннотации с новыми именами
            try:
                from core import annotate_frames
                annotate_frames(self.frames_dir)
                messagebox.showinfo("Успех", "Изменения сохранены и кадры обновлены.")
            except Exception as ae:
                messagebox.showwarning("Предупреждение", f"Изменения сохранены, но не удалось обновить кадры: {ae}")
                
            self.destroy()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить: {e}")

class FacesUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection Pipeline")
        self.root.geometry("820x600")
        self.root.configure(bg="#1e1e2e")
        self.root.minsize(600, 400)
        
        self.processing = False
        self.process = None
        self.last_frames_dir = None
        
        self._build_ui()
    
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#1e1e2e", foreground="#89b4fa", font=("Segoe UI", 14, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Run.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Review.TButton", font=("Segoe UI", 10, "bold"), foreground="#fab387")
        
        # Заголовок
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=16, pady=(16, 8))
        ttk.Label(header, text="🎭 Face Detection Pipeline", style="Title.TLabel").pack(side=tk.LEFT)
        
        # Путь к файлу/директории
        path_frame = ttk.Frame(self.root)
        path_frame.pack(fill=tk.X, padx=16, pady=4)
        
        ttk.Label(path_frame, text="Путь:").pack(side=tk.LEFT, padx=(0, 8))
        
        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(
            path_frame, textvariable=self.path_var,
            font=("Segoe UI", 10), bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4", relief=tk.FLAT, bd=4
        )
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        btn_file = ttk.Button(path_frame, text="📄 Файл", command=self._browse_file)
        btn_file.pack(side=tk.LEFT, padx=2)
        
        btn_dir = ttk.Button(path_frame, text="📁 Папка", command=self._browse_dir)
        btn_dir.pack(side=tk.LEFT, padx=2)
        
        # Интервал
        opts_frame = ttk.Frame(self.root)
        opts_frame.pack(fill=tk.X, padx=16, pady=4)
        
        ttk.Label(opts_frame, text="Интервал (сек):").pack(side=tk.LEFT, padx=(0, 8))
        self.interval_var = tk.StringVar(value="10.0")
        interval_entry = tk.Entry(
            opts_frame, textvariable=self.interval_var, width=8,
            font=("Segoe UI", 10), bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4", relief=tk.FLAT, bd=4
        )
        interval_entry.pack(side=tk.LEFT)
        
        # Кнопки управления
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=16, pady=8)
        
        self.run_btn = ttk.Button(btn_frame, text="▶  Запустить", style="Run.TButton", command=self._run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹  Остановить", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.review_btn = ttk.Button(btn_frame, text="🔍 Обзор", style="Review.TButton", command=self._open_review, state=tk.DISABLED)
        self.review_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.clear_btn = ttk.Button(btn_frame, text="🗑  Очистить лог", command=self._clear_log)
        self.clear_btn.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(btn_frame, textvariable=self.status_var, foreground="#a6adc8").pack(side=tk.RIGHT)
        
        # Консольный вывод
        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD,
            font=("Consolas", 9), bg="#11111b", fg="#a6adc8",
            insertbackground="#cdd6f4", relief=tk.FLAT, bd=8,
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видеофайлы", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v"), ("Все файлы", "*.*")]
        )
        if path:
            self.path_var.set(path)
    
    def _browse_dir(self):
        path = filedialog.askdirectory(title="Выберите директорию с видео")
        if path:
            self.path_var.set(path)
    
    def _log(self, text):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
    
    def _clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
    
    def _open_review(self):
        if self.last_frames_dir:
            ReviewDialog(self.root, self.last_frames_dir)

    def _collect_videos(self, path):
        p = Path(path)
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                return [str(p)]
            else:
                self._log(f"⚠ Файл '{p.name}' не является видеофайлом.\n")
                return []
        elif p.is_dir():
            videos = sorted([
                str(f) for f in p.rglob("*")
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            ])
            self._log(f"📂 Найдено видеофайлов в директории: {len(videos)}\n")
            return videos
        else:
            self._log(f"❌ Путь '{path}' не найден.\n")
            return []
    
    def _run(self):
        path = self.path_var.get().strip()
        if not path:
            self._log("❌ Укажите путь к файлу или директории.\n")
            return
        
        videos = self._collect_videos(path)
        if not videos:
            return
        
        self.processing = True
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.review_btn.configure(state=tk.DISABLED)
        
        thread = threading.Thread(target=self._process_videos, args=(videos,), daemon=True)
        thread.start()
    
    def _process_videos(self, videos):
        interval = self.interval_var.get().strip()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "faces.py")
        
        for idx, video_path in enumerate(videos, 1):
            if not self.processing:
                break
            
            video_name = Path(video_path).stem
            self.last_frames_dir = os.path.join(script_dir, "temp", video_name)

            self.root.after(0, lambda i=idx, t=len(videos), v=video_path: (
                self.status_var.set(f"Обработка {i}/{t}"),
                self._log(f"\n{'='*60}\n📹 [{i}/{t}] {Path(v).name}\n{'='*60}\n")
            ))
            
            cmd = [sys.executable, script_path, video_path, "-i", interval]
            
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                
                self.process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=script_dir, encoding="utf-8",
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                
                for line in self.process.stdout:
                    if not self.processing:
                        self.process.terminate()
                        break
                    self.root.after(0, lambda l=line: self._log(l))
                
                self.process.wait()
                
            except Exception as e:
                self.root.after(0, lambda e=e: self._log(f"❌ Ошибка: {e}\n"))
        
        self.root.after(0, self._on_done)
    
    def _stop(self):
        self.processing = False
        if self.process and self.process.poll() is None:
            self.process.terminate()
    
    def _on_done(self):
        self.processing = False
        self.process = None
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.review_btn.configure(state=tk.NORMAL)
        self.status_var.set("Готово")
        self._log("\n✅ Обработка завершена.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = FacesUI(root)
    root.mainloop()

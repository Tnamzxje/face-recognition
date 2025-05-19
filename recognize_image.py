import face_recognition_api
import cv2
import os
import pickle
import numpy as np
from tkinter import Tk, Button, Label, filedialog, Canvas, NW, LEFT, RIGHT, messagebox
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Nhận diện khuôn mặt từ ảnh")
        master.geometry("850x350")
        master.resizable(False, False)

        self.btn_select = Button(master, text="Chọn ảnh", command=self.select_images, font=("Arial", 11))
        self.btn_select.pack(pady=5)

        self.canvas_goc = Canvas(master, width=400, height=250, bg='gray')
        self.canvas_goc.pack(side=LEFT, padx=5, pady=5)
        self.canvas_kq = Canvas(master, width=400, height=250, bg='gray')
        self.canvas_kq.pack(side=RIGHT, padx=5, pady=5)

        self.label_status = Label(master, text="", font=("Arial", 10), fg="red")
        self.label_status.pack(pady=2)

        self.btn_next = Button(master, text="Tiếp", command=self.next_image, font=("Arial", 11), state='disabled')
        self.btn_next.pack(side=LEFT, padx=10)
        self.btn_save = Button(master, text="Lưu kết quả", command=self.save_results, font=("Arial", 11), state='disabled')
        self.btn_save.pack(side=LEFT, padx=10)
        self.btn_exit = Button(master, text="Thoát", command=master.quit, font=("Arial", 11))
        self.btn_exit.pack(side=RIGHT, padx=10)

        # Load a
        fname = 'classifier.pkl'
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                (self.le, self.clf) = pickle.load(f)
        else:
            self.label_status.config(text=f"Không tìm thấy file {fname}")
            self.btn_select.config(state='disabled')

        self.images = []  # List of file paths
        self.results = [] # List of PIL images (nhận diện)
        self.index = 0

    def select_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Chọn các file ảnh để nhận diện",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_paths:
            self.label_status.config(text="Không chọn file nào!")
            return
        self.images = list(file_paths)
        self.results = []
        self.index = 0
        self.btn_next.config(state='normal' if len(self.images) > 1 else 'disabled')
        self.btn_save.config(state='normal')
        self.show_image()

    def show_image(self):
        if not self.images:
            print("[DEBUG] Không có ảnh nào trong danh sách.")
            return
        file_path = self.images[self.index]
        print(f"[DEBUG] Đang xử lý file: {file_path}")
        self.label_status.config(text=f"Đang nhận diện: {os.path.basename(file_path)} ({self.index+1}/{len(self.images)})")
        self.master.update()
        try:
            image = face_recognition_api.load_image_file(file_path)
            print(f"[DEBUG] Đọc ảnh xong: shape={image.shape}, dtype={image.dtype}")
            image_rgb = image.copy()
            face_locations = face_recognition_api.face_locations(image_rgb)
            face_encodings = face_recognition_api.face_encodings(image_rgb, face_locations)
            print(f"[DEBUG] Số khuôn mặt phát hiện: {len(face_locations)}")
            predictions = []
            if len(face_encodings) > 0:
                closest_distances = self.clf.kneighbors(face_encodings, n_neighbors=1)
                is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]
                preds = self.clf.predict(face_encodings)
                predictions = [
                    (self.le.inverse_transform([int(pred)])[0].title() if rec else "Unknown", loc)
                    for pred, loc, rec in zip(preds, face_locations, is_recognized)
                ]
            # Vẽ kết quả lên ảnh
            image_result = image.copy()
            for name, (top, right, bottom, left) in predictions:
                cv2.rectangle(image_result, (left, top), (right, bottom), (0, 0, 255), 2)
                # Rút gọn tên nếu quá dài
                display_name = name
                if len(display_name) > 18:
                    display_name = display_name[:15] + '...'
                # Vẽ nền trắng cho text
                cv2.rectangle(image_result, (left, bottom - 32), (right, bottom), (255, 255, 255), cv2.FILLED)
                # Font vừa phải, màu chữ đỏ
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image_result, display_name, (left + 8, bottom - 10), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            # Hiển thị ảnh gốc
            img_pil_goc = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_pil_goc = img_pil_goc.resize((400, 250), Image.Resampling.LANCZOS)
            self.img_tk_goc = ImageTk.PhotoImage(img_pil_goc)
            self.canvas_goc.create_image(0, 0, anchor=NW, image=self.img_tk_goc)
            print("[DEBUG] Đã hiển thị ảnh gốc lên canvas.")
            # Hiển thị ảnh kết quả
            img_pil_kq = Image.fromarray(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
            img_pil_kq = img_pil_kq.resize((400, 250), Image.Resampling.LANCZOS)
            self.img_tk_kq = ImageTk.PhotoImage(img_pil_kq)
            self.canvas_kq.create_image(0, 0, anchor=NW, image=self.img_tk_kq)
            print("[DEBUG] Đã hiển thị ảnh kết quả lên canvas.")
            # Lưu kết quả vào list
            if len(self.results) <= self.index:
                self.results.append(img_pil_kq.copy())
            else:
                self.results[self.index] = img_pil_kq.copy()
            if predictions:
                self.label_status.config(text=f"{os.path.basename(file_path)}: {len(predictions)} khuôn mặt.")
            else:
                self.label_status.config(text=f"{os.path.basename(file_path)}: Không phát hiện khuôn mặt nào!")
        except Exception as e:
            self.label_status.config(text=f"Lỗi: {e}")
            print(f"[ERROR] {e}")

    def next_image(self):
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self.show_image()

    def save_results(self):
        if not self.results:
            messagebox.showinfo("Thông báo", "Chưa có ảnh kết quả để lưu!")
            return
        save_dir = filedialog.askdirectory(title="Chọn thư mục để lưu ảnh kết quả")
        if not save_dir:
            return
        for idx, img in enumerate(self.results):
            file_name = os.path.basename(self.images[idx])
            save_path = os.path.join(save_dir, f"result_{file_name}")
            img.save(save_path)
        messagebox.showinfo("Thông báo", f"Đã lưu {len(self.results)} ảnh kết quả vào {save_dir}")

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop() 
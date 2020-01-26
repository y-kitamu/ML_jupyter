"""
"""
import os
import io
import glob
import shutil
import tkinter as tk
import tkinter.filedialog

from PIL import Image, ImageTk


class LabelingAppFrame(tk.Frame):
    FILE_TYPE = [("", "*.jpg;*.png")]
    MAX_LIST_LENGTH = 20
    
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        
        self.file_list = []
        self.file_idx = 0
        self._frame = None
        self.frame_file_select = None
        self.frame_dst_directory = None
        
        self.root_dir = tkinter.filedialog.askdirectory(initialdir=os.path.dirname(__file__))
        print("selected directory : ", self.root_dir)
        
        self.create_file_selection_widgets()

    def _switch_frame(self, new_frame):
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

    def create_file_selection_widgets(self):
        root = tk.Frame(self)
        label_regex = tk.Label(root, text="ファイルを指定")
        label_regex.grid(row=0, column=0)

        self.text_regex = tk.Entry(root, justify="left")
        self.text_regex.grid(row=1, column=0)
        self.text_regex.bind("<KeyRelease>", self.update_file_list)

        xscrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
        xscrollbar.grid(row=3, column=0, sticky=tk.N + tk.S + tk.E + tk.W)

        yscrollbar = tk.Scrollbar(root, orient=tk.VERTICAL)
        yscrollbar.grid(row=2, column=1, sticky=tk.N + tk.S + tk.E + tk.W)

        self.text_file_list = tk.Text(
            root, wrap="none", height=20, bg="white",
            xscrollcommand=xscrollbar.set,
            yscrollcommand=yscrollbar.set)
        self.text_file_list.grid(row=2, column=0)

        xscrollbar.config(command=self.text_file_list.xview)
        yscrollbar.config(command=self.text_file_list.yview)

        button_next = tk.Button(root, text="Next",
                                     command=self.create_dst_directory_widgets)
        button_next.grid(row=4, column=0)

        self._switch_frame(root)

    def create_dst_directory_widgets(self):
        root = tk.Frame(self)
        
        label_select_label_dir = tk.Label(
            root, text="ラベル名に対応するディレクトリ階層を選択してください")
        label_select_label_dir.grid(row=0, column=0)
        
        root_dir_idx = len(self.root_dir.split("/"))
        subdirs = self.file_list[0].split("/")[root_dir_idx:]
        self.is_select_label_dirs = [tkinter.BooleanVar() for _ in subdirs]
        for idx, dirname in enumerate(subdirs):
            chk = tk.Checkbutton(root, text=dirname, variable=self.is_select_label_dirs[idx])
            chk.grid(row=0, column=idx + 1)
        
        chk_save_same_dir_chk = tk.Checkbutton(root, text="階層構造を維持する")
        chk_save_same_dir_chk.grid(row=1, column=0, sticky=tk.W)

        chk_save_custom_dir = tk.Checkbutton(root, text="一つにまとめる")
        chk_save_custom_dir.grid(row=2, column=0, sticky=tk.W)

        button_prev = tk.Button(root, text="Prev",
                                         command=self.create_file_selection_widgets)
        button_prev.grid(row=3, column=0)
        button_det = tk.Button(root, text="Determine",
                               command=self.create_image_labeling_widgets)
        button_det.grid(row=3, column=1)
        self._switch_frame(root)

    def extract_labelnames(self):
        root_dir_idx = len(self.root_dir.split("/"))
        for idx, chk in enumerate(self.is_select_label_dirs):
            if chk.get():
                self.label_dir_idx = root_dir_idx + idx
                break
        else:
            print("Failed to extract label name. No label directory is checked.")
            self.create_file_selection_widgets()
            return
        self.label_names = sorted(set(
            [fname.split("/")[self.label_dir_idx] for fname in self.file_list]))

    def create_image_labeling_widgets(self):
        root = tk.Frame(self)
        self._switch_frame(root)
        root.focus_set()
        self.extract_labelnames()

        self.button_prev = tk.Button(root, text="Prev",
                                     command=lambda: self.show_image(self.file_idx - 1))
        self.button_prev.grid(row=0, column=0)
        self.button_next = tk.Button(root, text="Next",
                                     command=lambda: self.show_image(self.file_idx + 1))
        self.button_next.grid(row=0, column=1)

        self.var_filename = tk.StringVar()
        label_filename = tk.Label(root, textvariable=self.var_filename)
        label_filename.grid(row=2, column=0, columnspan=len(self.label_names))

        self.button_labels = [tk.Button(root, text="{} ({})".format(label, idx))
                         for idx, label in enumerate(self.label_names)]

        self.image_pannel = tk.Label(root)
        self.show_image(self.file_idx)

    def show_image(self, idx):
        # print("idx : {}".format(idx))
        if idx <= 0:
            self.button_prev.config(state="disabled")
        else:
            self.button_prev.config(state="normal")

        if idx >= len(self.file_list) - 1:
            self.button_next.config(state="disabled")
        else:
            self.button_next.config(state="normal")
        
        self.image = Image.open(self.file_list[idx])
        self.image = ImageTk.PhotoImage(self.image)

        self.image_pannel.configure(image=self.image)
        self.image_pannel.grid(row=1, column=0, columnspan=len(self.label_names))
        self.file_idx = idx

        for idx, button_label in enumerate(self.button_labels):
            label_func = self.labeling(idx, self.file_idx)
            button_label.grid(row=3, column=idx)
            root.bind("<KeyRelease-{}>".format(idx), label_func)

        self.var_filename.set(self.file_list[idx])

    def labeling(self, label_idx, file_idx):
        def _labeling(event=None):
            filename = self.file_list[file_idx]
            dirs = filename.split("/")
            dirs[self.label_dir_idx] = self.label_names[label_idx]
            new_filename = os.path.join(filename[0], "/".join(dirs))

            shutil.move(filename, new_filename)
            print("move from {} to {}".format(filename, new_filename))
            self.file_list[file_idx] = new_filename
            self.show_image(file_idx + 1)
        return _labeling

    def update_file_list(self, event):
        path = self.text_regex.get()
        self.file_list = glob.glob(os.path.join(self.root_dir, path))

        with io.StringIO() as output:
            for idx, fname in enumerate(self.file_list):
                output.write("{}\n".format(fname))
                if idx > LabelingAppFrame.MAX_LIST_LENGTH:
                    output.write("etc...\n")
                    break

            self.text_file_list.delete("1.0", "end")  # if you want to remove the old data
            self.text_file_list.insert(tk.END, output.getvalue())


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingAppFrame(master=root)
    app.mainloop()

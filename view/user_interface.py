from controller.frame_controller import FrameController
from controller.identity_controller import IdentityController
from model.identity_model import Identity
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Tk, Canvas, Menu, Toplevel, Label, Entry, Button
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

PAD_X = 4
PAD_Y = 4
BUTTON_W = 8
BUTTON_H = 1


class UserInterface:
    def __init__(self):
        self.running = True
        self.camera = cv2.VideoCapture(0)
        camera_props = self.get_camera_props()
        self.frame_controller = FrameController(frame_proc_freq=camera_props[0])
        self.identity_controller = IdentityController()

        self.root = Tk()
        self.root.title('Main window')
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self.on_closing)
        self.canvas = Canvas(self.root, width=camera_props[1], height=camera_props[2])
        self.canvas.grid()
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        self.file_menu = Menu(self.menu)
        self.menu.add_cascade(label='Options', menu=self.file_menu)
        self.file_menu.add_command(label='Manage identities', command=self.show_identities_window)
        self.file_menu.add_command(label='Clear present face list', command=self.clear_face_list)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='Exit', command=self.quit)

        self.tree = None
        self.window = None
        self.name_entry = None
        self.surname_entry = None
        self.add_button = None
        self.browse_button = None
        self.edit_button = None
        self.delete_button = None
        self.img_path_label = None

    def run(self):
        while self.running:
            # img saved into variable to prevent deleting object from memory by garbage collector
            img = self.capture_image()
            self.root.update_idletasks()
            self.root.update()

    def stop(self):
        self.frame_controller.stop()

    def on_closing(self):
        self.running = False

    def quit(self):
        self.running = False
        self.root.quit()

    def set_file_path(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.img_path_label['text'] = filename

    def capture_image(self):
        check, frame = self.camera.read()
        processed_frame = self.frame_controller.process_frame(frame)
        img = ImageTk.PhotoImage(image=Image.fromarray(processed_frame))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        return img

    def get_camera_props(self):
        frame_proc_freq = max(int(self.camera.get(cv2.CAP_PROP_FPS) // 10), 1)
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return frame_proc_freq, width, height

    def clear_face_list(self):
        self.frame_controller.clear_face_list()

    def tree_item_clicked(self, event):
        selected = self.tree.focus()
        if selected:
            item = self.tree.item(selected)
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, item['values'][0])
            self.surname_entry.delete(0, tk.END)
            self.surname_entry.insert(0, item['values'][1])
            self.edit_button['state'] = tk.NORMAL
            self.delete_button['state'] = tk.NORMAL
        else:
            self.clear_entries()

    def add_identity(self):
        name = self.name_entry.get()
        surname = self.surname_entry.get()
        identity = Identity(name=name, surname=surname)
        try:
            self.identity_controller.add_identity(identity, self.img_path_label['text'])
        except ValueError as e:
            messagebox.showwarning('Warning', str(e))
        self.refresh_table()

    def edit_identity(self):
        selected = self.tree.focus()
        if selected:
            name = self.name_entry.get()
            surname = self.surname_entry.get()
            item = self.tree.item(selected)
            identity = Identity(name=name, surname=surname, identity_id=int(item['text']))
            try:
                self.identity_controller.edit_identity(identity, self.img_path_label['text'])
            except ValueError as e:
                messagebox.showwarning('Warning', str(e))
            self.refresh_table()

    def delete_identity(self):
        selected = self.tree.focus()
        if selected:
            item = self.tree.item(selected)
            result = messagebox.askquestion('Delete identity', 'Are you sure you want to delete this identity?')
            if result == 'yes':
                self.identity_controller.delete_identity(int(item['text']))
            self.refresh_table()

    def show_identities_window(self):
        self.window = Toplevel(self.root)
        self.window.title('Identity manager')
        self.window.focus_set()
        self.window.grab_set()

        Label(self.window, text='Name').grid(row=0, column=0, sticky=tk.W, padx=PAD_X, pady=PAD_Y)
        Label(self.window, text='Surname').grid(row=1, column=0, sticky=tk.W, padx=PAD_X, pady=PAD_Y)
        Label(self.window, text='Photo').grid(row=2, column=0, sticky=tk.W, padx=PAD_X, pady=PAD_Y)

        self.name_entry = Entry(self.window)
        self.name_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W+tk.E, padx=PAD_X, pady=PAD_Y)
        self.surname_entry = Entry(self.window)
        self.surname_entry.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E, padx=PAD_X, pady=PAD_Y)
        self.browse_button = Button(self.window, text='Browse', command=self.set_file_path, height=BUTTON_H, width=BUTTON_W)
        self.browse_button.grid(row=2, column=1, sticky=tk.W, padx=PAD_X, pady=PAD_Y)
        self.img_path_label = Label(self.window, text='')
        self.img_path_label.grid(row=2, column=2, sticky=tk.W, padx=PAD_X, pady=PAD_Y)

        self.add_button = Button(self.window, text='Add', command=self.add_identity, height=BUTTON_H, width=BUTTON_W)
        self.add_button.grid(row=0, column=3, sticky=tk.E, padx=PAD_X, pady=PAD_Y)
        self.edit_button = Button(self.window, text='Edit', command=self.edit_identity, height=BUTTON_H, width=BUTTON_W, state=tk.DISABLED)
        self.edit_button.grid(row=1, column=3, sticky=tk.E, padx=PAD_X, pady=PAD_Y)
        self.delete_button = Button(self.window, text='Delete', command=self.delete_identity, height=BUTTON_H, width=BUTTON_W, state=tk.DISABLED)
        self.delete_button.grid(row=2, column=3, sticky=tk.E, padx=PAD_X, pady=PAD_Y)
        self.show_table()

    def show_table(self):
        self.tree = ttk.Treeview(self.window, columns=('Name', 'Surname'))
        self.tree.heading('#0', text='ID')
        self.tree.heading('#1', text='Name')
        self.tree.heading('#2', text='Surname')
        self.tree.column('#0', stretch=tk.YES)
        self.tree.column('#1', stretch=tk.YES)
        self.tree.column('#2', stretch=tk.YES)
        self.tree.grid(columnspan=4)
        self.tree.bind('<ButtonRelease-1>', self.tree_item_clicked)
        self.refresh_table()

    def refresh_table(self):
        self.clear_entries()
        for row in self.tree.get_children():
            self.tree.delete(row)
        for identity in self.identity_controller.get_identities().values():
            self.tree.insert('', tk.END, text=str(identity.identity_id), values=(identity.name, identity.surname))

    def clear_entries(self):
        self.name_entry.delete(0, tk.END)
        self.surname_entry.delete(0, tk.END)
        self.edit_button['state'] = tk.DISABLED
        self.delete_button['state'] = tk.DISABLED
        self.img_path_label['text'] = ''

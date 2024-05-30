from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox

class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__config()
        self.__win()
        self.messagebox = messagebox
        self.tk_frame_ModelConfig = self.__tk_frame_ModelConfig(self)
        self.tk_label_SelectNetModel = self.__tk_label_SelectNetModel(self.tk_frame_ModelConfig) 
        self.tk_select_box_SelectNetModel = self.__tk_select_box_SelectNetModel(self.tk_frame_ModelConfig) 
        self.tk_label_ModelPath = self.__tk_label_ModelPath(self.tk_frame_ModelConfig) 
        self.tk_input_ModelPath = self.__tk_input_ModelPath(self.tk_frame_ModelConfig) 
        self.tk_button_Start = self.__tk_button_Start(self.tk_frame_ModelConfig) 
        self.tk_label_SetEpoch = self.__tk_label_SetEpoch(self.tk_frame_ModelConfig) 
        self.tk_scale_SetEpoch = self.__tk_scale_SetEpoch(self.tk_frame_ModelConfig) 
        self.tk_label_SelectDataset = self.__tk_label_SelectDataset(self.tk_frame_ModelConfig) 
        self.tk_select_box_SelectDataset = self.__tk_select_box_SelectDataset(self.tk_frame_ModelConfig) 
        self.tk_input_DatasetPath = self.__tk_input_DatasetPath(self.tk_frame_ModelConfig) 
        self.tk_check_button_EarlyStop = self.__tk_check_button_EarlyStop(self.tk_frame_ModelConfig) 
        self.tk_label_frame_Output = self.__tk_label_frame_Output(self)
        self.tk_label_ShowBackDoorImage = self.__tk_label_ShowBackDoorImage(self.tk_label_frame_Output) 
        self.tk_label_BackDoorLabel = self.__tk_label_BackDoorLabel(self.tk_label_frame_Output) 
        self.tk_progressbar_Process = self.__tk_progressbar_Process(self.tk_label_frame_Output) 
        self.tk_label_Process = self.__tk_label_Process(self.tk_label_frame_Output) 
        self.tk_label_ShowDatasetImage = self.__tk_label_ShowDatasetImage(self.tk_label_frame_Output) 
        self.tk_label_DatasetImage = self.__tk_label_DatasetImage(self.tk_label_frame_Output) 
        self.tk_button_DatasetPrvious = self.__tk_button_DatasetPrvious(self.tk_label_frame_Output) 
        self.tk_button_DatasetNext = self.__tk_button_DatasetNext(self.tk_label_frame_Output) 
        self.tk_button_BackDoorImgPrvious = self.__tk_button_BackDoorImgPrvious(self.tk_label_frame_Output) 
        self.tk_button_BackDoorImgNext = self.__tk_button_BackDoorImgNext(self.tk_label_frame_Output) 
        self.__set_defaults()
        self.tk_label_ShowDatasetImage.winfo_width()

    def __config(self):
        self.IsEarlyStop = BooleanVar(value=True)  # 是否提前停止
        self.Epochs = IntVar(value=10)  # 最大迭代次数
        self.progress_value = DoubleVar(value=0)  # 进度条值
    
    def __win(self):
        self.title("深度神经网络后门检测系统")
        width = 800
        height = 550
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        self.minsize(width=width, height=height)

    def __set_defaults(self):
        self.tk_select_box_SelectNetModel.current(0)
        self.tk_select_box_SelectDataset.current(0)
        self.tk_input_DatasetPath.place_forget()

    def scrollbar_autohide(self, vbar, hbar, widget):
        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)

        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)

        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())

    def v_scrollbar(self, vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')

    def h_scrollbar(self, hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')

    def create_bar(self, master, widget, is_vbar, is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def __tk_frame_ModelConfig(self, parent):
        frame = Frame(parent,)
        frame.place(relx=0.0000, rely=0.0000, relwidth=1.0000, relheight=0.2182)
        return frame

    def __tk_label_SelectNetModel(self, parent):
        label = Label(parent, text="网络模型：", anchor="center", )
        label.place(relx=0.0125, rely=0.0000, relwidth=0.0750, relheight=0.2500)
        return label

    def __tk_select_box_SelectNetModel(self, parent):
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("自动选择", "MNIST", "FashionMNIST", "CIFAR10", "自定义模型")
        cb.place(relx=0.0950, rely=0.0000, relwidth=0.1875, relheight=0.2500)
        return cb

    def __tk_label_ModelPath(self, parent):
        label = Label(parent, text="待检测模型路径：", anchor="center", )
        label.place(relx=0.2913, rely=0.0000, relwidth=0.1250, relheight=0.2500)
        return label

    def __tk_input_ModelPath(self, parent):
        ipt = Entry(parent, )
        ipt.place(relx=0.4200, rely=0.0000, relwidth=0.4412, relheight=0.2500)
        return ipt

    def __tk_button_Start(self, parent):
        btn = Button(parent, text="开始检测", takefocus=False, )
        btn.place(relx=0.9000, rely=0.0000, relwidth=0.1000, relheight=0.2500)
        return btn

    def __tk_label_SetEpoch(self, parent):
        label = Label(parent, text=f"最大检测轮数：{self.Epochs.get()}", anchor="center", )
        label.place(relx=0.0125, rely=0.3333, relwidth=0.1500, relheight=0.2500)
        return label

    def __tk_scale_SetEpoch(self, parent):
        scale = Scale(parent, orient=HORIZONTAL, from_=3, to=50, length=200, variable=self.Epochs)
        scale.place(relx=0.1750, rely=0.3333, relwidth=0.1875, relheight=0.2500)
        return scale

    def __tk_label_SelectDataset(self, parent):
        label = Label(parent, text="选择数据集：", anchor="center", )
        label.place(relx=0.3750, rely=0.3333, relwidth=0.1000, relheight=0.2500)
        return label

    def __tk_select_box_SelectDataset(self, parent):
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("自动加载", "MNIST", "FashionMNIST", "CIFAR10", "其他数据集")
        cb.place(relx=0.4875, rely=0.3333, relwidth=0.1875, relheight=0.2500)
        return cb

    def __tk_input_DatasetPath(self, parent):
        ipt = Entry(parent, )
        return ipt

    def __tk_check_button_EarlyStop(self, parent):
        cb = Checkbutton(parent, text="是否早停", variable=self.IsEarlyStop, )
        cb.place(relx=0.0125, rely=0.7083, relwidth=0.1000, relheight=0.2500)
        return cb

    def __tk_label_frame_Output(self, parent):
        frame = LabelFrame(parent, text="检测输出", )
        frame.place(relx=0.0000, rely=0.2364, relwidth=1.0000, relheight=0.7636)
        return frame

    def __tk_label_ShowBackDoorImage(self, parent):
        label = Label(parent, text="", anchor="center", )
        label.place(relx=0.5500, rely=0.0952, relwidth=0.4000, relheight=0.7619)
        return label

    def __tk_label_BackDoorLabel(self, parent):
        label = Label(parent, text="后门标签", anchor="center", )
        label.place(relx=0.6875, rely=0.0000, relwidth=0.1250, relheight=0.0714)
        return label

    def __tk_progressbar_Process(self, parent):  # 进度条
        progressbar = Progressbar(parent, orient=HORIZONTAL, variable=self.progress_value)
        progressbar.place(relx=0.0938, rely=0.9048, relwidth=0.9000, relheight=0.0238)
        return progressbar

    def __tk_label_Process(self, parent):
        label = Label(parent, text="进度", anchor="center", )
        label.place(relx=0.0063, rely=0.8810, relwidth=0.0650, relheight=0.0714)
        return label

    def __tk_label_ShowDatasetImage(self, parent):
        label = Label(parent, text="", anchor="center", )
        label.place(relx=0.0500, rely=0.0952, relwidth=0.4000, relheight=0.7619)
        return label

    def __tk_label_DatasetImage(self, parent):
        label = Label(parent, text="数据集图片", anchor="center", )
        label.place(relx=0.2062, rely=0.0000, relwidth=0.0875, relheight=0.0714)
        return label

    def __tk_button_DatasetPrvious(self, parent):
        btn = Button(parent, text="上一张", takefocus=False, )
        btn.place(relx=0.1250, rely=0.0000, relwidth=0.0625, relheight=0.0714)
        return btn

    def __tk_button_DatasetNext(self, parent):
        btn = Button(parent, text="下一张", takefocus=False, )
        btn.place(relx=0.3125, rely=0.0000, relwidth=0.0625, relheight=0.0714)
        return btn

    def __tk_button_BackDoorImgPrvious(self, parent):
        btn = Button(parent, text="上一张", takefocus=False, )
        btn.place(relx=0.5875, rely=0.0000, relwidth=0.0625, relheight=0.0714)
        return btn

    def __tk_button_BackDoorImgNext(self, parent):
        btn = Button(parent, text="下一张", takefocus=False, )
        btn.place(relx=0.8500, rely=0.0000, relwidth=0.0625, relheight=0.0714)
        return btn
    

class Win(WinGUI):
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)

    def __event_bind(self):
        self.tk_button_Start.bind('<Button-1>', self.ctl.StartDetect, )
        self.tk_scale_SetEpoch.bind("<Motion>", self.ctl.update_epoch_label)
        self.tk_select_box_SelectNetModel.bind("<<ComboboxSelected>>", self.ctl.on_net_model_select)
        self.tk_select_box_SelectDataset.bind("<<ComboboxSelected>>", self.ctl.on_dataset_select)
        self.tk_button_DatasetPrvious.bind('<Button-1>', self.ctl.on_dataset_prvious)
        self.tk_button_DatasetNext.bind('<Button-1>', self.ctl.on_dataset_next)
        self.tk_button_BackDoorImgPrvious.bind('<Button-1>', self.ctl.on_backdoor_img_prvious)
        self.tk_button_BackDoorImgNext.bind('<Button-1>', self.ctl.on_backdoor_img_next)

    def __style_config(self):
        pass

if __name__ == "__main__":
    win = WinGUI()
    win.mainloop()

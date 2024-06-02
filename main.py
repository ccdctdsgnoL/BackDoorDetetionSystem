# 导入布局文件
from utils.ui import Win as MainWin
# 导入窗口控制器
from utils.control import Controller as MainUIController


# 将窗口控制器 传递给UI
app = MainWin(MainUIController())

if __name__ == "__main__":
    # 启动
    app.mainloop()
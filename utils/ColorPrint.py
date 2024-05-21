import colorama
from datetime import datetime
colorama.init(autoreset=True)

def colorPrint(text, color):
    
    color_dic = {
        'red': "\033[31m",
        'green': "\033[32m",
        'yellow': "\033[33m",  # 黄色
        'blue': "\033[34m",
        'magenta': "\033[35m",  # 品红色
        'cyan': "\033[36m",    # 青色
        'white': "\033[37m",    # 白色
        'black': "\033[30m",     # 黑色
        'default': "\033[0m"
    }
    # 彩色打印
    color = color.lower()
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{formatted_time}]{text}"
    if color in color_dic:
        print(f"{color_dic[color]}{text}{color_dic['default']}")
    else:
        print(text)


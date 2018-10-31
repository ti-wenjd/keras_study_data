import tkinter as tk

window = tk.Tk()
window.title('my window')
window.geometry('200x100')



l = tk.Label(window,
    text='OMG! this is TK!',    # 标签的文字
    bg='yellow',     # 背景颜色
    font=('Yahei', 12),     # 字体和字体大小
    width=100, height=80  # 标签长宽
    )
l.pack()

window.mainloop()
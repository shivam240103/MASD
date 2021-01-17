from files import *


def loading():
    root = tk.Tk()
    root.iconbitmap(default='IMAGES/f.ico')

    root.image = tk.PhotoImage(file='IMAGES/front.gif')
    l = tk.Label(root, image=root.image, bg='white')
    root.overrideredirect(True) 
    root.geometry("+250+50")

    root.wm_attributes("-topmost", True)
    root.wm_attributes("-disabled", True)
    root.wm_attributes("-transparentcolor", "white")
    l.pack()
    l.after(500, lambda: l.destroy())
    root.after(500, lambda: root.destroy())  
    l.mainloop()

def data_load(win):
    load = cv2.imread('IMAGES/home_background.png', 1)
    x = cv2.cvtColor(load, cv2.COLOR_BGR2RGBA)
    load = Image.fromarray(x)
    load = load.resize((int(800), int(450)), Image.ANTIALIAS)
    c = ImageTk.PhotoImage(load)
    img = tk.Label(image=c)
    img.image = c
    img.place(x=-1, y=0)

    
  
    def store():
        win.destroy()
        main.detection()
    

    style = ttk.Style()
    style.configure('TButton', font = ('comic sans ms', 20, 'bold'), borderwidth = '10',foreground='red')
    b = ttk.Button(win, text='START', style='TButton', command=store)
    b.place(x=15, y=200, width=200, height=50)


def wen():
    t = tk.Tk()
    s = t.winfo_screenwidth()
    t.destroy()
    for i in range(0,3):
        loading()
    win = tk.Tk()
    win.iconbitmap(default='IMAGES/f.ico')
    option_win = data_load(win)
    win.config(background='black')
    win.title('MASD' + ' 1.0.0')
    win.geometry("750x450")
    win.mainloop()


Thread(target=wen).start()
Thread(target=start_on).start()
















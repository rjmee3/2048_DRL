#!/usr/bin/env python3
import tkinter as tk

window = tk.Tk()
greeting = tk.Label(
    text="Hello World!",
    foreground="white",
    background="black" 
    )
greeting.pack()
window.minsize(400, 400)
window.mainloop()
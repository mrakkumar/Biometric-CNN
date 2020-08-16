# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 22:12:50 2017

@author: Mukundbj
"""

from Tkinter import *
import CNN_test_hardware as user
from PIL import ImageTk, Image
import tkFileDialog
import tkMessageBox

top = Tk()

def runCallBack():
    pic=E1.get()
    acc=user.run(pic)
    img = ImageTk.PhotoImage(Image.open(pic))
    #panel = Label(bottomframe, image = img)
    panel.configure(image=img)
    panel.image=img
    #panel.pack_forget()
    #panel.pack(side = "bottom", fill = "both", expand = "yes")
    tkMessageBox.showinfo( "Final Results", str(acc))
    
    
def openCallBack():
    pic=tkFileDialog.askopenfilename()
    last=len(E1.get())
    E1.delete(0,last+1)
    E1.insert(0,pic)


frame = Frame(top)
frame.pack()
bottomframe = Frame(top)
bottomframe.pack( side = BOTTOM )
L1 = Label(frame, text="File Path")
L1.pack( side = LEFT)
E1 = Entry(frame, bd =5)
E1.pack(side = LEFT)

img = ImageTk.PhotoImage(Image.open('C:/Users/Mukundbj/Desktop/off.jpg'))
panel = Label(bottomframe, image = img)
panel.image=img
panel.pack(side = "bottom", fill = "both", expand = "yes")
    
B1= Button(frame, text ="Browse", command = openCallBack)
B1.pack( side = LEFT)
B2 = Button(bottomframe, text ="Verify", command = runCallBack)
B2.pack(side=BOTTOM)

top.mainloop()
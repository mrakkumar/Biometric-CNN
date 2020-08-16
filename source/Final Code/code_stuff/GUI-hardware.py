#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:37:04 2017

@author: Mukundbj
"""
import Tkinter as tk
#from PIL import ImageTk, Image
import tkFileDialog
import tkMessageBox


from picamera import PiCamera
from time import sleep



class GUI:
    def camCallBack(self):
        num=self.ID_entry.get()
        path=self.Browse_entry.get()
        path=path+"/"+num
        try:
            self.cam(path,5)
        except:
            tkMessageBox.showinfo( "Biometric","Error Occured. Please try again")
        tkMessageBox.showinfo( "Biometric","Capture Successful")
    
    def openCallBack(self):
        path=tkFileDialog.askdirectory()
        last=len(self.Browse_entry.get())
        self.Browse_entry.delete(0,last+1)
        self.Browse_entry.insert(0,path)
        
    def cam(self,path,rest):
        camera = PiCamera()
        camera.rotation = 180
        camera.start_preview()
        camera.awb_mode = 'fluorescent'
        sleep(rest)
        for i in range(10):
            sleep(1)
            s=path+str(i)+".jpg"
            camera.capture(s)
        camera.stop_preview()
        camera.close()
        
    def createWidgets(self):
        
        self.ID_label=tk.Label(text="User ID")
        self.Browse_label=tk.Label(text="Choose Directory")
        self.ID_entry=tk.Entry()
        self.Browse_entry=tk.Entry()
        self.Capture_button=tk.Button(text ="Capture", command = self.camCallBack)
        self.Browse_button=tk.Button(text ="Browse", command = self.openCallBack)
        
        self.ID_label.grid(row=1,column=1,padx=(20,15),pady=(20,10))
        self.ID_entry.grid(row=1,column=2,padx=(15,30),pady=(20,10))
        
        self.Browse_label.grid(row=2,column=1,padx=(20,10),pady=(20,30))
        self.Browse_entry.grid(row=2,column=2,padx=(10,20),pady=(20,30))
        self.Browse_button.grid(row=2,column=3,padx=(10,20),pady=(20,30))
        self.Capture_button.grid(row=3,column=2,padx=(10,20),pady=(20,30))

    def __init__(self, master=None):
        tk.Frame(master)
        #frame.pack()
        self.createWidgets()

root = tk.Tk()
app = GUI(master=root)
root.mainloop()
root.destroy()
    

#frame = Frame(top)
#frame.pack()
#
#bottomframe = Frame(top)
#bottomframe.pack( side = BOTTOM )
#
#middleframe = Frame(top)
#middleframe.pack( side = BOTTOM )
#
#
#L1 = Label(frame, text="User ID")
#L1.pack( side = LEFT)
#
#ID = Entry(frame, bd =5)
#ID.pack(side = LEFT)
#
#B1= Button(middleframe, text ="Capture", command = camCallBack)
#B1.pack( side = BOTTOM)
#
#L2 = Label(bottomframe, text="Choose folder")
#L2.pack( side = LEFT)
#
#PATH = Entry(bottomframe, bd =5)
#PATH.pack(side = LEFT)
#
#B2= Button(bottomframe, text ="Browse", command = openCallBack)
#B2.pack( side = LEFT)
#app=GUI()
#app.root.mainloop()
#top.mainloop()
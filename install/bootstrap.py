#!/usr/bin/env python
#
#    This should only be run ONCE! It checks for the existence of
#  BitKeeper and then does the bk clone bk://sidl.bkbits.net/BuildSystem
#  and then calls the installer in BuildSystem/install
#
import user
import curses
import curses.textpad
import os
import os.path
import sys
import commands

def CenterAddStr(stdscr,my,text,attr = 0):
  (y,x) = stdscr.getmaxyx()
  x = (x - len(text))/2
  stdscr.addstr(my,x,text,attr)

def ConvertReturnToExit(key):
  if key == ord('\n'): return 7
  return key

def CenterGetStr(stdscr,my,text = ''):
  (y,x) = stdscr.getmaxyx()
  width = x - 10
  x = int((x - width)/2)
  curses.textpad.rectangle(stdscr,my,x-1,my+2,x+width)
  subscr  = curses.newwin(1,width,my+1,x)
  subscr.addstr(0,0,text)
  stdscr.refresh()
  textobj = curses.textpad.Textbox(subscr)
  textobj.stripspaces = 1
  text = textobj.edit(ConvertReturnToExit)
  return text

class CursesInstall:
  def __init__(self):
    pass
  

  def Welcome(self,stdscr):
    CenterAddStr(stdscr,1,'Welcome to the TOPS Project Installer')
    CenterAddStr(stdscr,3,'The SIDL Language and compiler included were developed')
    CenterAddStr(stdscr,4,'by the LLNL Babel Team (http://www.llnl.gov/CASC/components)')
    CenterAddStr(stdscr,5,'(hit any key to continue)')
    stdscr.refresh()
    curses.halfdelay(50)
    c = stdscr.getch()
    curses.nocbreak()
  
  def IndicateBKMissing(self,stdscr):
    stdscr.clear()
    CenterAddStr(stdscr,1,'Installer requires the BitKeeper system')
    CenterAddStr(stdscr,2,'Unable to locate the BitKeeper bk command')
    CenterAddStr(stdscr,3,'Please enter the full path for bk')
    CenterAddStr(stdscr,4,'(or hit return if it is not installed)')
    self.bkpath = CenterGetStr(stdscr,5)
    while 1:
      if not self.bkpath:
        CenterAddStr(stdscr,8,'You can install BitKeeper yourself')
        CenterAddStr(stdscr,9,'http://www.bitkeeper.com/Products.Downloads.html')
        CenterAddStr(stdscr,10,'and then rerun the installer')          
        CenterAddStr(stdscr,11,'(hit return to exit)')
        c = stdscr.getkey()
        return
      else:
        if not os.path.isdir(self.bkpath):
          CenterAddStr(stdscr,8,'Directory does not exist. Enter a valid directory or nothing',curses.A_BLINK)
          self.bkpath = CenterGetStr(stdscr,5,self.bkpath)    
        else:
          if os.path.isfile(os.path.join(self.bkpath,'bk')): return
          CenterAddStr(stdscr,8,'Directory does not contain the bk command',curses.A_BLINK)
          self.bkpath = CenterGetStr(stdscr,5,self.bkpath)    

        
  def InstallDirectory(self,stdscr):
    stdscr.clear()
    CenterAddStr(stdscr,1,'Directory to install projects')
    path = os.getcwd()
    self.installpath = CenterGetStr(stdscr,2,path)

  def AlreadyInstalled(self,stdscr):
    stdscr.clear()
    CenterAddStr(stdscr,1,'Looks like BuildSystem is already installed at')
    CenterAddStr(stdscr,2,self.installpath)
    CenterAddStr(stdscr,4,'Use '+self.installpath+'/BuildSystem/install/installer.py')
    CenterAddStr(stdscr,5,'to install additional projects')
    CenterAddStr(stdscr,7,'OR')    
    CenterAddStr(stdscr,9,'Remove all directories in '+self.installpath)
    CenterAddStr(stdscr,10,'and rerun this installer for a complete reinstall')
    CenterAddStr(stdscr,11,'(hit return to exit)')
    c = stdscr.getkey()

  def CannotClone(self,stdscr,mess):
    stdscr.clear()
    CenterAddStr(stdscr,1,'Unable to download bk://sidl@sidl.bkbits.net/BuildSystem')
    CenterAddStr(stdscr,2,'(hit return to exit)')
    l = len(mess)
    l = min(l,500)
    stdscr.addstr(4,1,mess[0:l])
    c = stdscr.getkey()
                 


if __name__ ==  '__main__':
  
  installer = CursesInstall()
  curses.wrapper(installer.Welcome)

  curses.wrapper(installer.IndicateBKMissing)
  if not installer.bkpath: sys.exit()
  print 'BK directory '+installer.bkpath+':'
  
  curses.wrapper(installer.InstallDirectory)
  print 'Install directory '+installer.installpath

  if os.path.isdir(os.path.join(installer.installpath,'BuildSystem')):
    curses.wrapper(installer.AlreadyInstalled)
    sys.exit()
  else:
    (status,output) = commands.getstatusoutput('cd '+installer.installpath+';'+installer.bkpath+'/bk clone bk://sidl@sidl.bkbits.net/BuildSystem')
    if status:
      curses.wrapper(installer.CannotClone,output)
      sys.exit()

  sys.path.insert(0,os.path.join(installer.installpath,'BuildSystem'))
  sys.path.insert(0,os.path.join(installer.installpath,'BuildSystem','install'))
  import install.installer
  install.installer.runinstaller()
      

#!/usr/bin/env python
#
#    This should only be run ONCE! It checks for the existence of
#  BitKeeper and then does the bk clone bk://sidl.bkbits.net/BuildSystem
#  and then calls the installer in BuildSystem/install
#
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

def CenterGetStr(stdscr,my,width = 0,text = '',prompt = ''):
  (y,x) = stdscr.getmaxyx()
  if not width:  width = x - 10 - len(prompt)
  x = int((x - width)/2)
  if prompt:
    stdscr.addstr(my+1,x-len(prompt)-2,prompt)
  curses.textpad.rectangle(stdscr,my,x-1,my+2,x+width)
  subscr  = curses.newwin(1,width,my+1,x)
  subscr.addstr(0,0,text)
  stdscr.refresh()
  textobj = curses.textpad.Textbox(subscr)
  textobj.stripspaces = 1
  text = textobj.edit(ConvertReturnToExit)
  return text

def SelectFromList(stdscr,list,my = 1,text = 'Select desired value'):
  CenterAddStr(stdscr,my,text)
  (y,x) = stdscr.getmaxyx()
  my = my + 2
  i  = 0
  for l in list:
    stdscr.addstr(my,2,str(i)+') '+l)
    i  = i + 1
    my = my + 1
  stdscr.addstr(my+1,2,'Type a number 0 to '+str(i-1))
  stdscr.refresh()
  ch = -1
  while ch >= i or ch < 0:
    ch = stdscr.getch() - ord('0')
  return ch


# -----------------------------------------------------------------    

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

  def GetBrowser(self,stdscr):
    list = ['No browser']
    list.append('A different browser or a browser on a different machine (you will be prompted for it)')
    for l in ['netscape','lynx','opera','mozilla','galeon']:
      (status,output) = commands.getstatusoutput('which '+l)
      if status == 0 and not output[0:3] == 'no ':  # found it :-)
        list.append(output)

    stdscr.clear()
    key = SelectFromList(stdscr,list,my = 1,text = 'Select browser to view documentation')
    if key == 0:
      self.browser = None
      return
    if key == 1:
      stdscr.clear()
      CenterAddStr(stdscr,1,'Enter complete path of browser (for example /usr/bin/netscape or ssh mymachine netscape)')
      self.browser = CenterGetStr(stdscr,2)
      while 1:
        if (not os.path.isfile(self.browser)) and (not self.browser[0:3] == 'ssh'):
          CenterAddStr(stdscr,5,'Program does not exist. Enter a valid program or nothing',curses.A_BLINK)
          self.browser = CenterGetStr(stdscr,2,text = self.browser)
        else:
          break
    else:
      self.browser = list[key]

    stdscr.clear()
    CenterAddStr(stdscr,1,'Testing browser '+self.browser)
    stdscr.refresh()

    #set the method to open URLs for this browser
    if not self.browser.find("opera") == -1:
      self.OpenURL = self.OpenURL_Opera
    elif not self.browser.find("netscape") == -1:
      self.OpenURL = self.OpenURL_Netscape
    elif not self.browser.find("lynx") == -1:
      self.OpenURL = self.OpenURL_Lynx
    elif not self.browser.find("mozilla") == -1:
      self.OpenURL = self.OpenURL_Netscape
    elif not self.browser.find("galeon") == -1:
      self.OpenURL = self.OpenURL_Galeon
                  
    if self.OpenURL('http://www.mcs.anl.gov/petsc'):
      stdscr.clear()
      CenterAddStr(stdscr,1,'Unable to open browser '+self.browser)
      CenterAddStr(stdscr,2,'(Hit any key to continue)')
      stdscr.refresh()
      c = stdscr.getkey()
      self.browser = None
      return

    stdscr.clear()
    CenterAddStr(stdscr,1,'Testing browser '+self.browser)
    stdscr.refresh()
    CenterAddStr(stdscr,2,'Hit y if your browser opened to the PETSc web page, otherwise hit n')
    stdscr.refresh()
    c = stdscr.getkey()
    if c != 'y': self.browser = None
      
  
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
          self.bkpath = CenterGetStr(stdscr,5,text = self.bkpath)    
        else:
          if os.path.isfile(os.path.join(self.bkpath,'bk')): return
          CenterAddStr(stdscr,8,'Directory does not contain the bk command',curses.A_BLINK)
          self.bkpath = CenterGetStr(stdscr,5,text = self.bkpath)    

        
  def InstallDirectory(self,stdscr):
    stdscr.clear()
    CenterAddStr(stdscr,1,'Directory to install projects')
    path = os.getcwd()
    self.installpath = CenterGetStr(stdscr,2,text = path)

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
    CenterAddStr(stdscr,1,'Unable to download bk://sidl.bkbits.net/BuildSystem')
    CenterAddStr(stdscr,2,'(hit return to exit)')
    l = len(mess)
    l = min(l,500)
    stdscr.addstr(4,1,mess[0:l])
    c = stdscr.getkey()
                 
  def GetBuildSystem(self,stdscr):
    stdscr.clear()
    CenterAddStr(stdscr,1,'Downloading BuildSystem')
    stdscr.refresh()
    (self.status,output) = commands.getstatusoutput('cd '+self.installpath+';'+self.bkpath+'/bk clone bk://sidl.bkbits.net/BuildSystem')
    
#------------------------------------------------------------------

  def OpenURL_Netscape(self,url):
    (status,output) = commands.getstatusoutput(self.browser+" -remote 'openURL("+url+")'")
    if status == 0:  return
    pipe = os.popen(self.browser+" "+url+" &")
    import time
    time.sleep(5)
    return 0
    
  def OpenURL_Opera(self,url):
    pipe = os.popen(self.browser+" -remote 'openURL("+url+")'&")
    return 0

  def OpenURL_Lynx(self,url):
    return  os.system(self.browser+" "+url)

  def OpenURL_Galeon(self,url):
    pipe = os.popen(self.browser+" "+url)
    return 0

#------------------------------------------------------------------

if __name__ ==  '__main__':
  
  installer = CursesInstall()
  curses.wrapper(installer.Welcome)

  curses.wrapper(installer.GetBrowser)
  #print 'Browser '+installer.browser
  
  # need to have a more complete search for bk, but cannot use configure stuff
  (status,output) = commands.getstatusoutput('which bk')
  if status == 0:  # found it :-)
    installer.bkpath = os.path.dirname(output)
  else:
    curses.wrapper(installer.IndicateBKMissing)
    if not installer.bkpath: sys.exit()
  #print 'BK directory '+installer.bkpath+':'
  
  curses.wrapper(installer.InstallDirectory)
  #print 'Install directory '+installer.installpath

  if os.path.isdir(os.path.join(installer.installpath,'BuildSystem')):
    curses.wrapper(installer.AlreadyInstalled)
    sys.exit()
  else:
    curses.wrapper(installer.GetBuildSystem)
    if installer.status:
      curses.wrapper(installer.CannotClone,output)
      sys.exit()

  sys.path.insert(0,os.path.join(installer.installpath,'BuildSystem','install'))
  import installer
  installer.runinstaller(["-debugSections=[install]",'-debugLevel=2'])
      

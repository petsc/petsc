#!/usr/bin/env python
#
#    This should only be run ONCE! It checks for the existence of
#  BitKeeper and then does the bk clone bk://sidl.bkbits.net/BuildSystem
#  and then calls the installer in BuildSystem/install
#
import commands
import curses
import curses.textpad
import os
import sys

class BootstrapInstall (object):
  '''A BootstrapInstall assembles the minimum set of components necessary to begin the build and install process'''
  def __init__(self):
    self.bkPath      = None
    self.installPath = None
    return

  def getExecutable(com):
    '''This is a replacement for the configure functionality which is not yet present'''
    (status, output) = commands.getstatusoutput('which '+com)
    if status == 0 and not output[0:3] == 'no ' and output.find('not found') == -1:
      return output
    else:
      return None
  getExecutable = staticmethod(getExecutable)

  def welcome(self):
    '''Provide a user greeting'''
    sys.stdout.write('Welcome to the TOPS Project Installer\n')
    sys.stdout.write('The SIDL Language and initial compiler included were developed by the LLNL Babel Team\n')
    sys.stdout.write('(http://www.llnl.gov/CASC/components)\n')
    return

  def installBitkeeper(self):
    '''Find Bitkeeper if it is installed, otherwise install it.
       - Return True if installation is successful
       - Set self.bkPath to the directory containing Bitkeeper.'''
    # If self.bkPath is set, check it
    if self.bkPath:
      if os.path.isdir(self.bkPath):
        prog = os.path.join(self.bkPath, 'bk')
        if os.path.isfile(prog) and os.access(prog, os.X_OK):
          return 1
      return 0
    # Otherwise try to locate "bk"
    output = BootstrapInstall.getExecutable('bk')
    if output:
      # TODO: Log output
      # return 0
      pass
    self.bkPath = os.path.dirname(output)
    return 1

  def createInstallDirectory(self):
    '''The installation directory defaults to the current directory, unless we are in HOME, in which case we create "petsc-3.0"
       - Return True if creation succeeds
       - Set self.installPath to the installation directory'''
    # If self.bkPath is set, check it
    if not selfinstallPath:
      self.installPath = os.getcwd()
    if os.path.samefile(self.installPath, os.getenv('HOME')):
      self.installPath = os.path.join(self.installPath, 'petsc-3.0')
    if not os.path.isdir(self.installPath):
      try:
        os.makedirs(self.installPath)
        os.chdir(self.installPath)
      except IOError, e:
        #TODO: Log error
        return 0
    return 1

  def installBuildSystem(self):
    '''Check for BuildSystem and install it if it is not present.
       - Return True if installation succeeds'''
    # Should really check that it is functioning here
    if not os.path.isdir('BuildSystem'):
      (status, self.errorString) = commands.getstatusoutput(self.bkPath+'/bk clone bk://sidl.bkbits.net/BuildSystem')
      if status:
        # TODO: Log error
        return 0
    return 1

# ---------------------------------------------------------------------------------------------------------------------
class CursesInstall (BootstrapInstall):
  def __init__(self):
    self.bkPath      = None
    self.installPath = None
    return

  def ConvertReturnToExit(key):
    if key == ord('\n'): return 7
    return key
  ConvertReturnToExit = staticmethod(ConvertReturnToExit)

  def CenterAddStr(stdscr, my, text, attr = 0):
    (y,x) = stdscr.getmaxyx()
    x = (x - len(text))/2
    stdscr.addstr(my,x,text,attr)
  CenterAddStr = staticmethod(CenterAddStr)

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
    text = textobj.edit(CursesInstall.ConvertReturnToExit)
    return text
  CenterGetStr = staticmethod(CenterGetStr)

  def SelectFromList(stdscr,list,my = 1,text = 'Select desired value'):
    charactors = []
    for i in range(0,10):
      charactors.append(i+ord('0'))
    for i in range(0,26):
      charactors.append(i+ord('a'))
    for i in range(0,26):
      charactors.append(i+ord('A'))

    choices = []
    CursesInstall.CenterAddStr(stdscr,my,text)
    (y,x) = stdscr.getmaxyx()
    my = my + 2
    i  = 0
    for l in list:
      stdscr.addstr(my,2,chr(charactors[i])+') '+l)
      choices.append(charactors[i])
      i  = i + 1
      my = my + 1
    stdscr.addstr(my+1,2,'Type key to pick selection')
    stdscr.refresh()
    ch = -1
    while not ch in choices:
      ch = stdscr.getch()
    ch = choices.index(ch)
    return ch
  SelectFromList = staticmethod(SelectFromList)

  def getBrowser(self, stdscr):
    list = ['No browser']
    list.append('A different browser or a browser on a different machine (you will be prompted for it)')
    for l in ['netscape','lynx','opera','mozilla','galeon']:
      output = BootstrapInstall.getExecutable(l)
      if output: list.append(output)

    stdscr.clear()
    key = CursesInstall.SelectFromList(stdscr,list,my = 1,text = 'Select browser to view documentation')
    if key == 0:
      self.browser = None
      return
    if key == 1:
      stdscr.clear()
      CursesInstall.CenterAddStr(stdscr,1,'Enter complete path of browser (for example /usr/bin/netscape or ssh mymachine netscape)')
      self.browser = CursesInstall.CenterGetStr(stdscr,2)
      while 1:
        if (not os.path.isfile(self.browser)) and (not self.browser[0:3] == 'ssh'):
          CursesInstall.CenterAddStr(stdscr,5,'Program does not exist. Enter a valid program or nothing',curses.A_BLINK)
          self.browser = CursesInstall.CenterGetStr(stdscr,2,text = self.browser)
        else:
          break
    else:
      self.browser = list[key]

    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr,1,'Testing browser '+self.browser)
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
      CursesInstall.CenterAddStr(stdscr,1,'Unable to open browser '+self.browser)
      CursesInstall.CenterAddStr(stdscr,2,'(Hit any key to continue)')
      stdscr.refresh()
      c = stdscr.getkey()
      self.browser = None
      return

    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr,1,'Testing browser '+self.browser)
    stdscr.refresh()
    CursesInstall.CenterAddStr(stdscr,2,'Hit y if your browser opened to the PETSc web page, otherwise hit n')
    stdscr.refresh()
    c = stdscr.getkey()
    if c != 'y': self.browser = None
    return

  def cursesWelcome(self, stdscr):
    '''The curses callback for the user greeting'''
    CursesInstall.CenterAddStr(stdscr, 1, 'Welcome to the TOPS Project Installer')
    CursesInstall.CenterAddStr(stdscr, 3, 'The SIDL Language and initial compiler included were developed')
    CursesInstall.CenterAddStr(stdscr, 4, 'by the LLNL Babel Team (http://www.llnl.gov/CASC/components)')
    CursesInstall.CenterAddStr(stdscr, 5, '(hit any key to continue)')
    stdscr.refresh()
#    curses.halfdelay(50)
    c = stdscr.getch()
#    curses.nocbreak()
    stdscr.clear()
    self.getBrowser(stdscr)
    return

  def welcome(self):
    '''Provide a user greeting and select a browser'''
    return curses.wrapper(self.cursesWelcome)

  def cursesIndicateBKMissing(self, stdscr):
    '''Query the user for the location of Bitkeeper, and return True if it is found.'''
    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr, 1, 'Installer requires the BitKeeper system')
    CursesInstall.CenterAddStr(stdscr, 2, 'Unable to locate the BitKeeper bk command')
    CursesInstall.CenterAddStr(stdscr, 3, 'Please enter the full path for bk')
    CursesInstall.CenterAddStr(stdscr, 4, '(or hit return if it is not installed)')
    self.bkPath = CursesInstall.CenterGetStr(stdscr, 5)
    while 1:
      if not self.bkPath:
        CursesInstall.CenterAddStr(stdscr, 8,  'You can install BitKeeper yourself')
        CursesInstall.CenterAddStr(stdscr, 9,  'http://www.bitkeeper.com/Products.Downloads.html')
        CursesInstall.CenterAddStr(stdscr, 10, 'and then rerun the installer')
        CursesInstall.CenterAddStr(stdscr, 11, '(hit return to exit)')
        c = stdscr.getkey()
        return 0
      else:
        if not os.path.isdir(self.bkPath):
          CursesInstall.CenterAddStr(stdscr, 8, 'Directory does not exist. Enter a valid directory or nothing', curses.A_BLINK)
          self.bkPath = CursesInstall.CenterGetStr(stdscr, 5, text = self.bkPath)
        else:
          if BootstrapInstall.installBitkeeper(self): return 1
          CursesInstall.CenterAddStr(stdscr, 8, 'Directory does not contain the bk command', curses.A_BLINK)
          self.bkPath = CursesInstall.CenterGetStr(stdscr, 5, text = self.bkPath)    
    return 1

  def installBitkeeper(self):
    '''Call the default method, but ask the user if Bitkeeper is not found.'''
    if not BootstrapInstall.installBitkeeper(self):
      return curses.wrapper(self.cursesIndicateBKMissing)
    return 1
        
  def cursesInstallDirectory(self, stdscr):
    '''The installation directory defaults to the current directory, unless we are in HOME, in which case we create "petsc-3.0"
       - Return True if creation succeeds
       - Set self.installPath to the installation directory
       - Query user if things go wrong'''
    while 1:
      stdscr.clear()
      CursesInstall.CenterAddStr(stdscr, 1, 'Directory to install projects')
      path = os.getcwd()
      if os.path.samefile(path,os.getenv('HOME')):
        path = os.path.join(path, 'petsc-3.0')
      self.installPath = CursesInstall.CenterGetStr(stdscr, 2, text = path)
      if not os.path.isdir(self.installPath):
        CursesInstall.CenterAddStr(stdscr, 6, 'Directory '+self.installPath+' does not exist. Create (y/n)?')
        stdscr.refresh()
        c = stdscr.getkey()
        while not (c == 'y' or c == 'n'):
          c = stdscr.getkey()
        if c == 'y':
          try:
            os.makedirs(self.installPath)
            os.chdir(self.installPath)
            return 1
          except:
            CursesInstall.CenterAddStr(stdscr, 8, 'Cannot create directory '+self.installPath)
            CursesInstall.CenterAddStr(stdscr, 9, '(q to quit installer, t to try again)' )
            c = stdscr.getkey()
            while not (c == 'q' or c == 't'):
              c = stdscr.getkey()
            if c == 'q':
              self.installPath = None
              return 0
      else:
        return 1
    return 1

  def createInstallDirectory(self):
    '''Ask the user for the installation directory'''
    return curses.wrapper(installer.cursesInstallDirectory)

  def cursesAlreadyInstalled(self, stdscr):
    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr, 1, 'Looks like BuildSystem is already installed at')
    CursesInstall.CenterAddStr(stdscr, 2, self.installPath)
    CursesInstall.CenterAddStr(stdscr, 4, 'Use '+self.installPath+'/BuildSystem/install/gui.py')
    CursesInstall.CenterAddStr(stdscr, 5, 'to install additional projects after bootstrap finishes')
    CursesInstall.CenterAddStr(stdscr, 7, 'OR')    
    CursesInstall.CenterAddStr(stdscr, 9, 'Remove all directories in '+self.installPath)
    CursesInstall.CenterAddStr(stdscr, 10, 'and rerun this installer for a complete reinstall')
    CursesInstall.CenterAddStr(stdscr, 11, '(hit return to continue)')
    c = stdscr.getkey()
    return 0

  def cursesCannotClone(self, stdscr, mess):
    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr, 1, 'Unable to download bk://sidl.bkbits.net/BuildSystem')
    CursesInstall.CenterAddStr(stdscr, 2, '(hit return to exit)')
    l = len(mess)
    l = min(l,500)
    stdscr.addstr(4, 1, mess[0:l])
    c = stdscr.getkey()
    return 0

  def installBuildSystem(self):
    '''Check for BuildSystem and install it if it is not present.
       - Return True if installation succeeds'''
    if os.path.isdir('BuildSystem'):
      return curses.wrapper(self.cursesAlreadyInstalled)
    if not BootstrapInstall.installBuildSystem(self):
      return curses.wrapper(self.cursesCannotClone, self.errorString)
    return 1      

  def OpenURL_Netscape(self, url):
    (status, output) = commands.getstatusoutput(self.browser+" -remote 'openURL("+url+")'")
    if status == 0: return
    pipe = os.popen(self.browser+' '+url+' &')
    import time
    time.sleep(5)
    return 0

  def OpenURL_Opera(self, url):
    pipe = os.popen(self.browser+" -remote 'openURL("+url+")'&")
    return 0

  def OpenURL_Lynx(self,url):
    return os.system(self.browser+' '+url)

  def OpenURL_Galeon(self,url):
    pipe = os.popen(self.browser+' '+url)
    return 0

if __name__ ==  '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '-batch':
    installer = BootstrapInstall()
  else:
    installer = CursesInstall()
  installer.welcome()
  if not installer.installBitkeeper():
    sys.exit('Could not locate Bitkeeper')
  if not installer.createInstallDirectory():
    sys.exit('Could not create installation directory '+installer.installPath)
  if not installer.installBuildSystem():
    sys.exit('Could not install BuildSystem')
  # Handoff to installer
  sys.stdout.write('Installing the BuildSystem, Runtime and Compiler (this will take a while)\n')
  sys.stdout.flush()
  sys.path.insert(0, os.path.join(installer.installPath, 'BuildSystem'))
  import install.installer
  install.installer.runinstaller(["-debugSections=[install]",'-debugLevel=2'])
      

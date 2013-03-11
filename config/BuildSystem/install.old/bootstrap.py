#!/usr/bin/env python
#
#    This should only be run ONCE! It checks for the existence of
#  BitKeeper and then does the bk clone bk://sidl.bkbits.net/BuildSystem
#  and then calls the installer in sidl/BuildSystem/install
#
import commands
import curses
import curses.textpad
import os
import sys

if not hasattr(sys, 'version_info'):
  raise RuntimeError('You must have Python version 2.2 or higher to run the bootstrap')

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
    sys.stdout.write('Welcome to the ASE Project Installer\n')
    sys.stdout.write('The SIDL Language and initial compiler included were developed by the LLNL Babel Team\n')
    sys.stdout.write('(http://www.llnl.gov/CASC/components)\n')
    return

  def installBitkeeper(self):
    '''Find Bitkeeper if it is installed, otherwise install it.
       - Return True if installation is successful
       - Set self.bkPath to the directory containing Bitkeeper.'''
    # First check that hostname returns something BitKeeper is happy with
    import socket
    hostname = socket.gethostname()
    if len(hostname) > 8 and hostname[0:9] == 'localhost':
      os.putenv('BK_HOST','bkneedsname.org')
    elif hostname[-1] == '.':
      os.putenv('BK_HOST',hostname+'org')
    elif hostname.find('.') == -1:
      os.putenv('BK_HOST',hostname+'.org')
                
    # If self.bkPath is set, check it
    if self.bkPath:
      if os.path.isdir(self.bkPath):
        prog = os.path.join(self.bkPath, 'bk')
        if os.path.isfile(prog) and os.access(prog, os.X_OK):
          return 1
      return 0
    # Otherwise try to locate "bk"
    output = BootstrapInstall.getExecutable('bk')
    if not output:
      # TODO: Log output
      return 0
    self.bkPath = os.path.dirname(output)
    return 1

  def createInstallDirectory(self):
    '''The installation directory defaults to the current directory, unless we are in HOME, in which case we create "petsc-3.0"
       - Return True if creation succeeds
       - Set self.installPath to the installation directory'''
    # If self.bkPath is set, check it
    if not self.installPath:
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

  def killRDict(self):
    '''Kill any remaining RDict servers'''
    import signal
    import time

    if not os.path.isdir('/proc'):
      sys.stdout.write('WARNING: Cannot kill rouge RDict servers\n')
      return
    pids = []
    for f in os.listdir('/proc'):
      try:
        cmdline = os.path.join('/proc', str(int(f)), 'cmdline')
        if os.path.isfile(cmdline) and file(cmdline).read().find('RDict')>=0:
          pids.append(int(f))
          # sys.stdout.write('Killing RDict server '+str(pids[-1])+'\n')
          try: os.kill(pids[-1], signal.SIGTERM)
          except: pass
          time.sleep(1)
      except ValueError:
        pass
    return

  def cleanup(self):
    '''Kill all remaining RDict servers and cleanup aborted attempts'''
    import shutil

    self.killRDict()
    for d in [os.path.join(self.installPath, 'ply', 'ply-dev'), os.path.join(self.installPath, 'sidl', 'Runtime'), os.path.join(self.installPath, 'sidl', 'Compiler')]:
      if os.path.isdir(d):
        shutil.rmtree(d)
    return

  def setupPaths(self):
    '''Setup the paths'''
    sys.path.insert(0, os.path.join(self.installPath, 'sidl','BuildSystem'))
    return

  def installBuildSystem(self):
    '''Check for BuildSystem and install it if it is not present.
       - Return True if installation succeeds'''
    # Should really check that it is functioning here
    bsDir = os.path.join('sidl', 'BuildSystem')
    if not os.path.isdir(bsDir):
      if not os.path.isdir('sidl'): os.makedirs('sidl')
      (status, self.errorString) = commands.getstatusoutput(self.bkPath+'/bk clone bk://sidl.bkbits.net/BuildSystem sidl/BuildSystem')
      if status:
        # TODO: Log error
        return 0
    else:
      # Remove any existing RDict.loc, RDict.db, bsSource.db, and configure.log
      lockFile = os.path.join(bsDir, 'RDict.loc')
      if os.path.isfile(lockFile):
        sys.stdout.write('Removing old RDict lock file '+lockFile+'\n')
        os.remove(lockFile)
      dbFile = os.path.join(bsDir, 'RDict.db')
      if os.path.isfile(dbFile):
        sys.stdout.write('Removing old RDict database file '+dbFile+'\n')
        os.remove(dbFile)
      sourceDbFile = os.path.join(bsDir, 'bsSource.db')
      if os.path.isfile(sourceDbFile):
        sys.stdout.write('Removing old source database file '+sourceDbFile+'\n')
        os.remove(sourceDbFile)
      configureLogFile = os.path.join(bsDir, 'configure.log')
      if os.path.isfile(configureLogFile):
        sys.stdout.write('Removing old configure log file '+configureLogFile+'\n')
        os.remove(configureLogFile)
    return 1

  def runInstaller(self,args):
    import install.installer
    sys.stdout.write('Installing the BuildSystem, Runtime and Compiler (this will take a while)\n')
    sys.stdout.flush()
    install.installer.runinstaller(args)

# ---------------------------------------------------------------------------------------------------------------------
class ScrollingWindow:
  def __init__(self,stdscr,y,x,h,w,pwd):
    ''' Create subwindow with box around it'''
    curses.textpad.rectangle(stdscr,y,x,y+h,x+w)
    self.stdscr = stdscr
    self.stdscr.refresh()
    self.x  = x+1
    self.y  = y+1
    self.h  = h - 2
    self.w  = w - 2
    if pwd: self.pwd = pwd+'/'
    else:   self.pwd = ''
    self.lines = []
    for i in range(0,self.h):
      self.lines.append('\n')
    self.mess = ''
    self.tabsize = 0

  def tab(self,size):
    self.tabsize = size
    
  def write(self,mess):
    import re
    nmess = mess.split('\n')
    for mess in nmess:
      if mess == '': continue
      for i in range(0,self.h-1):
        self.lines[i] = self.lines[i+1]
      self.lines[self.h-1] = '                     '[0:self.tabsize]+mess
      for i in range(0,self.h):
        amess = self.lines[i]+'                                                                                                          '
        amess = amess.replace(self.pwd,'')
        self.stdscr.addstr(self.y+i,self.x,amess[0:self.w])
      curses.textpad.rectangle(self.stdscr,self.y-1,self.x-1,self.y+self.h+1,self.x+self.w+1)
      self.stdscr.refresh()
       
#-----------------------------------------------------------------------------------------------------------------------
    
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
    if x < 80 or y < 25:
      raise RuntimeError('Resize your window to be at least 25 rows by 80 columns')
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

  def SelectFromSubList(stdscr,list,my,text,charactors):
    stdscr.clear()
    choices = []
    CursesInstall.CenterAddStr(stdscr,my,text)
    (y,x) = stdscr.getmaxyx()
    omy = my
    my = my + 2
    ln = y - my - 2
    mess = 'Type key to pick selection'
    if ln < len(list):
      sublist = list[0:ln]
      mess    = mess + ' (+ for more choices)'
    else:              sublist = list
    if not charactors[0] == ord('0'):
      mess    = mess + ' (- for previous choices)'
    i  = 0
    for l in sublist:
      stdscr.addstr(my,2,chr(charactors[i])+') '+l)
      choices.append(charactors[i])
      i  = i + 1
      my = my + 1
    stdscr.addstr(my+1,2,mess)
    stdscr.refresh()
    ch = -1
    while not ch in choices:
      ch = stdscr.getch()
      if ch == ord('+'):
        r = CursesInstall.SelectFromSubList(stdscr,list[ln:-1],omy,text,charactors[ln:-1])
        if not r == '-':
          return ln + r
        else:
          return CursesInstall.SelectFromSubList(stdscr,list,omy,text,charactors)
      if ch == ord('-') and not charactors[0] == ord('0'):
        return '-'
    ch = choices.index(ch)
    return ch
  SelectFromSubList = staticmethod(SelectFromSubList)
    
  def SelectFromList(stdscr,list,my = 1,text = 'Select desired value'):
    charactors = []
    for i in range(0,10):
      charactors.append(i+ord('0'))
    for i in range(0,26):
      charactors.append(i+ord('a'))
    for i in range(0,26):
      charactors.append(i+ord('A'))

    return CursesInstall.SelectFromSubList(stdscr,list,my,text,charactors)
  SelectFromList = staticmethod(SelectFromList)

  def getBrowser(self, stdscr):
    'Not currently used; would allow user to select browser for html output'
    list = ['No browser (select this also for Internet Explorer and Safari)']
    list.append('A different browser or one on a different machine (you will be prompted)')
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
      CursesInstall.CenterAddStr(stdscr,1,'Enter complete path of browser (e.g. /usr/bin/netscape or ssh mymachine netscape)')
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
    CursesInstall.CenterAddStr(stdscr, 1, 'Welcome to the ASE Project Installer')
    CursesInstall.CenterAddStr(stdscr, 3, 'The SIDL Language and initial compiler included were developed')
    CursesInstall.CenterAddStr(stdscr, 4, 'by the LLNL Babel Team (http://www.llnl.gov/CASC/components)')
    CursesInstall.CenterAddStr(stdscr, 5, '(hit any key to continue)')
    stdscr.refresh()
#    curses.halfdelay(50)
    c = stdscr.getch()
#    curses.nocbreak()
    stdscr.clear()
#    self.getBrowser(stdscr)
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
        if os.path.isfile(self.bkPath): self.bkPath = os.path.dirname(self.bkPath)
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

  def cursesCleanup(self,stdscr):
    '''Display nice message while running cleanup'''
    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr, 3, 'Removing any previous ASE demons')
    stdscr.refresh()
    BootstrapInstall.cleanup(self)
    
  def cleanup(self):
    '''Display nice message while running cleanup'''
    return curses.wrapper(installer.cursesCleanup)
                          
  def cursesAlreadyInstalled(self, stdscr):
    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr, 1, 'Looks like BuildSystem is already installed at')
    CursesInstall.CenterAddStr(stdscr, 2, self.installPath+'/sidl/BuildSystem')
    CursesInstall.CenterAddStr(stdscr, 4, 'Use '+self.installPath+'/sidl/BuildSystem/install/gui.py')
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

  def cursesInstallBuildSystem(self,stdscr):
    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr, 3, 'Downloading ASE software')
    stdscr.refresh()
    if not os.path.exists(os.path.join(os.getenv('HOME'),'.bk','accepted')):
      os.putenv('BK_LICENSE','ACCEPTED')
    return BootstrapInstall.installBuildSystem(self)
      
  def installBuildSystem(self):
    '''Check for BuildSystem and install it if it is not present.
       - Return True if installation succeeds'''
    if os.path.isdir('sidl/BuildSystem'):
      return curses.wrapper(self.cursesAlreadyInstalled)
    if not curses.wrapper(self.cursesInstallBuildSystem):
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

  def cursesRunInstaller(self,stdscr,args):
    stdscr.clear()
    CursesInstall.CenterAddStr(stdscr,1,'Installing the BuildSystem, Runtime and Compiler')
    CursesInstall.CenterAddStr(stdscr,3,'Complete log file: make.log    Error file: installer_err.log')
    import install.installer
    import logging
    (y,x) = stdscr.getmaxyx()
    logging.dW = ScrollingWindow(stdscr,4,3,y-7,x-6,self.installPath)
    install.installer.runinstaller(args)

  def runInstaller(self,args):
    '''Display nice message while running installer'''
    return curses.wrapper(self.cursesRunInstaller,args)

#-------------------------------------------------------------------------------------
if __name__ ==  '__main__':
  try:
    if len(sys.argv) > 1 and (sys.argv[1] == '-batch' or sys.argv[1] == '--batch'):
      installer = BootstrapInstall()
    else:
      installer = CursesInstall()
    installer.welcome()
    if not installer.installBitkeeper():
      sys.exit('Could not locate Bitkeeper')
    if not installer.createInstallDirectory():
      sys.exit('Could not create installation directory '+installer.installPath)
    installer.cleanup()
    if not installer.installBuildSystem():
      sys.exit('Could not install BuildSystem')
    installer.setupPaths()
    installer.runInstaller(['-debugSections=[install,compile,bk,shell,build]','-debugLevel=4','-installedprojects=[]'])
  except Exception, e:
    import traceback

    print str(e)
    log = file('installer_err.log', 'w')
    log.write(str(e)+'\n')
    traceback.print_tb(sys.exc_info()[2], file = log)
    log.close()
    sys.exit(1)


#!/usr/bin/env python
#
#   Gets list of available projects (and installed projects)
# and allows user to select a new project to install
#
import user
import curses
import curses.textpad
import os
import os.path
import sys
import commands
import bootstrap
import project
import RDict
import urlparse
import urllib
import httplib
import HTMLParser

# -----------------------------------------------------------------    

class iproject:
  def __init__(self,url):
    self.url  = url
    self.text = 'None'
    
# -----------------------------------------------------------------

class CursesInstallGUI(HTMLParser.HTMLParser):
  def __init__(self):
    HTMLParser.HTMLParser.__init__(self)
    self.foundProject = 0
    
  def handle_starttag(self, tag, attrs):
    if tag == 'a':
      if not attrs[0][0] == 'href': return
      if attrs[0][1].find('8080') < 0: return
      p = attrs[0][1].replace(':8080','')
      p = p.replace('http:','bk:')
      self.iprojects.append(iproject(p))
      self.foundProject = 1
    if tag == 'td':
      if self.foundProject > 0: self.foundProject = self.foundProject + 1

  def handle_data(self, data):
    if self.foundProject == 3:
      self.iprojects[-1].text = data
      self.foundProject = 0
    
  def Welcome(self,stdscr):
    bootstrap.CursesInstall.CenterAddStr(stdscr,1,'Welcome to the ASE Project Installer')
    bootstrap.CursesInstall.CenterAddStr(stdscr,3,'The SIDL Language and compiler included were developed')
    bootstrap.CursesInstall.CenterAddStr(stdscr,4,'by the LLNL Babel Team (http://www.llnl.gov/CASC/components)')
    bootstrap.CursesInstall.CenterAddStr(stdscr,5,'(hit any key to continue)')
    stdscr.refresh()
    curses.halfdelay(50)
    c = stdscr.getch()
    curses.nocbreak()

  def InstalledProjects(self,stdscr):
    stdscr.clear()
    bootstrap.CursesInstall.CenterAddStr(stdscr,1,'Currently Installed Projects')

    self.argsDB   = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
    cnt = 3
    for i in self.argsDB['installedprojects']:
      bootstrap.CursesInstall.CenterAddStr(stdscr,cnt,i.getUrl())
      cnt = cnt+1
          
    bootstrap.CursesInstall.CenterAddStr(stdscr,cnt+1,'(hit any key to continue)')
    stdscr.refresh()
    curses.halfdelay(100)
    c = stdscr.getch()
    curses.nocbreak()

  def GetProjects(self):
    self.iprojects = []
    for l in ['http://tops.bkbits.net/index.html','http://petsc.bkbits.net/index.html','http://sidl.bkbits.net/index.html','http://mpib.bkbits.net/index.html']:
      # copy down the website
      machine,urlpath = urlparse.urlparse(l)[1:3]
      http = httplib.HTTP(machine)
      http.putrequest('GET',urlpath)
      http.putheader('Accept','*/*')
      http.putheader('Host',machine)
      http.endheaders()
      errcode, errmesg, self.headers = http.getreply()

      f    = http.getfile()
      self.feed(f.read())
      f.close()

      # remove projects that are already installed
      for i in self.argsDB['installedprojects']:
        j = i.getUrl()
        for k in self.iprojects:
          if j == k.url: self.iprojects.remove(k)

      for i in self.iprojects:
        if not i.url.find('-release') == -1: self.iprojects.remove(i)  # remove  PETSc 2 releases
        if not i.url.find('/bugdb') == -1: self.iprojects.remove(i)    # remove BitKeeper crap
        if not i.url.find('petsc-dev') == -1: self.iprojects.remove(i) # remove PETSc 2
        if not i.url.find('blaslapack') == -1: self.iprojects.remove(i) 
        

  def SelectProject(self,stdscr):
    stdscr.clear()
    list = []
    for i in self.iprojects:
      list.append(i.text+' '+i.url)
    key = bootstrap.CursesInstall.SelectFromList(stdscr,list,my = 1,text = 'Select project to install')
    self.project = self.iprojects[key]

  def cursesRunInstaller(self,stdscr):
    stdscr.clear()
    bootstrap.CursesInstall.CenterAddStr(stdscr,1,'Installing '+self.project.url)
    bootstrap.CursesInstall.CenterAddStr(stdscr,3,'Complete log file: make.log    Error file: installer_err.log')
    import install.installer
    import logging
    (y,x) = stdscr.getmaxyx()
    logging.dW = bootstrap.ScrollingWindow(stdscr,4,3,y-7,x-6,'')
    install.installer.runinstaller([self.project.url])

  def cursesGetProjects(self,stdscr):
    '''Display nice message while downloading list of projects'''
    stdscr.clear()
    bootstrap.CursesInstall.CenterAddStr(stdscr, 3, 'Downloading list of possible projects')
    stdscr.refresh()
    self.GetProjects()

#------------------------------------------------------------------

if __name__ ==  '__main__':

  gui = CursesInstallGUI()
  curses.wrapper(gui.Welcome)

  curses.wrapper(gui.InstalledProjects)
  curses.wrapper(gui.cursesGetProjects)
  curses.wrapper(gui.SelectProject)
  curses.wrapper(gui.cursesRunInstaller)

      

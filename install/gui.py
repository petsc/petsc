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

class CursesInstallGUI(HTMLParser.HTMLParser):
  def __init__(self):
    HTMLParser.HTMLParser.__init__(self)
  
  def handle_starttag(self, tag, attrs):
    if not tag == 'a': return
    if not attrs[0][0] == 'href': return
    if attrs[0][1].find('8080') < 0: return
    p = attrs[0][1].replace(':8080','')
    p = p.replace('http:','bk:')
    self.iprojects.append(p)

  def Welcome(self,stdscr):
    bootstrap.CenterAddStr(stdscr,1,'Welcome to the TOPS Project Installer')
    bootstrap.CenterAddStr(stdscr,3,'The SIDL Language and compiler included were developed')
    bootstrap.CenterAddStr(stdscr,4,'by the LLNL Babel Team (http://www.llnl.gov/CASC/components)')
    bootstrap.CenterAddStr(stdscr,5,'(hit any key to continue)')
    stdscr.refresh()
    curses.halfdelay(50)
    c = stdscr.getch()
    curses.nocbreak()

  def InstalledProjects(self,stdscr):
    stdscr.clear()
    bootstrap.CenterAddStr(stdscr,1,'Currently Installed Projects')

    self.argsDB   = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
    cnt = 3
    for i in self.argsDB['installedprojects']:
      bootstrap.CenterAddStr(stdscr,cnt,i.getUrl())
      cnt = cnt+1
          
    bootstrap.CenterAddStr(stdscr,cnt+1,'(hit any key to continue)')
    stdscr.refresh()
    curses.halfdelay(100)
    c = stdscr.getch()
    curses.nocbreak()

  def SelectProject(self,stdscr):
    stdscr.clear()
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
        if j in self.iprojects: self.iprojects.remove(j)
        
    key = bootstrap.SelectFromList(stdscr,self.iprojects,my = 1,text = 'Select project to install')
    self.project = self.iprojects[key]


#------------------------------------------------------------------

if __name__ ==  '__main__':

  gui = CursesInstallGUI()
  curses.wrapper(gui.Welcome)

  curses.wrapper(gui.InstalledProjects)
  curses.wrapper(gui.SelectProject)
  print 'Downloading and installing '+gui.project
  sys.stdout.flush()
  
  import installer
  installer.runinstaller([gui.project])
      

#!/usr/bin/env python
#!/bin/env python
# 
# Reads in a list of polymorphic functions (for C++) and adds them
# to the appropriate manual page
#
#  Usage:
#    processpoly.py
#
import os
import glob
import posixpath
from exceptions import *
from sys import *
from string import *


# Read in the filename contents, and search for the formatted
# String 'Level:' and return the level info.
# Also adds the BOLD HTML format to Level field
def modifylevel(filename,secname):
      import re
      try:
            fd = open(filename,'r')
      except:
            print 'Error! Cannot open file:',filename
            exit()
      buf    = fd.read()
      fd.close()
      re_level = re.compile(r'(Level:)\s+(\w+)')
      m = re_level.search(buf)
      level = 'none'
      if m:
            level = m.group(2)
      else:
            print 'Error! No level info in file:', filename

      # Now takeout the level info, and move it to the end,
      # and also add the bold format.
      tmpbuf = re_level.sub('',buf)
      re_loc = re.compile('(<FONT COLOR="#CC3333">Location:</FONT>)')
      tmpbuf = re_loc.sub('<P><B><FONT COLOR="#CC3333">Level:</FONT></B>' + level + r'\n<BR>\1',tmpbuf)

      # Modify .c#,.h# to .c.html#,.h.html
      re_loc = re.compile('.c#')
      tmpbuf = re_loc.sub('.c.html#',tmpbuf)
      re_loc = re.compile('.h#')
      tmpbuf = re_loc.sub('.h.html#',tmpbuf)
      
      re_loc = re.compile('</BODY></HTML>')
      outbuf = re_loc.sub('<BR><A HREF="./index.html">Index of all ' + secname + ' routines</A>\n<BR><A HREF="../../index.html">Table of Contents for all manual pages</A>\n<BR><A HREF="../singleindex.html">Index of all manual pages</A>\n</BODY></HTML>',tmpbuf)

      # write the modified manpage
      try:
            #fd = open(filename[:-1],'w')
            fd = open(filename,'w')
      except:
            print 'Error! Cannot write to file:',filename
            exit()            
      fd.write(outbuf)
      fd.close()
      return level
      

# Gets the list of man* dirs present in the doc dir.
# Each dir will have an index created for it.
def getallmandirs(dirs):
      mandirs = []
      for filename in dirs:
            path,name = posixpath.split(filename)
            if name == 'RCS' or name == 'sec' or name == "concepts" or name  == "SCCS" : continue
            if posixpath.isdir(filename):
                  mandirs.append(filename)
      return mandirs

def addtomanualpage(name,file,args,rep):
      print 'Opening '+file+' to add '+name
      f = open(file)
      page = f.read()
      f.close()

      if page.find('C++ variants') == -1:
        #look for Notes: bullet
        loc = page.find('<H3><FONT COLOR="#CC3333">Notes</FONT></H3>')
        if loc == -1:
           #look for See Also: bullet
           loc = page.find('<H3><FONT COLOR="#CC3333">See Also</FONT></H3>')
           if loc == -1:
             #look for Level bullet in page
             loc = page.find('<P><B><P><B><FONT COLOR="#CC3333">Level:</FONT></B>')
             if loc == -1:
               print 'Level bullet is missing in '+file
               return
        page = page[0:loc]+'<H3><FONT COLOR="#CC3333">C++ variants</FONT></H3><TABLE border="0" cellpadding="0" cellspacing="0">\n</TABLE>\n'+page[loc:]

      # now add variant 
      loc  = page.find('<H3><FONT COLOR="#CC3333">C++ variants</FONT></H3>')
      page = page[0:loc]+'<H3><FONT COLOR="#CC3333">C++ variants</FONT></H3><TABLE border="0" cellpadding="0" cellspacing="0">\n'+\
                        '<TR><TD WIDTH=40></TD><TD>'+name+args+'<TD WIDTH=20></TD><TD>-></TD><TD WIDTH=20></TD><TD>'+rep+'</TR></TD>\n'+page[loc+100:]
      
      f = open(file,'w')
      f.write(page)
      f.close()
      

# Extracts PETSC_DIR from the command line and
# starts genrating index for all the manpages.
def main():
      arg_len = len(argv)
      
      if arg_len < 3: 
            print 'Error! Insufficient arguments.'
            print 'Usage:', argv[0], 'PETSC_DIR','LOC'
            exit()

      PETSC_DIR = argv[1]
      LOC       = argv[2]

      # generate dictionary of all manual pages and their file location
      dirs      = glob.glob(LOC + '/docs/manualpages/*')
      mandirs   = getallmandirs(dirs)
      keys      = {}
      for m in mandirs:
            files = glob.glob(m+'/*.html')
            for f in files:
                  keys[os.path.basename(f)[0:-5]] = f
      poly = open('tmppoly')
      f = poly.readline()
      while f:
            name = f[0:f.find('(')-1]
            if not keys.has_key(name):
                  print 'Polymorphic function '+name+' does not have matching manual page'
            else:
                  args = f[f.find('('):f.find(')')+1]
                  rep  = f[f.find('(',f.find('(')+1):f.find(')',f.find(')')+1)+1]                  
                  addtomanualpage(name,keys[name],args,rep)
            f = poly.readline()
      poly.close()
      
# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
      main()
    

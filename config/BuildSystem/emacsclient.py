#
# Uses the emacsclient feature to allow editing a string
# We should reverse engineer the emacs lisp (looks fairly easy)
# to skip the file completely
#
import commands
import os
import pwd

def edit(astring):
  filename = os.path.join('/tmp', pwd.getpwuid(os.getuid())[0]+'-emacsclient')
  
  f = open(filename,'w')
  f.write(astring)
  f.close()
  (status, output) = commands.getstatusoutput('emacsclient '+filename)
  if status:
    print 'Problem running emacsclient'
    print output
    return astring
  f = open(filename,'r')
  astring = f.read()
  f.close()
  return astring

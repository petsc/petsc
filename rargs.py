
import atexit
import cPickle
import os
import re
import string
import sys
import types
import UserDict
import readline
import socket

class RArgs (UserDict.UserDict):
  def __init__(self,name = "default",readpw = "open",dictpw = "open",addpw = "open",writepw = "open"):
    UserDict.UserDict.__init__(self)
    self.name    = name
    if dictpw  == "open": dictpw  = readpw
    if addpw   == "open": addpw   = dict
    if writepw == "open": writepw = addpw
    self.readpw  = readpw
    self.dictpw  = dictpw
    self.addpw   = addpw
    self.writepw = writepw
    self.addr    = self.getServerAddr()
    
  def getServerAddr(self):
    import dargs
    filename = os.path.join(os.path.dirname(sys.modules['dargs'].__file__), 'DArgs.loc')
    if os.path.exists(filename):
      f    = open(filename, 'r')
      addr = cPickle.load(f)
      f.close()
    else:
     raise RuntimeError,"No running server"
    return addr

  def __setitem__(self,key,value):
    try:
      self.send(("__setitem__",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw,value))
    except:
      pass
    
  def __getitem__(self, key):
    try:
      obj = self.send(("__getitem__",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      pass
    if obj[0] == 1:
      return obj[1]
    else:
      raise KeyError
    
  def __delitem__(self, key):
    try:
      obj = self.send(("__delitem__",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      pass

  def has_key(self, key):
    try:
      obj = self.send(("has_key",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      pass
    if obj[0] == 1:
      return 1
    else:
      return None

  def clear(self):
    try:
      obj = self.send(("clear",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      pass

  def keys(self):
    try:
      obj = self.send(("keys",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      pass
    return obj[1]

  def dicts(self):
    try:
      obj = self.send(("dicts",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      pass
    return obj[1]

  def __len__(self):
    try:
      obj = self.send(("__len__",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      pass
    return obj[0]


  def send(self,object):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # if the file DArgs.loc exists but no server is running we are screwed
    try:
      s.connect(self.addr)
    except:
      self.addr = self.getServerAddr()
      try:
        s.connect(self.addr)
      except:
        raise RuntimeError,"Cannot connect to server"
              
    f = s.makefile("w")
    cPickle.dump(object,f)
    f.close()
    f = s.makefile("r")
    object = cPickle.load(f)
    f.close()
    s.close()
    return object
    
if __name__ ==  '__main__':
    A = RArgs("A")
    A['hi'] = 'low'
    print A['hi']             
    print A.has_key('joe')
    print A.has_key('hi')
    del A['hi']
    del A['joe']
    print A.has_key('hi')
    A[22] = 'newlow'
    print A[22]
    print A.keys()
    print len(A)
    print A.dicts()
    for d in RArgs().dicts():
      b = RArgs(d)
      for k in b.keys():
        print d+" "+str(k)+" "+str(b[k])
    A.clear()
        

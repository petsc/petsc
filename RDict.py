#!/usr/bin/env python
#
#   RDict - A remote dictionary server
#
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
import SocketServer
import socket
import time


#  These are the remote dictionaries
class Args (UserDict.UserDict):
  def __init__(self,name,readpw,addpw,writepw):
    UserDict.UserDict.__init__(self)
    self.name    = name
    self.readpw  = readpw
    self.addpw   = addpw
    self.writepw = writepw

#  This handles requests from the client to store or access data in
# the remote dictionaries
class ProcessHandler(SocketServer.StreamRequestHandler):
  def handle(self):
    object  = cPickle.load(self.rfile)
    #  all messages are of the form ("request",dict,key,readpw,addpw,writepw <,value>)
    request = object[0]
    name    = object[1]
    key     = object[2]
    readpw  = object[3]
    dictpw  = object[4]
    addpw   = object[5]
    writepw = object[6]
    
    dargs = self.server.dargs
    dargs.logfile.write("Received "+request+" in "+str(name)+" "+" from "+self.client_address[0]+" "+time.asctime(time.localtime())+'\n')
    dargs.logfile.flush()
    if request == "__setitem__":
      if not dargs.data.has_key(name):
        if dargs.dictpw == dictpw:
          dargs.data[name] = Args(name,readpw,addpw,writepw)
        else:
          dargs.logfile.write("Rejected, wrong dictpw\n");
          dargs.logfile.flush()
          cPickle.dump((0,None),self.wfile)
          return
              
      if dargs.data[name].writepw == writepw or (dargs.data[name].addpw == addpw and not dargs.data[name].has_key(key)):
         dargs.data[name].data[key] = object[7]
      cPickle.dump((0,None),self.wfile)
        
    elif request == "__getitem__":
      if dargs.data.has_key(name) and dargs.data[name].data.has_key(key) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,dargs.data[name].data[key]),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary, key or wrong readpw\n");
        dargs.logfile.flush()
        cPickle.dump((0,None),self.wfile)

    elif request == "has_key":
      if dargs.data.has_key(name) and dargs.data[name].data.has_key(key) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,None),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary, key or wrong readpw\n");
        dargs.logfile.flush()
        cPickle.dump((0,None),self.wfile)

    elif request == "__len__":
      if dargs.data.has_key(name) and dargs.data[name].readpw == readpw:
        cPickle.dump((len(dargs.data[name].data),None),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary, or wrong readpw\n");
        dargs.logfile.flush()
        cPickle.dump((0,None),self.wfile)

    elif request == "keys":
      if dargs.data.has_key(name) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,dargs.data[name].data.keys()),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary, or wrong readpw\n");
        dargs.logfile.flush()
        cPickle.dump((0,None),self.wfile)

    elif request == "dicts":
      di = []
      for d in dargs.data.keys():
        if dargs.data[d].readpw == readpw:
          di.append(d)
      cPickle.dump((0,tuple(di)),self.wfile)

    elif request == "clear":
      if dargs.data.has_key(name) and dargs.data[name].writepw == writepw:
        dargs.data[name].data.clear()
      else:
        dargs.logfile.write("Rejected, missing dictionary, wrong writepw\n");
        dargs.logfile.flush()
      cPickle.dump((0,None),self.wfile)

    elif request == "__delitem__":
      if dargs.data.has_key(name):
        if dargs.data[name].writepw == writepw:
          try:
            del dargs.data[name].data[key]
          except:
            dargs.logfile.write("Rejected, missing key\n");
            dargs.logfile.flush()
        else:
          dargs.logfile.write("Rejected, wrong writepw\n");
          dargs.logfile.flush()
      else:
        dargs.logfile.write("Rejected, missing dictionary\n");
        dargs.logfile.flush()
      cPickle.dump((0,None),self.wfile)

#  This is the remote dictionary server
class DArgs:
  def __init__(self, dictpw = "open"):
    self.data      = UserDict.UserDict()
    self.filename  = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.db')
    self.load()
    self.dictpw    = dictpw
    self.logfile   = open(os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.log'),'a')

  def load(self):
    if self.filename and os.path.exists(self.filename):
      dbFile    = open(self.filename, 'r')
      self.data = cPickle.load(dbFile)
      dbFile.close()

  def save(self):
    dbFile = open(self.filename, 'w')
    cPickle.dump(self.data, dbFile)
    dbFile.close()
    filename = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.loc')
    try: os.unlink(filename)
    except: pass
    self.logfile.write("Shutting down\n")
    self.logfile.flush()
    self.logfile.close()

  def loop(self):
    # wish there was a better way to get a usable socket
    flag = "nosocket"
    p    = 1
    while p < 1000 and flag == "nosocket":
      try:
        server = SocketServer.TCPServer((socket.gethostname(),6000+p),ProcessHandler)
        flag   = "socket"
      except Exception, e:
        p = p + 1
    if flag == "nosocket":
      p = 1
      while p < 1000 and flag == "nosocket":
        try:
          server = SocketServer.TCPServer(('localhost', 6000+p), ProcessHandler)
          flag   = "socket"
        except Exception, e:
          p = p + 1
    if flag == "nosocket":
      raise RuntimeError,"Cannot get available socket"
        
    filename = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.loc')
    if os.path.exists(filename):
      raise RuntimeError,filename+" already exists, are you sure the server is not running?\n"+                                                      "remove the file and restart server"
    f = open(filename, 'w')
    cPickle.dump(server.server_address, f)
    f.close()

    atexit.register(self.save)
    self.logfile.write("Started server"+time.asctime(time.localtime())+'\n')
    self.logfile.flush()
    server.dargs = self
    server.serve_forever()
    
# =============================================================================================

#  This is the client end of a remote dictionary
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
    filename = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.loc')
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
      raise RuntimeError
    
  def __getitem__(self, key):
    try:
      obj = self.send(("__getitem__",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      raise RuntimeError
    if obj[0] == 1:
      return obj[1]
    else:
      raise KeyError
    
  def __delitem__(self, key):
    try:
      obj = self.send(("__delitem__",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      raise RuntimeError

  def has_key(self, key):
    try:
      obj = self.send(("has_key",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      raise RuntimeError
    if obj[0] == 1:
      return 1
    else:
      return None

  def clear(self):
    try:
      obj = self.send(("clear",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      raise RuntimeError

  def keys(self):
    try:
      obj = self.send(("keys",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      raise RuntimeError
    return obj[1]

  def dicts(self):
    try:
      obj = self.send(("dicts",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      raise RuntimeError
    return obj[1]

  def __len__(self):
    try:
      obj = self.send(("__len__",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except:
      raise RuntimeError
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
              
    try:
      f = s.makefile("w")
      cPickle.dump(object,f)
      f.close()
      f = s.makefile("r")
      object = cPickle.load(f)
      f.close()
      s.close()
    except:
      raise RuntimeError,"Unable to get results from server"
    return object

#  support pickling of nargs objects
from nargs import *

#===============================================================================================
if __name__ ==  '__main__':
  if len(sys.argv) < 2:
    print "RDict.py [server] or [client]"
    exit(0)
  if sys.argv[1] == "server":
    try:
      dargs = DArgs()
      dargs.loop()
    except Exception, e:
      print 'ERROR: '+str(e)
      sys.exit(1)
    sys.exit(0)
  elif sys.argv[1] == "client":
    print "Entries in server dictionary"
    try:
      for d in RArgs().dicts():
        for k in RArgs(d).keys():
          print d+" "+str(k)+" "+str(RArgs(d)[k].getValue(k))
    except Exception, e:
      print 'ERROR: '+str(e)
      sys.exit(1)
    sys.exit(0)
  else:
    print "Unknown option "+sys.argv[1]
    sys.exit(1)

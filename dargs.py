
import atexit
import cPickle
import os
import re
import string
import sys
import types
import UserDict

import SocketServer
import socket
import time

class Args (UserDict.UserDict):
  def __init__(self,name,readpw,writepw):
    UserDict.UserDict.__init__(self)
    self.name    = name
    self.readpw  = readpw
    self.writepw = writepw

class ProcessHandler(SocketServer.StreamRequestHandler):
  def handle(self):
    object  = cPickle.load(self.rfile)
    #  all messages are of the form ("request",dict,key,readpw,writepw <,value>)
    request = object[0]
    name    = object[1]
    key     = object[2]
    readpw  = object[3]
    writepw = object[4]

    global dargs
    dargs.logfile.write("Received request "+request+" in "+name+" "+time.asctime(time.localtime())+'\n')
    dargs.logfile.flush()
    if request == "__setitem__":
      if not dargs.data.has_key(name):
        dargs.data[name] = Args(name,readpw,writepw)
        
      if dargs.data[name].writepw == writepw:
         dargs.data[name].data[key] = object[5]
      cPickle.dump((0,None),self.wfile)
        
    elif request == "__getitem__":
      if dargs.data.has_key(name) and dargs.data[name].data.has_key(key) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,dargs.data[name].data[key]),self.wfile)
      else:
        cPickle.dump((0,None),self.wfile)

    elif request == "has_key":
      if dargs.data.has_key(name) and dargs.data[name].data.has_key(key) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,None),self.wfile)
      else:
        cPickle.dump((0,None),self.wfile)

    elif request == "__len__":
      if dargs.data.has_key(name) and dargs.data[name].readpw == readpw:
        cPickle.dump((len(dargs.data[name].data),None),self.wfile)
      else:
        cPickle.dump((0,None),self.wfile)

    elif request == "keys":
      if dargs.data.has_key(name) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,dargs.data[name].data.keys()),self.wfile)
      else:
        cPickle.dump((0,None),self.wfile)

    elif request == "clear":
      if dargs.data.has_key(name) and dargs.data[name].writepw == writepw:
        dargs.data[name].data.clear()
      cPickle.dump((0,None),self.wfile)

    elif request == "__delitem__":
      if dargs.data.has_key(name):
        if dargs.data[name].writepw == writepw:
          try:
            del dargs.data[name].data[key]
          except:
            pass
      cPickle.dump((0,None),self.wfile)

class DArgs:
  def __init__(self, filename = "DArgs.db"):
    self.data      = UserDict.UserDict()
    self.filename  = filename
    self.load(filename)
    self.logfile   = open("DArgs.log",'w')
    atexit.register(self.save)

  def load(self, filename):
    if filename and os.path.exists(filename):
      dbFile    = open(filename, 'r')
      self.data = cPickle.load(dbFile)
      dbFile.close()

  def save(self):
    dbFile = open(self.filename, 'w')
    cPickle.dump(self.data, dbFile)
    dbFile.close()

  def loop(self):
    server = SocketServer.TCPServer(("terra.mcs.anl.gov",6001),ProcessHandler)
    self.logfile.write("Started server"+time.asctime(time.localtime())+'\n')
    self.logfile.flush()
    server.serve_forever()
    

if __name__ ==  '__main__':
    global dargs
    dargs = DArgs()
    dargs.loop()

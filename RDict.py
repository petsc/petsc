#!/usr/bin/env python
#
#   RDict - A remote dictionary server
#
# This is necessary for us to create Project objects on load
import project

import atexit
import cPickle
import os
import re
import string
import sys
import types
import UserDict
try:
  import readline
except ImportError: pass
import time

try:
  import socket
  import SocketServer
  #  This handles requests from the client to store or access data in
  # the remote dictionaries
  class ProcessHandler(SocketServer.StreamRequestHandler):
    def handle(self):
      dargs = self.server.dargs
      dargs.logfile.write('Started handler for '+self.client_address[0]+' '+time.asctime(time.localtime())+'\n')
      dargs.logfile.flush()
      # The 'closed' attribute does not seem to work correctly
      try:
        loop = 1
        while loop and not self.rfile.closed:
          loop = self.handle_item()
      except Exception, e:
        dargs.logfile.write(str(e)+' '+self.client_address[0]+' '+time.asctime(time.localtime())+'\n')
        dargs.logfile.flush()
      dargs.logfile.write('Closed handler for '+self.client_address[0]+' '+time.asctime(time.localtime())+'\n')
      dargs.logfile.flush()
      return

    def handle_item(self):
      dargs   = self.server.dargs
      #dargs.logfile.write('Reading request from '+self.client_address[0]+' '+time.asctime(time.localtime())+'\n')
      #dargs.logfile.flush()
      object  = cPickle.load(self.rfile)
      #  all messages are of the form ("request",dict,key,readpw,addpw,writepw <,value>)
      request = object[0]
      name    = object[1]
      key     = object[2]
      readpw  = object[3]
      dictpw  = object[4]
      addpw   = object[5]
      writepw = object[6]

      dargs.logfile.write("Received "+request+" in "+str(name)+"("+key+") "+" from "+self.client_address[0]+" "+time.asctime(time.localtime())+'\n')
      dargs.logfile.flush()
      if request == "__getitem__":
        self.getitem(dargs, name, key, readpw)
      elif request == "__setitem__":
        self.setitem(dargs, name, key, readpw, dictpw, addpw, writepw)
      elif request == "__delitem__":
        self.delitem(dargs, name, key, writepw)
      elif request == "has_key":
        self.has_key(dargs, name, key, readpw)
      elif request == "__len__":
        self.len(dargs, name, readpw)
      elif request == "keys":
        self.keys(dargs, name, readpw)
      elif request == "dicts":
        self.dicts(dargs, readpw)
      elif request == "clear":
        self.clear(dargs, name, writepw)
      elif request == "stop":
        return 0
      return 1

    def getitem(self, dargs, name, key, readpw):
      if dargs.data.has_key(name) and dargs.data[name].data.has_key(key) and dargs.data[name].readpw == readpw:
        cPickle.dump((1, dargs.data[name].data[key]), self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary key or wrong readpw\n");
        dargs.logfile.flush()
        cPickle.dump((0, None), self.wfile)
      return

    def setitem(self, dargs, name, key, readpw, dictpw, addpw, writepw):
      if not dargs.data.has_key(name):
        if dargs.dictpw == dictpw:
          dargs.data[name] = Args(name, readpw, addpw, writepw)
          dargs.saveifold()
        else:
          dargs.logfile.write("Rejected, wrong dictpw\n");
          dargs.logfile.flush()
          cPickle.dump((0, None), self.wfile)
          return

      if dargs.data[name].writepw == writepw or (dargs.data[name].addpw == addpw and not dargs.data[name].has_key(key)):
        dargs.data[name].data[key] = object[7]
        dargs.saveifold()
      cPickle.dump((0, None), self.wfile)
      return

    def delitem(self, dargs, name, key, writepw):
      if dargs.data.has_key(name):
        if dargs.data[name].writepw == writepw:
          try:
            del dargs.data[name].data[key]
            dargs.saveifold()
          except KeyError:
            dargs.logfile.write("Rejected, missing key\n");
            dargs.logfile.flush()
        else:
          dargs.logfile.write("Rejected, wrong writepw\n");
          dargs.logfile.flush()
      else:
        dargs.logfile.write("Rejected, missing dictionary\n");
        dargs.logfile.flush()
      cPickle.dump((0,None),self.wfile)
      return

    def has_key(self, dargs, name, key, readpw):
      if dargs.data.has_key(name) and dargs.data[name].data.has_key(key) and dargs.data[name].readpw == readpw:
        cPickle.dump((1, None), self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary key or wrong readpw\n");
        dargs.logfile.flush()
        cPickle.dump((0, None), self.wfile)
      return

    def len(self, dargs, name, readpw):
      if dargs.data.has_key(name) and dargs.data[name].readpw == readpw:
        cPickle.dump((len(dargs.data[name].data),None),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary or wrong readpw\n");
        dargs.logfile.flush()
        cPickle.dump((0, None), self.wfile)
      return

    def keys(self, dargs, name, readpw):
      if dargs.data.has_key(name) and dargs.data[name].readpw == readpw:
        cPickle.dump((1, dargs.data[name].data.keys()), self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary or wrong readpw\n");
        dargs.logfile.flush()
        cPickle.dump((0, None), self.wfile)
      return

    def dicts(self, dargs, readpw):
      di = []
      for d in dargs.data.keys():
        if dargs.data[d].readpw == readpw:
          di.append(d)
      cPickle.dump((0, tuple(di)), self.wfile)
      return

    def clear(self, dargs, writepw):
      if dargs.data.has_key(name) and dargs.data[name].writepw == writepw:
        dargs.data[name].data.clear()
        dargs.saveifold()
      else:
        dargs.logfile.write("Rejected, missing dictionary or wrong writepw\n");
        dargs.logfile.flush()
      cPickle.dump((0, None), self.wfile)
      return
except ImportError: pass

#
#   This is called by the timer savedelay seconds after a database update
import threading
def TimerSave(dargs):
  dargs.timer = 0
  dargs.save()

#  These are the remote dictionaries
class Args (UserDict.UserDict):
  def __init__(self,name,readpw,addpw,writepw):
    UserDict.UserDict.__init__(self)
    self.name    = name
    self.readpw  = readpw
    self.addpw   = addpw
    self.writepw = writepw

#  This is the remote dictionary server
class DArgs:
  def __init__(self, dictpw = "open"):
    self.dictpw    = dictpw
    self.savedelay = 30
    self.timer     = 0
    self.data      = UserDict.UserDict()
    self.logfile   = open(os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.log'), 'a')
    self.logfile.write('Greetings\n')
    self.filename  = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.db')
    self.load()
    return

  def load(self):
    if self.filename and os.path.exists(self.filename):
      try:
        dbFile    = file(self.filename)
        self.data = cPickle.load(dbFile)
        dbFile.close()
        self.logfile.write('Loaded dictionary from '+self.filename+'\n')
      except Exception, e:
        self.logfile.write('Problem loading dictionary from '+self.filename+'\n--> '+str(e)+'\n')
    else:
      self.logfile.write('No dictionary to load in this file: '+self.filename+'\n')
    return

#  Start new save timer running if it is not already running
  def saveifold(self):
    if not self.timer:
      self.timer = threading.Timer(self.savedelay,TimerSave,[self],{})
      self.timer.start()

  def save(self):
    dbFile = open(self.filename, 'w')
    cPickle.dump(self.data, dbFile)
    dbFile.close()
    self.lastsave = time.time()

  def shutdown(self):
    self.save()
    filename = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.loc')
    if os.path.isfile(filename): os.unlink(filename)
    self.logfile.write("Shutting down\n")
    self.logfile.flush()
    self.logfile.close()

  def loop(self):
    # wish there was a better way to get a usable socket
    flag = 'nosocket'
    p    = 1
    while p < 1000 and flag == 'nosocket':
      try:
        server = SocketServer.TCPServer((socket.gethostname(), 6000+p), ProcessHandler)
        flag   = 'socket'
      except Exception, e:
        p = p + 1
    if flag == 'nosocket':
      p = 1
      while p < 1000 and flag == 'nosocket':
        try:
          server = SocketServer.TCPServer(('localhost', 6000+p), ProcessHandler)
          flag   = 'socket'
        except Exception, e:
          p = p + 1
    if flag == 'nosocket':
      raise RuntimeError,'Cannot get available socket'
        
    filename = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.loc')
    if os.path.exists(filename):
      # check if server is running
      running = 1
      try: RArgs().dicts()
      except: running = 0
      if running: raise RuntimeError, 'Server already running'
    f = open(filename, 'w')
    cPickle.dump(server.server_address, f)
    f.close()

    atexit.register(self.shutdown)
    self.logfile.write('Started server'+time.asctime(time.localtime())+'\n')
    self.logfile.flush()
    server.dargs = self
    server.serve_forever()
    return

# =============================================================================================

#  This is the client end of a remote dictionary
class RArgs (UserDict.UserDict):
  def __init__(self, name = "default", readpw = "open", dictpw = "open", addpw = "open", writepw = "open", addr = None, purelocal = 0):
    UserDict.UserDict.__init__(self)
    self.name    = name
    if dictpw  == "open": dictpw  = readpw
    if addpw   == "open": addpw   = dictpw
    if writepw == "open": writepw = addpw
    self.readpw  = readpw
    self.dictpw  = dictpw
    self.addpw   = addpw
    self.writepw = writepw
    if not purelocal:
      if addr is None:
        self.addr  = self.getServerAddr()
      else:
        self.addr  = (addr[0], int(addr[1]))
    self.s          = None
    self.stopObj    = cPickle.dumps(('stop', self.name, 'dummykey', self.readpw, self.dictpw, self.addpw, self.writepw))
    self.persistent = 0
    self.debug      = 0
    if self.persistent:
      self.s = self.connectToServer()
    return

  def __del__(self):
    if self.persistent:
      f = self.s.makefile('w')
      f.write(self.stopObj)
      f.close()
      self.s.close()
    return

  def connectToServer(self):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      if self.debug: print 'Trying to connect to '+str(self.addr)
      s.connect(self.addr)
    except Exception, e:
      if self.debug: print 'Failed to connect: '+str(e)
      self.addr = self.getServerAddr()
      try:
        if self.debug: print 'Trying to connect again to '+str(self.addr)
        s.connect(self.addr)
      except Exception, e:
        # the file DArgs.loc exists but no server is running
        if self.debug: print 'ERROR: '+str(e)
        self.startServer()
        self.addr = self.getServerAddr()
        try:
          if self.debug: print 'Trying to connect to '+str(self.addr)
          s.connect(self.addr)
        except:
          raise RuntimeError('Cannot connect to server: '+str(e))
    if self.debug: print 'Connected to '+str(s)
    return s

  def startServer(sles):
    filename = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.loc')
    try: os.unlink(filename)
    except: pass
    os.spawnvp(os.P_NOWAIT,'python',['python',os.path.join(os.path.dirname(os.path.abspath(sys.modules['RDict'].__file__)),'RDict.py'),"server"])
    for i in range(10):
      time.sleep(1)
      if os.path.exists(filename): return
    raise RuntimeError,"No running server: Could not start it"

  def getServerAddr(self):
    filename = os.path.join(os.path.dirname(sys.modules['RDict'].__file__), 'DArgs.loc')
    if not os.path.exists(filename): self.startServer()
    f = open(filename, 'r')
    addr = cPickle.load(f)
    f.close()
    return addr

  def __setitem__(self,key,value):
    try:
      self.send(("__setitem__",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw,value))
    except Exception, e:
      raise RuntimeError(str(e))
    
  def __getitem__(self, key):
    try:
      obj = self.send(("__getitem__",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except Exception, e:
      raise RuntimeError(str(e))
    if obj[0] == 1:
      return obj[1]
    else:
      raise KeyError('Could not find '+key)
    
  def __delitem__(self, key):
    try:
      obj = self.send(("__delitem__",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except Exception, e:
      raise RuntimeError(str(e))

  def has_key(self, key):
    try:
      obj = self.send(("has_key",self.name,key,self.readpw,self.dictpw,self.addpw,self.writepw))
    except Exception, e:
      raise RuntimeError(str(e))
    if obj[0] == 1:
      return 1
    else:
      return None

  def clear(self):
    try:
      obj = self.send(("clear",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except Exception, e:
      raise RuntimeError(str(e))

  def keys(self):
    try:
      obj = self.send(("keys",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except Exception, e:
      raise RuntimeError(str(e))
    return obj[1]

  def dicts(self):
    try:
      obj = self.send(("dicts",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except Exception, e:
      raise RuntimeError(str(e))
    return obj[1]

  def __len__(self):
    try:
      obj = self.send(("__len__",self.name,"dummykey",self.readpw,self.dictpw,self.addpw,self.writepw))
    except Exception, e:
      raise RuntimeError(str(e))
    return obj[0]

  def send(self, object):
    if not self.persistent:
      self.s = self.connectToServer()
    for i in range(3):
      try:
        f = self.s.makefile('w')
        if self.debug: print 'Sending object: '+str(object)
        cPickle.dump(object, f)
        f.close()
        f = self.s.makefile('r')
        if self.debug: print 'Receiving object'
        object = cPickle.load(f)
        f.close()
        if not self.persistent:
          self.s.close()
          self.s = None
        return object
      except IOError, e:
        if self.debug: print 'IOError: '+str(e)
        if e.errno == 32:
          self.s = self.connectToServer()
        else:
          raise RuntimeError('IOError: '+str(e))
      except Exception, e:
        if self.debug: print 'Exception: '+str(e)+' : '+str(e.__class__)
        raise RuntimeError('Unable to get results from server: '+str(e))
    return None

#  support pickling of nargs objects
from nargs import *

#===============================================================================================
if __name__ ==  '__main__':
  try:
    if len(sys.argv) < 2:
      print 'RDict.py [server | client]'
    else:
      action = sys.argv[1]
      if action == 'server':
        dargs = DArgs()
        dargs.loop()
      elif action == 'client':
        print 'Entries in server dictionary'
        rdict = RArgs()
        for d in rdict.dicts():
          rdict.name = d
          for k in rdict.keys():
            print d+' '+str(k)+' '+str(rdict[k])
      elif action == 'clear':
        print 'Clearing remote dictionary database'
        RArgs('ArgDict').clear()
      else:
        sys.exit('Unknown action: '+sys.argv[1])
  except Exception, e:
    sys.exit(str(e))
  sys.exit(0)

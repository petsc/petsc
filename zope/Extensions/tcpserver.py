import SocketServer
from socket import *
from urllib import *
import string
import threading
from re import search
import pickle
import __builtin__
import os
import datetime

global startmsg
global users

class User(object):
	def __init__(self, msg):
		self.gs = msg
		self.glast = msg
		self.info = msg
		self.error = msg
		self.status = -1
	def addgs(self, mgs):
		self.gs += mgs
	def replacegs(self, msg):
		self.gs = msg
	def getgs(self):
		return self.gs
	def addglast(self, mgs):
		self.glast += mgs
	def replaceglast(self, msg):
		self.glast = msg
	def getglast(self):
		return self.glast
	def addinfo(self, mgs):
		self.info += mgs
	def replaceinfo(self, msg):
		self.info = msg
	def getinfo(self):
		return self.info
	def adderror(self, mgs):
		self.error += mgs
	def replaceerror(self, msg):
		self.error = msg
	def geterror(self):
		return self.error
	def getstatus(self):
		return self.status
	def replacestatus(self, i):
		self.status = i
	def replaceall(self, msg):
		self.gs = msg
		self.glast = msg
		self.info = msg
		self.error = msg

def removeUser(i):
	global users
	i = i.strip()
	del users[i]

def createuser(i):
	global users
	global startmsg
	i = i.strip()
	users[i] = User(startmsg)

#Socket handler
class RecHandler(SocketServer.StreamRequestHandler):
	def handle(self):
		global users
		global startmsg
		peer = self.client_address
		ip = peer[0]
		line = self.rfile.readline()
		linestrip = line.strip()
		joinline = "".join(line)
		if joinline.rfind("<<<user>>>") >= 0:
			joinline = joinline.lstrip("<<<user>>>")
			joinline = joinline.strip()
			n = joinline
			if users.has_key(n):
				curr = users[n]
			else:
				users[n] = User(startmsg)
				curr = users[n]
		infocheck = errorcheck = 1
		#Checks to see if this is the first time the strings are used
		if curr.getgs() == startmsg: 
			curr.replacegs("")
		alias = gethostbyaddr(peer[0])
		#Used to distingush between seperate communications
		start = "\n======= %s %s ======\n\n" % (alias[0], peer[1])
		curr.addgs(start)
		curr.replaceglast(start)
		while 1:
			line = self.rfile.readline()
			linestrip = line.strip()
			joinline = "".join(line)
			if joinline.rfind("<<<user>>>") >= 0:
				joinline = joinline.lstrip("<<<user>>>")
				n = joinline
				if users.has_key(n):
					curr = users[n]
				else:
					users[n] = User(startmsg)
					curr = users[n]
				continue
			# Check to see if the string is info or error output
			endinfo = search(r'\[[0-9]+\]', joinline)
			infocheckRE = (endinfo != None)
			errorinfo = joinline.rfind("PETSC ERROR:")
			if joinline.rfind("<<<start>>>") >= 0:
				curr.replacestatus(1)
				joinline = joinline.lstrip("<<<start>>>")
			if joinline.rfind("<<<end>>>") >= 0:
				curr.replacestatus(0)
				joinline = joinline.lstrip("<<<end>>>")
			if infocheckRE :
				if infocheck:
					if curr.getinfo() == startmsg:
						curr.replaceinfo("")
					infocheck = 0
					curr.addinfo(start)
				curr.addinfo(joinline.lstrip("<<info>>"))
			elif errorinfo >= 0:
				if errorcheck:
					if curr.geterror() == startmsg:
						curr.replaceerror("")
					errorcheck = 0
					curr.adderror(start)
				curr.adderror(joinline)
			else:
				curr.addglast(joinline)
				curr.addgs(joinline)
			if not string.strip(line):
				#Ending tag for a communication
				end = "\n========== END ==========\n\n"
				curr.addgs(end)
				curr.addglast(end)
				if not infocheck:
					curr.addinfo(end)
				if not errorcheck:
					curr.error(end)
				break

#Start the server and intilize the global variables
def runserver():
	global startmsg
	global users
	__builtin__.User = User
	startmsg = "No Output"
	d = User(startmsg)
	users = {"default" : d}
	serv = SocketServer.ThreadingTCPServer(("", 9999), RecHandler)
	sockfd = serv.fileno()
	thread = threading.Thread(target=serv.serve_forever)
	thread.setDaemon(1)
	thread.start()
	return "TCP server started"

def gettime():	
	today = datetime.datetime.now()
	cur = today.ctime()
	cur = cur.replace(" ", "")
	cur = cur.replace(":", ".")
	return cur

def createpickle(i):
	global users
	i = i.strip()
	petscdir = os.environ["PETSC_DIR"]
	cur = gettime()
	f = open(petscdir+"/zope/Extensions/pickle/"+i+"_"+cur, "w")
	pickle.dump(users[i], f)
	f.close()

def getpickles():
	petscdir = os.environ["PETSC_DIR"]
	files = os.listdir(petscdir + "/zope/Extensions/pickle/")
	return files

def unpickle(i):
	global users
	i = i.strip()
	petscdir = os.environ["PETSC_DIR"]
	f = open(petscdir+"/zope/Extensions/pickle/"+i)
	loc = i.split("_")
	users[loc[0]] = pickle.load(f)

def getgs(i):
	global users
	i = i.strip()
	return users[i].getgs()

def getgsn(i):
	global users
	i = i.strip()
	return users[i].getgs()

def getglast(i):
	global users
	i = i.strip()
	return users[i].getglast()

def getinfo(i):
	global users
	i = i.strip()
	return users[i].getinfo()

def geterror(i):
	global users
	i = i.strip()
	return users[i].geterror()

def checkuser(i):
	global users
	i = i.strip()
	if users.has_key(i):
		return "true"
	else:
		return "false"

def getstatus(i):
	global users
	i = i.strip()
	if checkuser(i) == 'false':
		createuser(i)
	stat = users[i].getstatus()
	if stat == -1:
		return "No Programs have been started"
	elif stat == 1:
		return "Active Program"
	else: 
		return "Not Active"

def clearoutput(i):
	global users
	global startmsg
	i = i.strip()
	if users.has_key(i):
		users[i].replaceall(startmsg)

if __name__ == '__main__':
	runserver()

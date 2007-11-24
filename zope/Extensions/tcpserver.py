import SocketServer
from socket import *
import urllib
import string
import threading
from re import search
import pickle
import __builtin__
import os
import datetime

global startmsg
global users
global pdir

#Structure used to store all the PETSc output information
class User(object):
	def __init__(self, msg):
		self.gs = msg
		self.glast = msg
		self.info = msg
		self.error = msg
		self.log = msg
		self.status = -1
		self.update = 1
		self.autopickle = 0
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
	def addlog(self, msg):
		self.log +=msg
	def replacelog(self, msg):
		self.log = msg
	def getlog(self):
		return self.log
	def getstatus(self):
		return self.status
	def replacestatus(self, i):
		self.status = i
	def replaceall(self, msg):
		self.gs = msg
		self.glast = msg
		self.info = msg
		self.error = msg
		self.log = msg
	def getupdate(self):
		return self.update
	def newinfo(self):
		self.update = 1
	def old(self):
		self.update = 0
	def ap(self):
		self.autopickle = 1
	def noap(self):
		self.autopickle = 0
	def returnap(self):
		return self.autopickle

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
		infocheck = logcheck = errorcheck = 1
		#Checks to see if this is the first time the strings are used
		if curr.getgs() == startmsg: 
			curr.replacegs("")
		try:
			alias = gethostbyaddr(peer[0])
		except:
			alias = peer[0]
		#Used to distingush between seperate communications
		start = "\n======= %s %s ======\n\n" % (alias[0], peer[1])
		curr.addgs(start)
		curr.replaceglast(start)
		while 1:
			line = self.rfile.readline()
			linestrip = line.strip()
			joinline = "".join(line)
			# Check to see if the string is info or error output
			endinfo = search(r'\[[0-9]+\]', joinline)
			errorinfo = joinline.rfind("PETSC ERROR:")
			if joinline.find("<<<start>>>") >= 0:
				curr.replacestatus(1)
				joinline = joinline.strip("<<<start>>>")
			if joinline.find("<<<end>>>") >= 0:
				curr.replacestatus(0)
				joinline = joinline.strip("<<<end>>>")
			if endinfo != None :
				if infocheck:
					if curr.getinfo() == startmsg:
						curr.replaceinfo("")
					infocheck = 0
					curr.addinfo(start)
				curr.addinfo(joinline)
			elif joinline.find("<<<log>>>") >=0:
				if logcheck:
					if curr.getlog() == startmsg:
						curr.replacelog("")
					logcheck = 0
					curr.addlog(start)
				joinline = joinline.strip(" <<<log>>>");
				curr.addlog(joinline)
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
					curr.adderror(end)
				if not logcheck:
					curr.addlog(end)
				if curr.getstatus() == 0 and curr.returnap() == 1:
					createpickle(n, "noname")
				curr.newinfo()
				break

#Start the server and intilize the global variables
def runserver():
	global startmsg
	global users
	global pdir
	pdir = "/zope/pickle/"
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

#Return a formated timestap for the pickle file name
def gettime():	
	today = datetime.datetime.now()
	cur = today.ctime()
	cur = cur.replace(" ", "")
	cur = cur.replace(":", ".")
	return cur

#Create pickle of users current data
def createpickle(i, n):
	global users
	global pdir
	i = i.strip()
	n = n.strip()
	petscdir = os.environ["PETSC_DIR"]
	if n == "noname":
		cur = gettime()
	else:
		cur = n
	f = open(petscdir+pdir+i+"_"+cur, "w")
	pickle.dump(users[i], f)
	f.close()

#If a pickle directory does not exist, create one
def checkpickledir():
	e = os.environ["PETSC_DIR"]
	path = e+pdir
	if not os.access(path, os.F_OK):
		os.mkdir(path)

def getpickles():
	petscdir = os.environ["PETSC_DIR"]
	files = os.listdir(petscdir + pdir)
	return files

#Returns user information to previous state
def unpickle(i):
	global users
	i = i.strip()
	petscdir = os.environ["PETSC_DIR"]
	f = open(petscdir+pdir+i)
	loc = i.split("_")
	users[loc[0]] = pickle.load(f)
	users[loc[0]].newinfo()

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

def getlog(i):
	global users
	i = i.strip()
	return users[i].getlog()

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
	users[i].newinfo()

#Fork off a new process started by the user of the webpage
def startprog(path, args):
	if os.fork() == 0:
		path = path.strip()
		args = args.strip()
		args = args.replace(" ", "")
		args = args.split(",")
		a = ["", "-zope", "-nostdout"]
		a[3:] = args
		os.execv(path,a)
		exit(0)

def getupdate(i):
	global users
	i = i.strip()
	u = users[i].getupdate()
	if u:
		return "new"
	else:
		return "old"

def old(i):
	global users
	i = i.strip()
	users[i].old()

def setautopickle(i):
	global users
	i = i.strip()
	users[i].ap()

def setnoautopickle(i):
	global users
	i = i.strip()
	users[i].noap()
	
if __name__ == '__main__':
	runserver()

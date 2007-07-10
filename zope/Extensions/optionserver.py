from socket import *
import string

def sendoptions(iname, value):
	s = socket(AF_INET, SOCK_STREAM)
	try: 
		s.connect(('hookshot.mcs.anl.gov', 9998))
	except:
		return
	s.send(iname)
	buf = s.recv(256)
	s.send(value)
	s.close()

sendoptions("iname", "value")

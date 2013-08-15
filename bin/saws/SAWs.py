#!/usr/bin/env python
import os, sys
import requests
import json

#
#  If requests does not exist use sudo easy_install requests to install it.
#
host = os.getenv('SAWS_HOST')
if not host:
  host = 'localhost'
port = os.getenv('SAWS_PORT')
if not port:
  port = '8080'

url = 'http://'+host+':'+port+'/SAWs'

print url

r = requests.get(url)

print r.content

j = json.loads(r.content)

print j['directories']


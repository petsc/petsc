import sys

found = 0
prevnum = -1
fd = open(sys.argv[1])
for a in fd.readlines():
  f   = a[0:a.find(':')]
  num = int(a[a.find(':')+1:-2])
  if num == prevnum+1: found += 1
  if found == 2:
    print('Found double blank line '+str(num)+' '+f)
    found = 0
  prevnum = num

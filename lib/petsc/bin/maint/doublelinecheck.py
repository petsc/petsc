import sys

prevf   = ''
prevnum = -1
fd = open(sys.argv[1])
for a in fd.readlines():
  aa = a.split(':')
  f = aa[0]
  num = int(aa[1])
  if f == prevf and num == prevnum+1:
    print('Found double blank line '+str(num)+' '+f)
  prevf = f
  prevnum = num

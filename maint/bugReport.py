#!/usr/bin/env python
import os

class BugReporter(object):
  def __init__(self):
    self.mailServer = 'mail.mcs.anl.gov'
    self.maintAddr  = 'petsc-maint@mcs.anl.gov'
    if 'PETSC_DIR' in os.environ:
      self.dir  = os.environ['PETSC_DIR']
    else:
      self.dir  = None
    if 'PETSC_ARCH' in os.environ:
      self.arch = os.environ['PETSC_ARCH']
    else:
      self.arch = None
    if 'LOGNAME' in os.environ:
      self.user = os.environ['LOGNAME']
    if 'USER' in os.environ:
      self.user = os.environ['USER']
    else:
      self.user = 'unknown'
    return

  def writeHeader(self, writer):
    writer.addheader('Subject', 'Bug Report')
    writer.addheader('MIME-Version', '1.0')
    writer.startmultipartbody('mixed')

    # start off with a text/plain part
    part = writer.nextpart()
    body = part.startbody('text/plain')
    body.write('PETSc 2 bug report\n')
    body.write('PETSC_DIR:  '+str(self.dir)+'\n')
    body.write('PETSC_ARCH: '+str(self.arch)+'\n')
    return

  def writeFile(self, writer, mimetype, filename):
    import base64

    part = writer.nextpart()
    if os.path.isfile(filename):
      part.addheader('Content-Transfer-Encoding', 'base64')
      body = part.startbody(mimetype+'; name='+os.path.basename(filename))
      base64.encode(open(filename, 'r'), body)
    else:
      body = part.startbody('text/plain')
      body.write('Missing file '+filename+'\n')
    return

  def writeText(self, writer, textFile):
    return self.writeFile(writer, 'text/ascii', textFile)

  def writeImage(self, writer, imageFile):
    return self.writeFile(writer, 'image/jpeg', imageFile)

  def sendMail(self, message):
    import socket
    import smtplib

    smtp = smtplib.SMTP(self.mailServer)
    smtp.sendmail(self.user+'@'+socket.gethostname(), self.maintAddr, message)
    smtp.quit()
    return

  def sendBug(self):
    import MimeWriter
    import StringIO

    message = StringIO.StringIO()
    writer  = MimeWriter.MimeWriter(message)

    self.writeHeader(writer)
    if not self.dir is None:
      os.chdir(self.dir)
    self.writeText(writer, 'make.log')
    self.writeText(writer, os.path.join('bmake', str(self.arch), 'packages'))
    self.writeText(writer, os.path.join('bmake', str(self.arch), 'petscconf.h'))
    self.writeText(writer, os.path.join('bmake', str(self.arch), 'petscfix.h'))
    self.writeText(writer, os.path.join('bmake', str(self.arch), 'petscmachineinfo.h'))
    self.writeText(writer, os.path.join('bmake', str(self.arch), 'rules'))
    self.writeText(writer, os.path.join('bmake', str(self.arch), 'variables'))
    writer.lastpart()
    self.sendMail(message.getvalue())
    return

if __name__ == '__main__':
  BugReporter().sendBug()

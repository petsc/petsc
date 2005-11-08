#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-fpp=<name>', nargs.Arg(None, None, 'Specify Fortran preprocessor (broken)'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    return

  def configureFortranCPP(self):
    '''Handle case where Fortran cannot preprocess properly'''
    if hasattr(self.compilers, 'FC'):
      self.addMakeRule('.f.o .f90.o','',['-${FC} -c ${FFLAGS} ${FC_FLAGS} ${FOPTFLAGS} $<'])
      self.addMakeRule('.f.a',       '',['-${FC} -c ${FFLAGS} ${FC_FLAGS} ${FOPTFLAGS} $<', \
                                         '-${AR} ${AR_FLAGS} ${LIBNAME} $*.o', \
                                         '-${RM} $*.o'])

      if not self.compilers.fortranPreprocess:
        if self.compilers.isGCC:
          TRADITIONAL_CPP = '-traditional-cpp'
        else:
          TRADITIONAL_CPP = ''
        # Fortran does NOT handle CPP macros 
        self.addMakeRule('.F.o .F90.o','',['-@${RM} __$*.F __$*.c',\
                                           '-@${GREP} -v "^!" $*.F > __$*.c',\
                                           '-${CC} ${FCPPFLAGS} -E '+TRADITIONAL_CPP+' __$*.c | ${GREP} -v \'^ *#\' > __$*.F',\
                                           '-${FC} -c ${FFLAGS} ${FC_FLAGS} __$*.F -o $*.o',\
                                           '-if [ "${SKIP_RM}" != "yes" ] ;then  ${RM} __$*.F __$*.c ; fi'])
        self.addMakeRule('.F.a',       '',['-@${RM} __$*.F __$*.c',\
                                           '-@${GREP} -v "^!" $*.F > __$*.c',\
                                           '-${CC} ${FCPPFLAGS} -E '+TRADITIONAL_CPP+' __$*.c | ${GREP} -v \'^ *#\' > __$*.F',\
                                           '-${FC} -c ${FFLAGS} ${FC_FLAGS} __$*.F -o $*.o',\
                                           '-${AR} cr ${LIBNAME} $*.o',\
                                           '-${RM} __$*.F __$*.c'])
      else:
        # Fortran handles CPP macros correctly
        self.addMakeRule('.F.o .F90.o','',['-${FC} -c ${FFLAGS} ${FC_FLAGS} ${FCPPFLAGS} $<'])
        self.addMakeRule('.F.a','',       ['-${FC} -c ${FFLAGS} ${FC_FLAGS} ${FCPPFLAGS} $<',\
                                           '-${AR} ${AR_FLAGS} ${LIBNAME} $*.o',\
                                           '-${RM} $*.o'])

    else:
      self.addMakeRule('.F.o','',['-@echo "Your system was not configured for Fortran use"', \
                                  '-@echo "  Check configure.log under the checkFortranCompiler test for the specific failure"',\
                                  '-@echo "  You can reconfigure using --with-fc=<usable compiler> to enable Fortran"'])
    return

  def configure(self):
    self.executeTest(self.configureFortranCPP)
    return

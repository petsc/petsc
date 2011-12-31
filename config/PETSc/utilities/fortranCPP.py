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
    help.addArgument('PETSc', '-with-fortran-datatypes=<bool>', nargs.ArgBool(None, 0, 'Declare PETSc objects in Fortran like type(Vec) instead of Vec'))
    help.addArgument('PETSc', '-with-fortran-interfaces=<bool>', nargs.ArgBool(None, 0, 'Generate Fortran interface definitions for PETSc functions'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    return

  def configureFortranCPP(self):
    '''Handle case where Fortran cannot preprocess properly'''
    if hasattr(self.compilers, 'FC'):
      # these rules do not have preprocessing hence FCPPFLAGS is not used
      self.addMakeRule('.f.o .f90.o .f95.o','',['${PETSC_MAKE_STOP_ON_ERROR}${FC} -c ${FFLAGS} ${FC_FLAGS} -o $@ $<'])
      self.addMakeRule('.f.a',       '',['${PETSC_MAKE_STOP_ON_ERROR}${FC} -c ${FFLAGS} ${FC_FLAGS} $<', \
                                         '-${AR} ${AR_FLAGS} ${LIBNAME} $*.o', \
                                         '-${RM} $*.o'])

      if not self.compilers.fortranPreprocess:
        if self.compilers.isGCC:
          TRADITIONAL_CPP = '-traditional-cpp'
        else:
          TRADITIONAL_CPP = ''
        # Fortran does NOT handle CPP macros 
        self.addMakeRule('.F.o .F90.o .F95.o','',['-@${RM} __$< __$*.c',\
                                           '-@${GREP} -v "^!" $< > __$*.c',\
                                           '-${CC} ${FCPPFLAGS} -E '+TRADITIONAL_CPP+' __$*.c | ${GREP} -v \'^ *#\' > __$<',\
                                           '${PETSC_MAKE_STOP_ON_ERROR}${FC} -c ${FFLAGS} ${FC_FLAGS} __$< -o $*.o',\
                                           '-@if [ "${SKIP_RM}" != "yes" ] ;then  ${RM} __$< __$*.c ; fi'])
        self.addMakeRule('.F.a',       '',['-@${RM} __$< __$*.c',\
                                           '-@${GREP} -v "^!" $< > __$*.c',\
                                           '-${CC} ${FCPPFLAGS} -E '+TRADITIONAL_CPP+' __$*.c | ${GREP} -v \'^ *#\' > __$<',\
                                           '${PETSC_MAKE_STOP_ON_ERROR}${FC} -c ${FFLAGS} ${FC_FLAGS} __$< -o $*.o',\
                                           '-${AR} cr ${LIBNAME} $*.o',\
                                           '-${RM} __$< __$*.c'])
      else:
        # Fortran handles CPP macros correctly
        self.addMakeRule('.F.o .F90.o .F95.o','',['${PETSC_MAKE_STOP_ON_ERROR}${FC} -c ${FFLAGS} ${FC_FLAGS} ${FCPPFLAGS} -o $@ $<'])
        self.addMakeRule('.F.a','',       ['${PETSC_MAKE_STOP_ON_ERROR}${FC} -c ${FFLAGS} ${FC_FLAGS} ${FCPPFLAGS} $<',\
                                           '-${AR} ${AR_FLAGS} ${LIBNAME} $*.o',\
                                           '-${RM} $*.o'])

      if self.framework.argDB['with-fortran-datatypes']:
        self.fortranDatatypes = True
        self.addDefine('USE_FORTRAN_DATATYPES', '1')
      else:
        self.fortranDatatypes = False
      if self.framework.argDB['with-fortran-interfaces']:
        if self.framework.argDB['with-fortran-datatypes']:
          raise RuntimeError('Cannot use generated fortran interface definitions with fortran datatypes')
        self.addDefine('USE_FORTRAN_INTERFACES', '1')
      
    else:
      self.addMakeRule('.F.o','',['-@echo "Your system was not configured for Fortran use"', \
                                  '-@echo "  Check configure.log under the checkFortranCompiler test for the specific failure"',\
                                  '-@echo "  You can reconfigure using --with-fc=<usable compiler> to enable Fortran"'])
    return

  def configure(self):
    self.executeTest(self.configureFortranCPP)
    return

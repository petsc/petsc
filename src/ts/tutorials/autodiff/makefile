-include ../../../../petscdir.mk

LOCDIR          = src/ts/tutorials/autodiff/
EXAMPLESCXX     = adr_ex1.cxx adr_ex5adj.cxx adr_ex5adj_mf.cxx ex16adj.cxx ex16adj_tl.cxx ex16opt_ic.cxx
EXAMPLESF       =
MANSEC          = TS
DIRS            =
CLEANFILES      = TS-data-* *.tap

LIB             = ${PETSC_TS_LIB}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

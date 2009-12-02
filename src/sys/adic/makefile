
#requirespackage  'PETSC_HAVE_ADIC'
#requiresscalar real 
#requiresprecision double

ALL: lib

SOURCEH  =  ../../../include/adic/ad_deriv.h             ../../../include/adic/run-map.h \
            ../../../include/adic/ad_grad.h              ../../../include/adic/ad_grad_daxpy.h \
            ../../../include/adic/ad_grad_macro_axpys.h  ../../../include/adic/ad_utils.h \
            ../../../include/adic/adintr_proto.h         ../../../include/adic/adintrinsics.h \
            ../../../include/adic/initcommon.h           ../../../include/adic/initenum.h \
            ../../../include/adic/knr-compat.h           ../../../include/adic/run-alloc.h
DIRS     = adic adintrinsics Gradient
LOCDIR   = src/sys/adic/
NOADIC   = false
MANSEC   = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

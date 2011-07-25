# $Id$ 

ALL: 

MANSEC		 = Unconstrained
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/unconstrained/examples/tutorials/


DOCS		 = 
DIRS		 = 

include ${TAO_DIR}/conf/tao_base

rosenbrock1: rosenbrock1.o tao_chkopts
	-${CLINKER} -o rosenbrock1 rosenbrock1.o ${TAO_LIB} 
	${RM} rosenbrock1.o


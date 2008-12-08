# -*- makefile -*-

F2PY = f2py --fcompiler=gnu95
F2PY_FLAGS =
F2PY_FLAGS = -DF2PY_REPORT_ATEXIT -DF2PY_REPORT_ON_ARRAY_COPY=0
F2PY_FLAGS =--noarch --f90flags=''
F2PY_FLAGS +=-DF2PY_REPORT_ON_ARRAY_COPY=1

MODULE= bratu2df90

${MODULE}: ${MODULE}.so
${MODULE}.so: ${MODULE}.f90
	${F2PY} ${F2PY_FLAGS} -c $< -m ${MODULE}

clean:
	${RM} ${MODULE}.so *.py[co]

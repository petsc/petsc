# $Id: makefile,v 1.34 1998/01/12 16:03:20 bsmith Exp balay $ 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  =
SOURCEF  =
SOURCEH  = 
DOCS     = tex/manual/manual.tex manualpages/manualpages.html tex/manual/part1.tex \
	   tex/manual/part2.tex tex/manual/intro.tex tex/manual/petsc.bib \
	   tex/manual/abstract.tex tex/manual/acknowl.tex \
	   tex/manual/manual_tex.tex \
           tex/petsc-patches.html tex/petsc-benchmarks.html tex/petsc-components.html
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = ${SOURCEH}
DIRS     = 
LOCDIR   = docs/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

runexamples:



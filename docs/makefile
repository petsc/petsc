# $Id: makefile,v 1.42 2000/08/17 04:53:53 bsmith Exp bsmith $ 

CFLAGS   =
FFLAGS   =
SOURCEC  =
SOURCEF  =
SOURCEH  = 
DOCS     = manualpages/sec/* manualpages/index.html tex/manual/manual.tex \
	   tex/manual/part1.tex tex/manual/part2.tex tex/manual/intro.tex \
	   tex/manual/abstract.tex tex/manual/acknowl.tex tex/manual/manual_tex.tex \
	   tex/petsc.bib tex/petsc-patches.html tex/petsc-benchmarks.html \
	   tex/petsc-components.html tex/petsc-patches*
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = ${SOURCEH}
DIRS     = 
LOCDIR   = docs/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

bibtex2html:
	/home/petsc/software/bibtex2html-1.44-linux/bibtex2html -nokeys -rawurl tex/petscapps.bib

bib2html:
	${RM} petscapps.html
	/home/gropp/public_html/bib2html tex/petscapps.bib > petscapps.html



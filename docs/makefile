# $Id: makefile,v 1.47 2001/02/20 15:21:09 bsmith Exp bsmith $ 

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

bib2html:
	${RM} /mcs/www-unix/petsc/petscapps.html /mcs/www-unix/petsc/petscapps-bib.html
	/home/gropp/public_html/bib2html -group tex/doctext/petsc.group -listfile /mcs/www-unix/petsc/petscapps.html -listconfig tex/doctext/petscapps.cfg tex/petscapps.bib -nospecials > /mcs/www-unix/petsc/petscapps-bib.html
	chmod g+w /mcs/www-unix/petsc/petscapps.html /mcs/www-unix/petsc/petscapps-bib.html


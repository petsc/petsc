# $Id: makefile,v 1.49 2001/08/22 18:03:15 balay Exp $ 

CFLAGS   =
FFLAGS   =
SOURCEC  =
SOURCEF  =
SOURCEH  = 
DOCS     = manualpages/sec/bop.* tex/manual/manual.tex \
	   tex/manual/part1.tex tex/manual/part2.tex tex/manual/intro.tex \
	   tex/manual/abstract.tex tex/manual/acknowl.tex \
	   tex/petsc.bib tex/petsc-patches.html tex/petsc-benchmarks.html \
	   tex/petsc-components.html tex/petsc-patches*
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = ${SOURCEH}
DIRS     = 
LOCDIR   = docs/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

bib2html:
	${RM} /mcs/www-unix/petsc/petscapps.html /mcs/www-unix/petsc/petscapps-bib.html
	/home/gropp/public_html/bib/bib2html -group tex/doctext/petsc.group -listfile /mcs/www-unix/petsc/petscapps.html -listconfig tex/doctext/petscapp.cfg tex/petscapp.bib -nospecials > /mcs/www-unix/petsc/petscapps-bib.html
	chmod g+w /mcs/www-unix/petsc/petscapps.html /mcs/www-unix/petsc/petscapps-bib.html


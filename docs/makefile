# $Id: makefile,v 1.49 2001/08/22 18:03:15 balay Exp $ 

CFLAGS   =
FFLAGS   =
SOURCEC  =
SOURCEF  =
SOURCEH  = 
DOCS     = manualpages/sec/bop.* tex/manual/manual.tex \
	   tex/manual/part1.tex tex/manual/part2.tex tex/manual/intro.tex \
	   tex/manual/abstract.tex tex/manual/acknowl.tex tex/petsc.bib
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = ${SOURCEH}
DIRS     = 
LOCDIR   = docs/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

# petscapps* get built with 'builddist' and 'rsync'ed over to www-fp via the crontab
bib2html:
	${RM} website/publications/petscapps.html website/publications/petscapps-bib.html
	/home/gropp/public_html/bib/bib2html -group tex/doctext/petsc.group -listfile website/publications/petscapps.html -listconfig tex/doctext/petscapp.cfg tex/petscapp.bib -nospecials > website/publications/petscapps-bib.html


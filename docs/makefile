#PETSC_DIR = ../

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  =
SOURCEF  =
SOURCEH  = tex/manual/manual.tex www/www.html tex/manual/part1.tex \
           tex/manual/part2.tex tex/manual/intro.tex tex/manual/petsc.bib \
           tex/manual/abstract.tex tex/manual/acknowl.tex \
           tex/manual/manual_tex.tex man/mandesc
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = 
LOCDIR   = docs/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

runexamples:



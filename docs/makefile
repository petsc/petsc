PETSC_DIR = ../

CFLAGS   = $(OPT) -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = tex/manual/manual.tex www/www.html tex/manual/part1.tex \
           tex/manual/part2.tex tex/manual/intro.tex \
           tex/manual/abstract.tex tex/manual/acknowl.tex \
           tex/manual/manual_tex.tex man/mandesc
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

runexamples:



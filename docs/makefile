IPETSCDIR = ../

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = tex/manual.tex
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

runexamples:

#
#   This rebuilds the WWW PETSc homepage and ftp site from 
# the master copies of the documents.
#
homepage:
	@echo "Making PETSc homepage"
	/usr/bin/cp  onepager.ps /home/ftp/pub/petsc/onepager.ps
	/usr/bin/cp  manual.ps /home/ftp/pub/petsc/manual.ps
	/usr/bin/cp  ../readme /home/ftp/pub/petsc/installation
	/usr/bin/cp  tex/ftp_readme /home/ftp/pub/petsc/README
	@echo "Finshed making PETSc homepage"

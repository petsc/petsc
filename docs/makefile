IPETSCDIR = ../

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = tex/manual.tex www.html tex/part1.tex tex/intro.tex
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
# the master copies of the documents. Since it must copy all the 
# manual pages; it may take a few minutes
#
homepage:
	@echo "Making PETSc homepage"
	-/usr/bin/cp  onepager.ps /home/ftp/pub/petsc/onepager.ps
	-/usr/bin/cp  manual.ps /home/ftp/pub/petsc/manual.ps
	-/usr/bin/cp  ../readme /home/ftp/pub/petsc/installation
	-/usr/bin/cp  tex/ftp_readme /home/ftp/pub/petsc/README
	-/usr/bin/cp  ../Machines /home/ftp/pub/petsc
	-/usr/bin/cp  ../Changes /home/ftp/pub/petsc
	-/usr/bin/cp tex/papers/mpi.ps /home/ftp/pub/petsc/mpidev.ps
	-chmod ug+w /home/ftp/pub/petsc/installation /home/ftp/pub/petsc/README
	-chmod ug+w /home/ftp/pub/petsc/Machines /home/ftp/pub/petsc/manual.ps
	-chmod ug+w /home/ftp/pub/petsc/onepager.ps 
	-chmod ug+w /home/ftp/pub/petsc/intro.ps
	-chmod ug+w /home/ftp/pub/petsc/Changes /home/ftp/pub/petsc/mpidev.ps
	-/usr/bin/cp tex/petsc.html /mcs/www/petsc
	-/usr/bin/cp tex/petsc_color.gif /mcs/www/petsc
	-/usr/bin/cp tex/petsc.map /mcs/www/petsc
	-/usr/bin/cp manual.html /mcs/www/petsc
	-/usr/bin/cp tex/onepager.html /mcs/www/petsc
	-/usr/bin/cp tex/summaries.html /mcs/www/petsc
	-/usr/bin/cp www.html /mcs/www/petsc
	-/usr/bin/cp tex/intro.html /mcs/www/petsc
	-/usr/bin/cp intro.ps /home/ftp/pub/petsc
	-/usr/bin/cp tex/papers/mpi.html /mcs/www/petsc/mpidev.html
	-/usr/bin/cp ../faq /home/ftp/pub/petsc
	-chmod ug+w /home/ftp/pub/petsc/faq /mcs/www/petsc/*
	-tar cf $(HOME)/www.tar www 
	-cd /mcs/www/petsc/; /bin/rm -rf www; tar xf $(HOME)/www.tar; 
	-cd /mcs/www/petsc/; chgrp -R pdetools www; chmod -R g+w www
	-/bin/rm $(HOME)/www.tar
	@echo "Finshed making PETSc homepage"

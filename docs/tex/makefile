# $Id: makefile,v 1.52 1997/11/25 05:43:09 bsmith Exp bsmith $ 

onepager.ps: onepager.tex
	latex onepager
	dvips onepager
	rm -r onepager.dvi onepager.aux onepager.log
	-@chmod g+w onepager.ps
	-mv onepager.ps ..

onepager.html: onepager.tex
	tohtml onepager.tex -notopnames -nocontents -nonavnames 
	tohtml onepager.tex -notopnames -nocontents -nonavnames 
	rm -f *.gif *.xbm latex.err
	-@chmod g+w onepager.html

clean:
	rm -f manual/routines.ps */*.dvi */*.aux */*.toc */*.log */*.bbl */*.blg \
              */*.hux */*.err */*.ilg */*.idx */*.ind manual/sindex.tex manual/findex.tex \
              */*/*.dvi */*/*.aux */*/*.toc */*/*.log */*/*.bbl */*/*.blg \
              */*/*.hux */*/*.err */*/*.ilg */*/*.idx */*/*.ind */*/talk.ps */*/talk4.ps \
              */*/paper.ps */*/talk6.ps */*~ */*/*~ */*/*.stolen *.stolen
#
#   This rebuilds the WWW PETSc homepage and ftp site from 
# the master copies of the documents. Since it must copy all the 
# manual pages; it may take a few minutes
#
homepage:
	@echo "Making PETSc homepage"
	-/usr/bin/cp  ../manual.ps /home/ftp/pub/petsc/manual.ps
	-/usr/bin/cp intro.ps /home/ftp/pub/petsc
	-/usr/bin/cp  ../installation.html /mcs/www/petsc/docs
	-/usr/bin/cp  ../machines.html /mcs/www/petsc/docs
	-/usr/bin/cp  ../changes.html /mcs/www/petsc/docs
	-/usr/bin/cp  ../codemanagement.html /mcs/www/petsc/docs
	-/usr/bin/cp  ../troubleshooting.html /mcs/www/petsc/docs
	-/usr/bin/cp  ../faq.html /mcs/www/petsc/docs
	-/usr/bin/cp petsc.html /mcs/www/petsc
	-/usr/bin/cp petsc-patches.html /mcs/www/petsc
	-/usr/bin/cp petscwww.gif /mcs/www/petsc
	-/usr/bin/cp ../manual.html /mcs/www/petsc
	-/usr/bin/cp -r ../manual /mcs/www/petsc
	-/usr/bin/cp onepager.html /mcs/www/petsc
	-chmod ug+w /home/ftp/pub/petsc/* /mcs/www/petsc/*
	-cd /mcs/www/petsc/; chgrp -R pdetools docs/manualpages; chmod -R g+w docs/manualpages
	@echo "Finshed making PETSc homepage: remember to do the source! and manpages!"




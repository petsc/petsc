
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
	rm -f routines.ps *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err \
                          *.ilg *.idx *.ind sindex.tex findex.tex


#
#   This rebuilds the WWW PETSc homepage and ftp site from 
# the master copies of the documents. Since it must copy all the 
# manual pages; it may take a few minutes
#
homepage:
	@echo "Making PETSc homepage"
	-/usr/bin/cp  ../onepager.ps /home/ftp/pub/petsc/onepager.ps
	-/usr/bin/cp  ../manual.ps /home/ftp/pub/petsc/manual.ps
	-/usr/bin/cp intro.ps /home/ftp/pub/petsc
	-/usr/bin/cp  ftp_readme /home/ftp/pub/petsc/README
	-/usr/bin/cp  ../../Installation /home/ftp/pub/petsc/Installation
	-/usr/bin/cp  ../../Machines /home/ftp/pub/petsc
	-/usr/bin/cp  ../../Changes /home/ftp/pub/petsc
	-/usr/bin/cp  ../../CodeManagement /home/ftp/pub/petsc
	-/usr/bin/cp  ../../Troubleshooting /home/ftp/pub/petsc
	-/usr/bin/cp ../../FAQ /home/ftp/pub/petsc
	-/usr/bin/cp ../../Performance /home/ftp/pub/petsc
	-/usr/bin/cp petsc.html /mcs/www/petsc
	-/usr/bin/cp petscwww.gif /mcs/www/petsc
	-/usr/bin/cp petsc.map /mcs/www/petsc
	-/usr/bin/cp ../manual.html /mcs/www/petsc
	-/usr/bin/cp onepager.html /mcs/www/petsc
	-/usr/bin/cp summaries.html /mcs/www/petsc
	-/usr/bin/cp talks/supercomputing95/petscview.ps /home/ftp/pub/petsc
	-/usr/bin/cp talks/supercomputing95/petscview2.ps /home/ftp/pub/petsc
	-chmod ug+w /home/ftp/pub/petsc/FAQ /mcs/www/petsc/*
	-chmod ug+w /home/ftp/pub/petsc/Installation /home/ftp/pub/petsc/README
	-chmod ug+w /home/ftp/pub/petsc/Machines /home/ftp/pub/petsc/manual.ps
	-chmod ug+w /home/ftp/pub/petsc/onepager.ps 
	-chmod ug+w /home/ftp/pub/petsc/intro.ps
	-chmod ug+w /home/ftp/pub/petsc/Troubleshooting
	-chmod ug+w /home/ftp/pub/petsc/Changes
	-chmod ug+w /home/ftp/pub/petsc/CodeManagement
	-chmod ug+w /home/ftp/pub/petsc/Performance
	-cd ..; tar cf $(HOME)/www.tar www 
	-cd /mcs/www/petsc/; /bin/rm -rf www; tar xf $(HOME)/www.tar 
	-cd /mcs/www/petsc/www; /bin/rm -rf *.cit 
	-cd /mcs/www/petsc/; chgrp -R pdetools www; chmod -R g+w www
	-/bin/rm $(HOME)/www.tar
	@echo "Finshed making PETSc homepage: remember to do the source!"

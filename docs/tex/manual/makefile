# $Id: makefile,v 1.53 2001/04/07 02:43:36 bsmith Exp $ 

# Currently works on linux

PDFELATEX = unsetenv TEXINPUTS; pdfelatex 
NOADIC    = true
SHELL     = csh
GONULL    = >& /dev/null

# Build the entire manual as well as the intro part
manual.pdf: intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex
# First build the manual
	${PDFELATEX} manual ${GONULL}
	bibtex manual
	${PDFELATEX} manual ${GONULL}
	makeindex manual
	sed -e 's/_/\\_/g' manual.ind > index.tex
	${PDFELATEX} manual  ${GONULL}
	${PDFELATEX} manual  ${GONULL}
	${PDFELATEX} manual 
	mv manual.pdf ../../manual.pdf
# Now make the intro part ... note that this depends on latexing the manual
	cp manual.aux intro.aux
	${PDFELATEX} intro 
	${MAKE} clean

clean:
	rm -f  *.aux *.toc *.log *.bbl *.blg  \
                          *.ilg *.idx *.ind index.tex *.out

manual:  intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex
	-@sed -e s%man+../%man+manualpages/% ${LOC}/docs/manualpages/manualpages.cit > /tmp/tmpmap
	-@cat /home/MPI/www/www.index >> /tmp/tmpmap
	-@${RM} part1tmp.tex part2tmp.tex
	-@cat part1.tex | /home/gropp/bin/linux/mapnames -map /tmp/tmpmap | sed -e s%\<A\ href=\"%\\\\href{%g -e s%\"\>%}{%g -e s%\</A\>%}%g > part1tmp.tex
	-@cat part2.tex | /home/gropp/bin/linux/mapnames -map /tmp/tmpmap | sed -e s%\<A\ href=\"%\\\\href{%g -e s%\"\>%}{%g -e s%\</A\>%}%g > part2tmp.tex
	${PDFELATEX} manual 
	bibtex manual
	${PDFELATEX} manual 
	makeindex manual
	sed -e 's/_/\\_/g' manual.ind > index.tex
	${PDFELATEX} manual 
	${PDFELATEX} manual 
	${PDFELATEX} manual 
	mv manual.pdf ../../manual.pdf
# Now make the intro part ... note that this depends on latexing the manual
	cp manual.aux intro.aux
	${PDFELATEX} intro 
	${MAKE} clean



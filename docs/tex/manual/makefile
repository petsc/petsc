# $Id: makefile,v 1.53 2001/04/07 02:43:36 bsmith Exp $ 

# Currently works on linux

PDFELATEX = unset TEXINPUTS; pdfelatex 
NOADIC    = true

# Build the entire manual as well as the intro part
manual.pdf: intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex
# First build the manual
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

clean:
	rm -f  *.aux *.toc *.log *.bbl *.blg  \
                          *.ilg *.idx *.ind index.tex *.out

developers.dvi: developers.tex
# First build the manual
	latex developers
	bibtex developers
	latex developers
	latex developers
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg *.ilg *.ind \
                    *.idx

developers.ps: 
	dvips developers
	mv developers.ps ../../developers.ps
	-@chmod g+w ../../developers.ps






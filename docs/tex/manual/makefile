# $Id: makefile,v 1.53 2001/04/07 02:43:36 bsmith Exp $ 

NOADIC    = true

# Build the entire manual as well as the intro part
manual.pdf: intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex
# First build the manual
	pdfelatex manual 
	bibtex manual
	pdfelatex manual 
	makeindex manual
	sed -e 's/_/\\_/g' manual.ind > index.tex
	pdfelatex manual 
	pdfelatex manual 
	pdfelatex manual 
	mv manual.pdf ../../manual.pdf
# Now make the intro part ... note that this depends on latexing the manual
	cp manual.aux intro.aux
	pdfelatex intro 
	-@chmod g+w intro.ps ../../manual.pdf
	rm -r *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err \
                          *.ilg *.idx *.ind index.tex *.aus *.out

clean:
	rm -f  *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err \
                          *.ilg *.idx *.ind index.tex  *.aus *.out

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






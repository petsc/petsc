rem $Id: makefile,v 1.53 1998/01/03 16:04:05 bsmith Exp $ 
rem
rem makes manual in pdf format
rem
        pdfelatex manual 
	bibtex manual
	pdfelatex manual 
	makeindex manual
	rm index.tex
	sed -e s@_@\\_@g -e "s@{theindex}@{theindex}\\\\addcontentsline{toc}{chapter}{Index}\\\\label{sec:index}@g" manual.ind > index.tex
	pdfelatex manual 
	pdfelatex manual 
	pdfelatex manual 
	mv manual.pdf ../../manual.pdf
rem
rem Now make the intro part ... note that this depends on latexing the manual
rem
	cp manual.aux intro.aux
	pdfelatex intro 
	chmod go+w intro.pdf ../../manual.pdf
	rm -r *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err *.ilg *.idx *.ind index.tex  *.aus *.out
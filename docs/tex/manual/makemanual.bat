rem makes manual in pdf format
        pdfelatex manual 
	bibtex manual
	pdfelatex manual 
	makeindex manual
	rm index.tex
	sed -e s@_@\\_@g manual.ind > index.tex
	pdfelatex manual 
	pdfelatex manual 
	pdfelatex manual 
	mv manual.pdf ../../manual.pdf
rem Now make the intro part ... note that this depends on latexing the manual
	cp manual.aux intro.aux
	pdfelatex intro 
	-@chmod g+w intro.pdf ../../manual.pdf
	rm -r *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err *.ilg *.idx *.ind index.tex  *.aus *.out

manual.dvi: manual.tex routines.red
	latex manual
	bibtex manual
	latex manual
	latex manual
	chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 

manual.ps: 
	dvips manual
	mv manual.ps ..
	chmod g+w ../manual.ps

manual.html: manual.tex
	tohtml -mapman ../www/www.cit manual.tex
	tohtml  -mapman ../www/www.cit manual.tex
	mv *.html ..
	mv *.xbm ..

clean:
	rm -f routines.ps *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err

routines.red: routines.tex rsum/rsum1.tex rsum/rsum2.tex rsum/rsum3.tex \
              rsum/rsum4.tex rsum/rsum5.tex rsum/rsum6.tex rsum/rsum7.tex \
              rsum/rsumx.tex
	rm -f rsum/ssum1.tex rsum/ssum2.tex rsum/ssum3.tex rsum/ssum4.tex \
              rsum/ssum5.tex rsum/ssum6.tex rsum/ssum7.tex rsum/ssumx.tex
	sort rsum/rsum1.tex > rsum/ssum1.tex
	sort rsum/rsum2.tex > rsum/ssum2.tex
	sort rsum/rsum3.tex > rsum/ssum3.tex
	sort rsum/rsum4.tex > rsum/ssum4.tex
	sort rsum/rsum5.tex > rsum/ssum5.tex
	sort rsum/rsum6.tex > rsum/ssum6.tex
	sort rsum/rsum7.tex > rsum/ssum7.tex
	sort rsum/rsumx.tex > rsum/ssumx.tex

routines.dvi: routines.red
	latex routines 
	chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 

routines.ps: routines.dvi 
	dvips routines.dvi
	chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 



manual.dvi: manual.tex routines.red
	latex manual
	bibtex manual
	latex manual
	makeindex manual
	echo  \\begin{theindex} > findex.tex
	echo \\indexspace >> findex.tex
	grep FUNCTION manual.ind | sed "s/FUNCTION/ /g"  | \
             sed 's/_/\\_/g' >> findex.tex
	echo \\end{theindex} >> findex.tex
	grep -v FUNCTION manual.ind | sed 's/_/\\_/g' > sindex.tex
	latex manual
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg *.ilg *.ind \
                    *.idx findex.tex sindex.tex

manual.ps: 
	dvips manual
	mv manual.ps ..
	-@chmod g+w ../manual.ps

manual.html: manual.tex part1.tex
	tohtml -mapman ../www/www.cit manual.tex -notopnames \
	       -mapref../www/petsc.cit -nobottomnav
	tohtml -mapman ../www/www.cit manual.tex -notopnames \
	       -mapref../www/petsc.cit -nobottomnav
	-@chmod g+w manual.hux latex.err manual.html
	-mv manual.html ..
	rm -f  *.xbm latex.err findex.tex sindex.tex

intro.dvi: intro.tex part1.tex
	latex intro
	bibtex intro
	latex intro
	-@chmod g+w intro.dvi intro.aux intro.bbl intro.blg intro.log

intro.ps: 
	dvips intro
	-@chmod g+w intro.ps

intro.html: intro.tex part1.tex
	tohtml -mapman ../www/www.cit intro.tex -notopnames \
	       -mapref../www/petsc.cit  -nobottomnav
	tohtml -mapman ../www/www.cit intro.tex -notopnames \
	       -mapref../www/petsc.cit  -nobottomnav
	-@chmod g+w intro.hux latex.err intro.html
	rm -f *.xbm latex.err findex.tex sindex.tex

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

routines.red: routines.tex rsum/rsum1.tex rsum/rsum2.tex rsum/rsum3.tex \
              rsum/rsum4.tex rsum/rsum5.tex rsum/rsum6.tex rsum/rsum7.tex \
              rsum/rsumx.tex
	rm -f rsum/ssum1.tex rsum/ssum2.tex rsum/ssum3.tex rsum/ssum4.tex \
              rsum/ssum5.tex rsum/ssum6.tex rsum/ssum7.tex rsum/ssumx.tex
	sort rsum/rsum1.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g' | \
                            sed 's/\\DeFn/}{\\DeFn/g' > rsum/ssum1.tex
	sort rsum/rsum2.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                            sed 's/\\DeFn/}{\\DeFn/g' > rsum/ssum2.tex
	sort rsum/rsum3.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                            sed 's/\\DeFn/}{\\DeFn/g' > rsum/ssum3.tex
	sort rsum/rsum4.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                            sed 's/\\DeFn/}{\\DeFn/g' > rsum/ssum4.tex
	sort rsum/rsum5.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                            sed 's/\\DeFn/}{\\DeFn/g' > rsum/ssum5.tex
	sort rsum/rsum6.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                            sed 's/\\DeFn/}{\\DeFn/g' > rsum/ssum6.tex
	sort rsum/rsum7.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                            sed 's/\\DeFn/}{\\DeFn/g' > rsum/ssum7.tex
	sort rsum/rsumx.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                            sed 's/\\DeFn/}{\\DeFn/g' > rsum/ssumx.tex

routines.dvi: routines.red
	latex routines 
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 

routines.ps: routines.dvi 
	dvips routines.dvi
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 


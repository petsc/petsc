
# Build the entire manual as well as the intro part
manual.dvi: intro.tex abstract.tex acknowl.tex manual_tex.tex part1.tex routines.red part2.tex
# First build the manual
	latex manual_tex
	bibtex manual_tex
	latex manual_tex
	makeindex manual_tex
	echo  \\begin{theindex} > findex.tex
	echo \\indexspace >> findex.tex
	grep FUNCTION manual_tex.ind | sed "s/FUNCTION/ /g"  | \
             sed 's/_/\\_/g' >> findex.tex
	echo \\end{theindex} >> findex.tex
	grep -v FUNCTION manual_tex.ind | sed 's/_/\\_/g' > sindex.tex
	latex manual_tex
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg *.ilg *.ind \
                    *.idx findex.tex sindex.tex
# Now make the intro part ... note that this depends on latexing the manual
	cp manual_tex.aux intro.aux
	latex intro
	-@chmod g+w intro.dvi intro.aux intro.log

manual.ps: 
	dvips manual_tex
	mv manual_tex.ps ../../manual.ps
	-@chmod g+w ../../manual.ps
# Now make the intro part
	dvips intro
	-@chmod g+w intro.ps
	mv intro.ps ../intro.ps

manual.html: manual.tex abstract.tex acknowl.tex part1.tex part2.tex
	rm -f www.cit
	cat ../../www/www.cit | sed 's/+man+../+man+www/g' > www.cit 
	tohtml -mapman www.cit manual.tex -notopnames \
	       -mapref ../../www/petsc.cit -nobottomnav
	tohtml -mapman www.cit manual.tex -notopnames \
	       -mapref ../../www/petsc.cit -nobottomnav
	-@chmod g+w manual.hux latex.err manual.html
	-mv manual.html manual_html.html
	cat manual_html.html | grep -v Up: | grep -v Next: | grep -v Previous: |sed 's/H1/H2/g' | sed 's/H0/H1/g' | sed 's/<pre><tt>ierr/<pre><tt>   ierr/g' | sed 's/<pre><tt>call/<pre><tt>   call/g' > manual.html
	-mv manual.html ../..
	-@chmod g+w ../../manual.html www.cit
	rm -f  *.xbm latex.err findex.tex sindex.tex tmp.html manual_html.html

clean:
	rm -f routines.ps *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err \
                          *.ilg *.idx *.ind sindex.tex findex.tex *.aus

routines.red: routines.tex ../rsum/rsum1.tex ../rsum/rsum2.tex \
              ../rsum/rsum3.tex \
              ../rsum/rsum4.tex ../rsum/rsum5.tex ../rsum/rsum6.tex \
              ../rsum/rsum7.tex \
              ../rsum/rsumx.tex
	rm -f ../rsum/ssum1.tex ../rsum/ssum2.tex ../rsum/ssum3.tex \
              ../rsum/ssum4.tex \
              ../rsum/ssum5.tex ../rsum/ssum6.tex ../rsum/ssum7.tex \
              ../rsum/ssumx.tex
	sort ../rsum/rsum1.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g' | \
                 sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssum1.tex
	sort ../rsum/rsum2.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssum2.tex
	sort ../rsum/rsum3.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssum3.tex
	sort ../rsum/rsum4.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssum4.tex
	sort ../rsum/rsum5.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssum5.tex
	sort ../rsum/rsum6.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssum6.tex
	sort ../rsum/rsum7.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssum7.tex
	sort ../rsum/rsumv.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumv.tex
	sort ../rsum/rsump.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssump.tex
	sort ../rsum/rsumx.tex|sed 's/_/\\_/g' |sed 's/\\\\/ /g'| \
                grep -v document | sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumx.tex
	-@chmod g+w ../rsum/ssum*.tex

routines.dvi: routines.red
	latex routines 
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 

routines.ps: routines.dvi 
	dvips routines.dvi
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 



manual.pdf: abstract.tex acknowl.tex manual.tex part0.tex part1.tex part2.tex part3.tex addsolver.tex license.tex changes.tex cover.tex part2b.tex
# First build the manual
	cp ../sty/*.sty .
	pdflatex manual | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1
	-bibtex manual
	makeindex manual
	pdflatex manual | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1
	echo  \\begin\{theindex\} > findex.tex
	echo \\indexspace >> findex.tex
	grep FUNCTION manual.ind | sed "s/FUNCTION/ /g"  | \
             sed 's/_/\\_/g' >> findex.tex
	echo \\end\{theindex\} >> findex.tex
	rm -f sindex.tex
	cat manual.ind | sed 's/_/\\_/g' > sindex.tex
	pdflatex manual | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1
	pdflatex manual | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1
	chmod -f g+w  *.aux *.toc *.log *.bbl *.blg *.ilg *.ind \
                    *.idx findex.tex sindex.tex
	mv manual.pdf ${LOC}/docs/tao_manual.pdf


clean:
	rm -f routines.ps *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err \
                  manual.pdf *.ilg *.idx *.ind sindex.tex findex.tex *.aus *.out

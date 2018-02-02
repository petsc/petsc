
#clean is required - otherwise 'temporary file manual.pdf can upset the build
ALL: clean manual.pdf

#FILTER    = | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1
FILTER    = > /dev/null 2>&1
LOCDIR    = src/docs/tao_tex/manual/
PDFLATEXOPTS = -interaction=nonstopmode -halt-on-error

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

manual.pdf: chk_loc abstract.tex acknowl.tex manual.tex part0.tex part1.tex part2.tex part3.tex addsolver.tex license.tex changes.tex cover.tex part2b.tex
	-@${MAKE} rclean
# First build the manual
	${PDFLATEX} ${PDFLATEXOPTS} manual ${FILTER}
	-bibtex manual ${FILTER}
	makeindex manual ${FILTER}
	${PDFLATEX} ${PDFLATEXOPTS} manual ${FILTER}
	echo  \\begin\{theindex\} > findex.tex
	echo \\indexspace >> findex.tex
	grep FUNCTION manual.ind | sed "s/FUNCTION/ /g"  | \
             sed 's/_/\\_/g' >> findex.tex
	echo \\end\{theindex\} >> findex.tex
	${RM} sindex.tex
	cat manual.ind | sed 's/_/\\_/g' > sindex.tex
	${PDFLATEX} ${PDFLATEXOPTS} manual ${FILTER}
	${PDFLATEX} ${PDFLATEXOPTS} manual ${FILTER}
	${MV} manual.pdf ${LOC}/docs/tao_manual.pdf

rclean:
	-@${RM} *.aux *.toc *.log *.bbl *.blg part1tmp.tex part2tmp.tex *.ilg *.idx *.ind *.out tmp anchors manual.pdf intro.pdf index.tex \
		sindex.tex findex.tex

# $Id: makefile,v 1.53 2001/04/07 02:43:36 bsmith Exp $ 

# Currently works on linux

PDFELATEX = unset TEXINPUTS; pdfelatex 
NOADIC    = true
SHELL     = sh
GONULL    = > /dev/null 2>&1

# Build the entire manual as well as the intro part

clean:
	rm -f  *.aux *.toc *.log *.bbl *.blg part1tmp.tex part2tmp.tex *.ilg *.idx *.ind *.out htmlmap tmp anchors manual.pdf intro.pdf ../manual.pdf

manual.pdf:  intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex
# Generate a list of labels
	-@rm -f anchors htmlmap tmp
	-@cat  part1.tex part2.tex | egrep '(\\label|\\chapter|\\section|\\subsection)' | ${PETSC_DIR}/maint/getpdflabels.py > anchors
	-@sed -e s%man+../%man+manualpages/% ${LOC}/docs/manualpages/manualpages.cit > htmlmap
	-@cat ${PETSC_DIR}/docs/mpi.www.index >> htmlmap
	-@cat anchors >> ${LOC}/docs/manualpages/manualpages.cit
	-@${RM} part1tmp.tex part2tmp.tex index.tex
	-@PYTHONPATH=../../../maint;cat part1.tex | ../../../maint/mapnameslatex.py > part1tmp.tex
	-@PYTHONPATH=../../../maint;cat part2.tex | ../../../maint/mapnameslatex.py > part2tmp.tex
	${PDFELATEX} manual  ${GONULL}
	bibtex manual ${GONULL}
	${PDFELATEX} manual  ${GONULL}
	${PDFELATEX} manual  ${GONULL}
	makeindex manual  ${GONULL}
	mv part2tmp.tex tmp
	grep -v "end{theindex}" tmp > part2tmp.tex
	sed -e 's/_/\\_/g' manual.ind | grep -v "begin{theindex}" >>  part2tmp.tex
	${PDFELATEX} manual  ${GONULL}
	${PDFELATEX} manual 
	mv manual.pdf ../../manual.pdf
# Now make the intro part ... note that this depends on latexing the manual
	cp manual.aux intro.aux
	${PDFELATEX} intro 
	${MAKE} clean



# $Id: makefile,v 1.53 2001/04/07 02:43:36 bsmith Exp $ 

#clean is required - otherwise 'temporary file manual.pdf can upset the build
ALL: clean manual.pdf

PDFELATEX = unset TEXINPUTS; pdfelatex 
NOADIC    = true
SHELL     = sh
MV        = mv
CP        = cp
RM        = rm -f
GONULL    = > /dev/null 2>&1
LOCDIR    = src/docs/tex/manual/

# Build the entire manual as well as the intro part

clean:
	-@${RM} *.aux *.toc *.log *.bbl *.blg part1tmp.tex part2tmp.tex *.ilg *.idx *.ind *.out tmp anchors manual.pdf intro.pdf index.tex

manual.pdf: intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex
# Generate a list of labels
	-@${MAKE} clean
	-@sed "s?hyperbaseurl?\\\hyperbaseurl{${LOC}/docs/}?g" manual.tex > manual1.tex
	-@cat  part1.tex part2.tex | egrep '(\\label|\\chapter|\\section|\\subsection)' | ${PETSC_DIR}/maint/getpdflabels.py > anchors
	-@cat anchors >> ${LOC}/docs/manualpages/manualpages.cit
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat part1.tex | ../../../../maint/mapnameslatex.py ${LOC} > part1tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat part2.tex | ../../../../maint/mapnameslatex.py ${LOC} > part2tmp.tex
	${PDFELATEX} manual1  ${GONULL}
	bibtex manual1 ${GONULL}
	${PDFELATEX} manual1  ${GONULL}
	${PDFELATEX} manual1  ${GONULL}
	makeindex manual1  ${GONULL}
	${MV} part2tmp.tex tmp
	grep -v "end{theindex}" tmp > part2tmp.tex
	sed -e 's/_/\\_/g' manual1.ind | grep -v "begin{theindex}" >>  part2tmp.tex
	${PDFELATEX} manual1  ${GONULL}
	${PDFELATEX} manual1 
	${MV} manual1.pdf ${LOC}/docs/manual.pdf
# Now make the intro part ... note that this depends on latexing the manual
	${CP} manual1.aux intro.aux
	${PDFELATEX} intro 
	${MAKE} clean



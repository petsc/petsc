
#clean is required - otherwise 'temporary file manual.pdf can upset the build
ALL: clean manual.pdf

NOADIC    = true
GONULL    = > /dev/null 2>&1
LOCDIR    = src/docs/tex/manual/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

# Build the entire manual as well as the intro part

rclean:
	-@${RM} *.aux *.toc *.log *.bbl *.blg part1tmp.tex part2tmp.tex *.ilg *.idx *.ind *.out tmp anchors manual.pdf intro.pdf index.tex

manual.pdf: chk_loc intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex
# Generate a list of labels
	-@${MAKE} rclean
	-@sed "s?hyperbaseurl?\\\hyperbaseurl{${LOC}/docs/}?g" manual.tex > manual1.tex
	-@cat  part1.tex part2.tex | egrep '(\\label|\\chapter|\\section|\\subsection)' | ${PYTHON} ${PETSC_DIR}/bin/maint/getpdflabels.py > anchors
	-@cat anchors >> ${LOC}/docs/manualpages/manualpages.cit
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat part1.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > part1tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat part2.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > part2tmp.tex
	${PDFLATEX} -halt-on-error manual1  ${GONULL}
	-bibtex manual1 ${GONULL}
	${PDFLATEX} -halt-on-error manual1  ${GONULL}
	${PDFLATEX} -halt-on-error manual1  ${GONULL}
	makeindex manual1  ${GONULL}
	${MV} part2tmp.tex tmp
	grep -v "end{theindex}" tmp > part2tmp.tex
	sed -e 's/_/\\_/g' manual1.ind | grep -v "begin{theindex}" >>  part2tmp.tex
	${PDFLATEX} -halt-on-error manual1  ${GONULL}
	${PDFLATEX} -halt-on-error manual1 
	${MV} manual1.pdf ${LOC}/docs/manual.pdf
# Now make the intro part ... note that this depends on latexing the manual
	${CP} manual1.aux intro.aux
	${PDFLATEX} -halt-on-error intro 
	${RM} part1tmp.tex part2tmp.tex
#	${MAKE} rclean

developers.pdf: chk_loc developers.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat developers.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > developerstmp.tex
	${PDFLATEX} developerstmp.tex
	${PDFLATEX} developerstmp.tex
	${MV} developerstmp.pdf ${LOC}/docs/developers.pdf


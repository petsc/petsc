
#clean is required - otherwise 'temporary file manual.pdf can upset the build
ALL: clean manual.pdf

GONULL    = > /dev/null 2>&1
LOCDIR    = src/docs/tex/manual/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

# Build the entire manual as well as the intro part

rclean:
	-@${RM} *.aux *.toc *.log *.bbl *.blg part1tmp.tex part2tmp.tex *.ilg *.idx *.ind *.out tmp anchors manual.pdf intro.pdf index.tex

manual.pdf: chk_loc intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex listing_kspex1tmp.tex listing_kspex2tmp.tex
# Generate a list of labels
	-@${MAKE} rclean
	-@sed "s?hyperbaseurl?\\\hyperbaseurl{${LOC}/docs/}?g" manual.tex > manual1.tex
	-@cat  part1.tex part2.tex | egrep '(\\label|\\chapter|\\section|\\subsection)' | ${PYTHON} ${PETSC_DIR}/bin/maint/getpdflabels.py > anchors
	-@cat anchors >> ${LOC}/docs/manualpages/manualpages.cit
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat part1.tex | \
	${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > part1tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat part2.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > part2tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat acknowl.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > acknowltmp.tex
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
	#${RM} part1tmp.tex part2tmp.tex acknowltemp.tex listing_kspex1tmp.tex listing_kspex2tmp.tex
#	${MAKE} rclean

# Wrap some listings so that they can be processed
listing_kspex2tmp.tex :
	echo '\\begin{lstlisting}' > listing_tmp.tex
	cat ../../../../src/ksp/ksp/examples/tutorials/ex2.c >> listing_tmp.tex
	echo '\\end{lstlisting}' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	$(RM) listing_tmp.tex

listing_kspex1tmp.tex :
	echo '\\begin{lstlisting}' > listing_tmp.tex
	cat ../../../../src/ksp/ksp/examples/tutorials/ex1.c >> listing_tmp.tex
	echo '\\end{lstlisting}' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	$(RM) listing_tmp.tex

developers.pdf: chk_loc developers.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat developers.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > developerstmp.tex
	${PDFLATEX} developerstmp.tex
	bibtex developerstmp ${GONULL}
	${PDFLATEX} developerstmp.tex
	${PDFLATEX} developerstmp.tex
	${MV} developerstmp.pdf ${LOC}/docs/developers.pdf


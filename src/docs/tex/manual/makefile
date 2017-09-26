
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

MANUAL_LISTINGS_INCLUDE=\
                        listing_kspex1tmp.tex \
                        listing_kspex2tmp.tex \
                        listing_snesex1tmp.tex \
                        listing_snesex3tmp.tex \
                        listing_vecex19ftmp.tex \
                        listing_vecex4ftmp.tex \
                        listing_drawex5ftmp.tex \
                        listing_snesex1ftmp.tex \
                        listing_drawex3tmp.tex \

manual.pdf: chk_loc intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex ${MANUAL_LISTINGS_INCLUDE}
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
	${RM} part1tmp.tex part2tmp.tex acknowltemp.tex tmp ${MANUAL_LISTINGS_INCLUDE}
#	${MAKE} rclean

# Wrap some listings so that they can be processed
# Some relax the break-only-at-whitespace requirement
listing_kspex2tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/ksp/ksp/examples/tutorials/ex2.c
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/ksp/ksp/examples/tutorials/ex2.c >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_kspex1tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/ksp/ksp/examples/tutorials/ex1.c
	-@printf '\\begin{lstlisting}[breakatwhitespace=false]\n' > listing_tmp.tex
	-@cat ../../../../src/ksp/ksp/examples/tutorials/ex1.c >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_snesex1tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/snes/examples/tutorials/ex1.c
	-@printf '\\begin{lstlisting}[breakatwhitespace=false]\n' > listing_tmp.tex
	-@cat ../../../../src/snes/examples/tutorials/ex1.c >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_snesex3tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/snes/examples/tutorials/ex3.c
	-@printf '\\begin{lstlisting}[breakatwhitespace=false]\n' > listing_tmp.tex
	-@cat ../../../../src/snes/examples/tutorials/ex3.c >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_vecex19ftmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/vec/vec/examples/tests/ex19f.F
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/vec/vec/examples/tests/ex19f.F >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_vecex4ftmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/vec/vec/examples/tutorials/ex4f.F
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/vec/vec/examples/tutorials/ex4f.F >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_drawex5ftmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/sys/classes/draw/examples/tests/ex5f.F
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/sys/classes/draw/examples/tests/ex5f.F >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_snesex1ftmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/snes/examples/tutorials/ex1f.F90
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/snes/examples/tutorials/ex1f.F90 >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_drawex3tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/sys/classes/draw/examples/tests/ex3.c
	-@printf '\\begin{lstlisting}[breakatwhitespace=false]\n' > listing_tmp.tex
	-@cat ../../../../src/sys/classes/draw/examples/tests/ex3.c >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

developers.pdf: chk_loc developers.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat developers.tex | \
	  ${PYTHON} ../../../../bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > developerstmp.tex
	${PDFLATEX} developerstmp.tex
	bibtex developerstmp ${GONULL}
	${PDFLATEX} developerstmp.tex
	${PDFLATEX} developerstmp.tex
	${MV} developerstmp.pdf ${LOC}/docs/developers.pdf


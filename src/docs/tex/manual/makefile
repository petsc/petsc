
#clean is required - otherwise 'temporary file manual.pdf can upset the build
ALL: clean manual.pdf

#  Run make manual.pdf LOC=$PETSC_DIR GONULL="" to display all the error messages from running pdflatex on the manual
GONULL    = > /dev/null 2>&1
LOCDIR    = src/docs/tex/manual/
PDFLATEXOPTS = -interaction=nonstopmode -halt-on-error

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

# Build the entire manual as well as the intro part

chk:
	@if [ ! -f ${LOC}/docs/manualpages/htmlmap ] ; then \
          printf ${PETSC_TEXT_HILIGHT}"*********************** ERROR ************************\n" ; \
          echo "You must first run make alldoc in ${PETSC_DIR}"; \
          printf "******************************************************"${PETSC_TEXT_NORMAL}"\n" ;false; fi
	@if [ ${PDFLATEX}foo = foo ] ; then \
          printf ${PETSC_TEXT_HILIGHT}"*********************** ERROR ************************\n" ; \
          echo "You must have sowing installed, run ./configure again with the additional option"; \
          echo " --download-sowing and make sure pdflatex is in your path"; \
          printf "******************************************************"${PETSC_TEXT_NORMAL}"\n" ;false; fi

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

manual.pdf: chk_loc chk intro.tex abstract.tex acknowl.tex manual.tex part1.tex part2.tex gettinginfo.tex ${MANUAL_LISTINGS_INCLUDE}
# Generate a list of labels
	-@${MAKE} rclean
	-@sed "s?hyperbaseurl?\\\hyperbaseurl{${LOC}/docs/}?g" manual.tex > manual1.tex
	-@cat  part1.tex part2.tex | egrep '(\\label|\\chapter|\\section|\\subsection)' | ${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/getpdflabels.py > anchors
	-@cat anchors >> ${LOC}/docs/manualpages/manualpages.cit
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat part1.tex | \
	${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > part1tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat part2.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > part2tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat acknowl.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > acknowltmp.tex
	${PDFLATEX} ${PDFLATEXOPTS} manual1  ${GONULL}
	-bibtex manual1 ${GONULL}
	${PDFLATEX} ${PDFLATEXOPTS} manual1  ${GONULL}
	${PDFLATEX} ${PDFLATEXOPTS} manual1  ${GONULL}
	makeindex manual1  ${GONULL}
	${MV} part2tmp.tex tmp
	grep -v "end{theindex}" tmp > part2tmp.tex
	sed -e 's/_/\\_/g' manual1.ind | grep -v "begin{theindex}" >>  part2tmp.tex
	${PDFLATEX} ${PDFLATEXOPTS} manual1  ${GONULL}
	${PDFLATEX} ${PDFLATEXOPTS} manual1  ${GONULL}
	${MV} manual1.pdf ${LOC}/docs/manual.pdf
# Now make the intro part ... note that this depends on latexing the manual
	${CP} manual1.aux intro.aux
	${PDFLATEX} ${PDFLATEXOPTS} intro  ${GONULL}
	${RM} part1tmp.tex part2tmp.tex acknowltemp.tex tmp ${MANUAL_LISTINGS_INCLUDE}
	${MV} intro.pdf ${LOC}/docs/
#	${MAKE} rclean

# Wrap some listings so that they can be processed
# Some relax the break-only-at-whitespace requirement
listing_kspex1tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/ksp/ksp/tutorials/ex1.c
	-@printf '\\begin{lstlisting}[breakatwhitespace=false]\n' > listing_tmp.tex
	-@cat ../../../../src/ksp/ksp/tutorials/ex1.c | sed "/TEST/,/TEST/d" >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_kspex2tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/ksp/ksp/tutorials/ex2.c
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/ksp/ksp/tutorials/ex2.c  | sed "/TEST/,/TEST/d" >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_snesex1tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/snes/tutorials/ex1.c
	-@printf '\\begin{lstlisting}[breakatwhitespace=false]\n' > listing_tmp.tex
	-@cat ../../../../src/snes/tutorials/ex1.c >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_snesex3tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/snes/tutorials/ex3.c
	-@printf '\\begin{lstlisting}[breakatwhitespace=false]\n' > listing_tmp.tex
	-@cat ../../../../src/snes/tutorials/ex3.c >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_vecex19ftmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/vec/vec/tests/ex19f.F
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/vec/vec/tests/ex19f.F >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_vecex4ftmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/vec/vec/tutorials/ex4f.F
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/vec/vec/tutorials/ex4f.F >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_drawex5ftmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/sys/classes/draw/tests/ex5f.F
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/sys/classes/draw/tests/ex5f.F >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_snesex1ftmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/snes/tutorials/ex1f.F90
	-@printf '\\begin{lstlisting}\n' > listing_tmp.tex
	-@cat ../../../../src/snes/tutorials/ex1f.F90 >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

listing_drawex3tmp.tex : ${LOC}/docs/manualpages/htmlmap ../../../../src/sys/classes/draw/tests/ex3.c
	-@printf '\\begin{lstlisting}[breakatwhitespace=false]\n' > listing_tmp.tex
	-@cat ../../../../src/sys/classes/draw/tests/ex3.c >> listing_tmp.tex
	-@printf '\\end{lstlisting}\n' >> listing_tmp.tex
	-@PYTHONPATH=../../../../maint;export PYTHONPATH;cat listing_tmp.tex | \
	  ${PYTHON} ../../../../lib/petsc/bin/maint/mapnameslatex.py ${PETSC_DIR} ${LOC}/docs/manualpages/htmlmap > $@
	-@$(RM) listing_tmp.tex

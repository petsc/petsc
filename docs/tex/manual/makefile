
NOADIC    = true

# Build the entire manual as well as the intro part
manual.dvi: intro.tex abstract.tex acknowl.tex manual_tex.tex part1.tex routines.red part2.tex
# First build the manual
	latex manual_tex | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1
	-bibtex manual_tex
	latex manual_tex | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1
	/nfs/local-sun4/tex3.14/bin/makeindex manual_tex
	echo  \\\\begin\{theindex\} > findex.tex
	echo \\indexspace >> findex.tex
	grep FUNCTION manual_tex.ind | sed "s/FUNCTION/ /g"  | \
             sed 's/_/\\_/g' >> findex.tex
	echo \\end\{theindex\} >> findex.tex
	grep -v FUNCTION manual_tex.ind | sed 's/_/\\_/g' > sindex.tex
	latex manual_tex | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg *.ilg *.ind \
                    *.idx findex.tex sindex.tex
# Now make the intro part ... note that this depends on latexing the manual
	cp manual_tex.aux intro.aux
	latex intro | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1
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
	cat ../../manualpages/manualpages.cit | sed 's/+man+../+man+manualpages/g' > www.cit 
	/home/gropp/bin/solaris/tohtml -cvtlatex -cvttables -mapman www.cit manual.tex -notopnames \
	       -nonavnames -nobottomnav -mapref ../../manualpages/manualpages.cit
	/home/gropp/bin/solaris/tohtml -cvtlatex -cvttables -mapman www.cit manual.tex -notopnames \
	       -nonavnames -nobottomnav -useimg -mapref ../../manualpages/manualpages.cit
	-@chmod g+w manual.hux latex.err manual.html
	-mv manual.html manual_html.html
	cat manual_html.html \
            | sed 's/H1/H2/g' \
            | sed 's/H0/H1/g' \
            | sed 's/<pre><tt>/<pre><tt>   /g' \
            | sed 's/&amp;gt;/   /g' \
            | sed 's/bullet://g' \
            | sed 's/$$ //g' \
            | sed 's?<H2>PETSc 2.0 Users Manual</H2>?<center><H2>PETSc 2.0 Users Manual</H2></center>?g' \
            | sed 's?Argonne National Laboratory<BR><BR></b><P>?Argonne National Laboratory<BR><BR></b><P></center>?g' \
            | sed 's/<pre><tt>   #ifndef/<pre><tt>#ifndef/g' \
            | sed 's?<b>Figure 1: </b><A NAME="Figure1">Organization of the PETSc Library</a><P>?<P> <center><IMG WIDTH=452 HEIGHT=392 ALIGN=BOTTOM SRC="petsc.gif"></center><BR><P><center><b>Figure 1: </b><A NAME="Figure1">Organization of the PETSc Library</a><P></center>?g' \
            | sed 's?<b>Figure 2: </b><A NAME="Figure2">Numerical Components of PETSc</a><P>?<P> <center><IMG WIDTH=443 HEIGHT=386 ALIGN=BOTTOM SRC="components.gif"></center><BR><P><center><b>Figure 2: </b><A NAME="Figure2">Numerical Components of PETSc</a><P></center>?g' \
            | sed 's?<b>Figure 8: </b><A NAME="Figure8">Schematic of the PETSc Directory Structure</a><P>?<P> <center><IMG WIDTH=689 HEIGHT=684 ALIGN=BOTTOM SRC="directories.gif"></center><BR><P><center><b>Figure 8: </b><A NAME="Figure8">Schematic of the PETSc Directory Structure</a><P></center>?g' \
            | sed 's?<b>Figure 10: </b><A NAME="Figure10">Ghost Points for Two Stencil Types on the Seventh Processor</a><P>?<P> <center><IMG WIDTH=490 HEIGHT=263 ALIGN=BOTTOM SRC="ghost.gif"></center><BR><P><center><b>Figure 10: </b><A NAME="Figure10">Ghost Points for Two Stencil Types on the Seventh Processor</a><P></center>?g' \
            | sed 's?<b>Figure 11: </b><A NAME="Figure10">Natural Ordering and PETSc Ordering for a 2D Distributed Array (Four Processors)</a><P>?<P> <center><IMG WIDTH=542 HEIGHT=295 ALIGN=BOTTOM SRC="danumbering.gif"></center><BR><P><center><b>Figure 10: </b><A NAME="Figure10">Natural Ordering and PETSc Ordering for a 2D Distributed Array (Four Processors)</a><P></center>?g' \
            | awk '{if ($$1 == "<b>Figure") {print "<center>" $$0 "</center>"} else {print $$0} }' \
            | awk '{if ($$1 == "<b>Table") {print "<center>" $$0 "</center>"} else {print $$0} }' \
            > manual.html
	-mv manual.html ../..
	-@chmod ug+w ../../manual.html www.cit ../../*.gif
	rm -f  latex.err findex.tex sindex.tex tmp.html manual_html.html

#
# This is the split version; some users prefer this
manual/manual.html: manual.tex abstract.tex acknowl.tex part1.tex part2.tex
	rm -f www.cit latex.err
	cat ../../manualpages/manualpages.cit | sed 's#+man+..#+man+../manualpages#g' > www.cit 
	/home/gropp/bin/solaris/tohtml -default -numbers -mapman www.cit manual.tex -notopnames \
	       -mapref ../../manualpages/manualpages.cit 
	/home/gropp/bin/solaris/tohtml -default -numbers -mapman www.cit manual.tex -notopnames \
	       -mapref ../../manualpages/manualpages.cit 
	-@chmod g+w manual.hux latex.err 
	-@chmod -R ug+w manual
	-@/bin/rm -r ../../manual
	-mv manual ../..
	-@chmod ug+w ../../manual www.cit 
	rm -f  latex.err findex.tex sindex.tex tmp.html

clean:
	rm -f routines.ps *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err \
                          *.ilg *.idx *.ind sindex.tex findex.tex *.aus

routines.red: routines.tex 
	rm -f ../rsum/ssum*.tex
	sort ../rsum/rsumVec.tex|sed 's/\\\\/ /g' | \
                 sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumVec.tex
	sort ../rsum/rsumMat.tex|sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumMat.tex
	sort ../rsum/rsumSLES.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumSLES.tex
	sort ../rsum/rsumPC.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumPC.tex
	sort ../rsum/rsumKSP.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumKSP.tex
	sort ../rsum/rsumDA.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumDA.tex
	sort ../rsum/rsumSys.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumSys.tex
	sort ../rsum/rsumViewer.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumViewer.tex
	sort ../rsum/rsumProfiling.tex|sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumProfiling.tex
	sort ../rsum/rsumSNES.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumSNES.tex
	sort ../rsum/rsumTS.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumTS.tex
	sort ../rsum/rsumMatFD.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumMatFD.tex
	sort ../rsum/rsumMatOrderings.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumMatOrderings.tex
	sort ../rsum/rsumSNESMF.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumSNESMF.tex
	sort ../rsum/rsumAO.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumAO.tex
	sort ../rsum/rsumIS.tex |sed 's/\\\\/ /g'| \
                sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumIS.tex
	sort ../rsum/rsumDraw.tex |sed 's/\\\\/ /g'| \
                grep -v document | sed 's/\\DeFn/}{\\DeFn/g' > ../rsum/ssumDraw.tex
	-@chmod g+w ../rsum/ssum*.tex

routines.dvi: routines.red
	latex routines 
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 

routines.ps: routines.dvi 
	dvips routines.dvi
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg 

latex2htmlmanual.html:
	latex2html.new -split 0 -no_navigation -no_subdir manual.tex

developers.dvi: developers.tex
# First build the manual
	latex developers
	bibtex developers
	latex developers
	latex developers
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg *.ilg *.ind \
                    *.idx

developers.ps: 
	dvips developers
	mv developers.ps ../../developers.ps
	-@chmod g+w ../../developers.ps


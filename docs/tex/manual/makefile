
NOADIC    = true

# Build the entire manual as well as the intro part
manual.dvi: intro.tex abstract.tex acknowl.tex manual_tex.tex part1.tex part2.tex
# First build the manual
	latex manual_tex 
	-bibtex manual_tex
	latex manual_tex 
	makeindex manual_tex
	echo  \\\\begin\{theindex\} > findex.tex
	echo \\indexspace >> findex.tex
	grep FUNCTION manual_tex.ind | sed "s/FUNCTION/ /g"  | \
             sed 's/_/\\_/g' >> findex.tex
	echo \\end\{theindex\} >> findex.tex
	grep -v FUNCTION manual_tex.ind | sed 's/_/\\_/g' > sindex.tex
	latex manual_tex 
	cp manual_tex.bbl manual.bbl
	-@chmod g+w *.dvi *.aux *.toc *.log *.bbl *.blg *.ilg *.ind \
                    *.idx *.ler findex.tex sindex.tex
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
	-@rm -f www.cit
	-@cat ../../manualpages/manualpages.cit | sed 's#+man+..#+man+../manualpages#g' > www.cit 
	/home/gropp/bin/solaris/tohtml -headeroffset 1 -cvtmath -quietlatex  -basedef ../doctext/url.def \
               -cvtlatex -cvttables -mapman www.cit manual.tex -notopnames \
	       -nonavnames -nobottomnav -mapref ../../manualpages/manualpages.cit
	/home/gropp/bin/solaris/tohtml -headeroffset 1 -cvtmath -quietlatex  -basedef ../doctext/url.def \
               -cvtlatex -cvttables -mapman www.cit manual.tex -notopnames \
	       -nonavnames -nobottomnav -useimg -mapref ../../manualpages/manualpages.cit
	-@chmod g+w manual.hux latex.err manual.html *.xbm *.gif
	-mv manual.html manual_html.html
	cat manual_html.html \
            | sed 's/H1/H2/g' \
            | sed 's/H0/H1/g' \
            | sed 's/&amp;gt;/   /g' \
            | sed 's/bullet://g' \
            | sed 's/$$ //g' \
            > manual.html
	-@mv manual.html *.xbm ../../manual
	-@cp *.gif ../../manual
	-@chmod -R ug+w www.cit ../../manual/*
	rm -f  latex.err findex.tex sindex.tex tmp.html manual_html.html

#
# This is the split version; some users prefer this
splitmanual.html: manual.tex abstract.tex acknowl.tex part1.tex part2.tex
	-@rm -f www.cit latex.err
	-@cat ../../manualpages/manualpages.cit | sed 's#+man+..#+man+../manualpages#g' > www.cit 
	/home/gropp/bin/solaris/tohtml -default -numbers -mapman www.cit manual.tex -notopnames \
	       -mapref ../../manualpages/manualpages.cit -quietlatex   -basedef ../doctext/url.def 
	/home/gropp/bin/solaris/tohtml -default -numbers -mapman www.cit manual.tex -notopnames \
	       -useimg -mapref ../../manualpages/manualpages.cit -quietlatex   -basedef ../doctext/url.def 
	-@chmod g+w manual.hux latex.err 
	-@/bin/rm -fr ../../splitmanual
	-@mv manual ../../splitmanual
	-@chmod -R ug+w ../../splitmanual www.cit 
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



NOADIC    = true

# Build the entire manual as well as the intro part
# manual.dvi: abstract.tex acknowl.tex manual_tex.tex part1.tex part2.tex routines.red
manual.dvi: abstract.tex acknowl.tex manual_tex.tex part0.tex part1.tex part2.tex part3.tex addsolver.tex
# First build the manual
	cp ../sty/*.sty .
	latex manual_tex | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1
	-bibtex manual_tex
	/usr/bin/makeindex manual_tex
	latex manual_tex | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1
	echo  \\begin\{theindex\} > findex.tex
	echo \\indexspace >> findex.tex
	grep FUNCTION manual_tex.ind | sed "s/FUNCTION/ /g"  | \
             sed 's/_/\\_/g' >> findex.tex
	echo \\end\{theindex\} >> findex.tex
	rm -f sindex.tex
	cat manual_tex.ind | sed 's/_/\\_/g' > sindex.tex
	latex manual_tex | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1
	latex manual_tex | grep -v Overfull 2>&1 | grep -v Underfull 2>&1 | grep -v "\[\]" 2>&1 | tr -s '\n' '\n' 2>&1

	cp manual_tex.bbl manual.bbl
	-@chmod -f g+w *.dvi *.aux *.toc *.log *.bbl *.blg *.ilg *.ind \
                    *.idx findex.tex sindex.tex

oldmanual.html: manual.tex abstract.tex acknowl.tex part0.tex part1.tex part2.tex part3.tex
	-@rm -f www.cit
	-@cat ../../manualpages/manualpages.cit | sed 's#+man+..#+man+../manualpages#g' > www.cit 
	tohtml -headeroffset 1 -cvtmath -quietlatex \
               -cvtlatex -cvttables -mapman www.cit manual.tex -notopnames \
	       -nonavnames -nobottomnav -mapref ../../manualpages/manualpages.cit
	tohtml -headeroffset 1 -cvtmath -quietlatex \
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

manual.pdf: manual_tex.tex abstract.tex acknowl.tex part0.tex part1.tex part2.tex \
            part3.tex manual_tex.dvi
	dvipdfm manual_tex
	-@cp manual_tex.pdf ../../tao_manual.pdf

manual.ps: manual_tex.tex abstract.tex acknowl.tex part0.tex part1.tex part2.tex \
           part3.tex manual_tex.dvi
	-@echo running dvips
	-@dvips manual_tex 2> tempout  
	-@cat tempout | grep -v pdf | grep -v ignored | grep -v "\[" | grep -v macro
	-@rm tempout
	-cp manual_tex.ps ../../tao_manual.ps
	-@chmod g+w ../../manual.ps


manual_tex.dvi: manual_tex.tex abstract.tex acknowl.tex part0.tex part1.tex part2.tex part3.tex addsolver.tex
	-@rm manual.dvi
	-@make manual.dvi


#
# This is the split version; some users prefer this
splitmanual.html: manual.tex abstract.tex acknowl.tex  part0.tex part1.tex part2.tex part3.tex
	-@rm -f www.cit latex.err
	-@cat ../../manualpages/manualpages.cit | sed 's#+man+..#+man+../manualpages#g' > www.cit 
	tohtml -default -numbers -mapman www.cit manual.tex -notopnames \
	       -mapref ../../manualpages/manualpages.cit -quietlatex
	tohtml -default -numbers -mapman www.cit manual.tex -notopnames \
	       -useimg -mapref ../../manualpages/manualpages.cit -quietlatex
	-@chmod g+w manual.hux latex.err 
	-@/bin/rm -fr ../../splitmanual
	-@mv manual ../../splitmanual
	-@chmod -R ug+w ../../splitmanual www.cit 
	rm -f  latex.err findex.tex sindex.tex tmp.html

clean:
	rm -f routines.ps *.dvi *.aux *.toc *.log *.bbl *.blg *.hux *.err \
                          *.ilg *.idx *.ind sindex.tex findex.tex *.aus *.out

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

latex2htmlmanual1.html:
	latex2html -split 0 -no_navigation -image_type gif -reuse 0 manual.tex

manual.html:
	cp ../sty/*.sty .
	latex manual
	bibtex manual
	latex manual
	makeindex manual.idx
	latex manual
	rm -rf manual
	latex2html -no_math -image_type gif -reuse 0 -split 0 -local_icons manual
#	/home/sarich/latex2html/bin/latex2html -no_math -image_type gif -reuse 0 -split 0 -local_icons manual
	cp ./manual/manual.html ./manual_html.html
	cat ./manual_html.html | sed "s/<BODY/<BODY BGCOLOR=\"#ffffff\" /" >manual/manual1.html 
	cat ./manual/manual1.html | sed "s/<LINK REL=\"STYLESHEET\" HREF=\"manual.css\">//" > manual/manual.html
	rm ./manual_html.html
	rm -f *.sty
	rm -f www.cit
	cat ../../manualpages/manualpages.cit | sed 's#\.\./#\.\./manualpages/#g' > www.cit 
	${TAO_DIR}/docs/tex/manual/linkmans.pl www.cit ./manual/manual.html
	${TAO_DIR}/docs/tex/manual/swapimages.pl ./manual/manual.html.linkmans ../../manual > ../../manual/manual.html
	cp -f ./manual/*.gif ../../manual/


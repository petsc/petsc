

book.dvi: book.tex ch1.tex ch2.tex ch3.tex ch4.tex ch5.tex apA.tex apB.tex
	/usr/bin/rm -f book.log *.aux book.bbl index.tex
	latex book
	bibtex book
	latex book
	makeindex book
	mv -f book.ind index.tex
	latex book
	-chmod g+w book.bbl book.dvi book.idx book.ilg book.log \
                  book.toc book.blg *.aux book.bbl index.tex 
	
book.ps: book.dvi
	dvips book

ch1.dvi: ch1.tex
	/bin/rm -f chalone.log chalone.aux chalone.bbl ch1.dvi chalone.dvi
	echo "ch1" | latex chalone 
	bibtex chalone
	echo "ch1" | latex chalone
	bibtex chalone
	echo "ch1" | latex chalone
	bibtex chalone
	-chmod g+w book.bbl book.dvi book.idx book.ilg book.log \
                  book.toc book.blg *.aux book.bbl index.tex 
	mv chalone.dvi ch1.dvi
ch1.ps: ch1.dvi
	dvips ch1
ch2.dvi: ch2.tex
	/bin/rm -f chalone.log chalone.aux chalone.bbl ch2.dvi chalone.dvi
	echo "ch2" | latex chalone 
	bibtex chalone
	echo "ch2" | latex chalone
	bibtex chalone
	echo "ch2" | latex chalone
	bibtex chalone
	-chmod g+w book.bbl book.dvi book.idx book.ilg book.log \
                  book.toc book.blg *.aux book.bbl index.tex 
	mv chalone.dvi ch2.dvi
ch2.ps: ch2.dvi
	dvips ch2

ch3.dvi: ch3.tex
	/bin/rm -f chalone.log chalone.aux chalone.bbl ch3.dvi chalone.dvi
	echo "ch3" | latex chalone 
	bibtex chalone
	echo "ch3" | latex chalone
	bibtex chalone
	echo "ch3" | latex chalone
	bibtex chalone
	-chmod g+w book.bbl book.dvi book.idx book.ilg book.log \
                  book.toc book.blg *.aux book.bbl index.tex 
	mv chalone.dvi ch3.dvi
ch3.ps: ch3.dvi
	dvips ch3

ch4.dvi: ch4.tex
	/bin/rm -f chalone.log chalone.aux chalone.bbl ch4.dvi chalone.dvi
	echo "ch4" | latex chalone 
	bibtex chalone
	echo "ch4" | latex chalone
	bibtex chalone
	echo "ch4" | latex chalone
	bibtex chalone
	-chmod g+w book.bbl book.dvi book.idx book.ilg book.log \
                  book.toc book.blg *.aux book.bbl index.tex 
	mv chalone.dvi ch4.dvi
ch4.ps: ch4.dvi
	dvips ch4

ch5.dvi: ch5.tex
	/bin/rm -f chalone.log chalone.aux chalone.bbl ch5.dvi chalone.dvi
	echo "ch5" | latex chalone 
	bibtex chalone
	echo "ch5" | latex chalone
	bibtex chalone
	echo "ch5" | latex chalone
	bibtex chalone
	-chmod g+w book.bbl book.dvi book.idx book.ilg book.log \
                  book.toc book.blg *.aux book.bbl index.tex 
	mv chalone.dvi ch5.dvi
ch5.ps: ch5.dvi
	dvips ch5
#
# Print the book.  This version puts 5 minutes between each wad of pages.
# This means that it will take about half an hour to print.
printbook: book.dvi
	/bin/rm -f booktmp.dvi
	cp book.dvi booktmp.dvi
	@-for lims in 1 31 61 91 121 151 181 211 241 ; do \
	umask 2 ; \
	lims2=`expr $$lims + 29` ;\
	dviselect $$lims:$$lims2 booktmp.dvi > book$$lims-$$lims2.dvi ;\
	dvips -q book$$lims-$$lims2 ; \
	lpr book$$lims-$$lims2.ps ; sleep 240 ; \
	done
	/bin/rm -f booktmp.dvi

chunkbook: book.dvi
	@-for lims in 1 31 61 91 121 151 181 211 241 ; do \
	umask 2 ; \
	lims2=`expr $$lims + 29` ;\
	dviselect $$lims:$$lims2 book.dvi > book$$lims-$$lims2.dvi ;\
	dvips -q book$$lims-$$lims2 ; \
	done

clean: 
	/bin/rm -f *.dvi *.log

find_locks:
	@echo finding locks in top directory
	@-rlog -L RCS/* | egrep "^RCS file|locked by"



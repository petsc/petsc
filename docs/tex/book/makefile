

book.dvi: book.tex 
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




manual.pdf: manual.xml sharedLibraryCheck.jpg sharedLibraryCheck.eps sharedLibraryCheck.pdf
	docbook2pdf manual.xml

sharedLibraryCheck.jpg: sharedLibraryCheck.fig
	fig2dev -L jpeg $< $@

sharedLibraryCheck.eps: sharedLibraryCheck.fig
	fig2dev -L eps $< $@

sharedLibraryCheck.pdf: sharedLibraryCheck.fig
	fig2dev -L pdf $< $@

view:
	xpdf manual.pdf&

clean:
	-rm manual.pdf sharedLibraryCheck.jpg

.PHONY: view clean

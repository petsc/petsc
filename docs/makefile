manual.pdf: manual.xml sharedLibraryCheck.jpg
	docbook2pdf manual.xml

sharedLibraryCheck.jpg: sharedLibraryCheck.fig
	fig2dev -L jpeg $< $@

view:
	xpdf manual.pdf&

clean:
	-rm manual.pdf sharedLibraryCheck.jpg

.PHONY: view clean

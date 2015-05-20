all: taorefs taopubs

taorefs:  taorefs.bib taorefs.cfg
	./bib2html taorefs.bib -listfile taorefs.html -listconfig taorefs.cfg -nospecials | sed s^%7E^~^g > taorefs-bib.html

taopubs:  taopubs.bib taopubs.cfg
	./bib2html taopubs.bib -listfile taopubs.html -listconfig taopubs.cfg -nospecials | sed s^%7E^~^g > taopubs-bib.html

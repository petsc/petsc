import os

if not os.access("log", os.F_OK):
	os.mkdir("log")

if not os.access("pickle", os.F_OK):
	os.mkdir("pickle")

if not os.access("Products", os.F_OK):
	os.mkdir("Products")

#!/usr/bin/env python3
from __future__ import print_function
import os
import re
from collections import defaultdict

path = "./src/ts/"


#
# Build a list of files in src/ by list comprehension. Fortran-stuff is ignored
#
print("Stage 1: Building function dictionary from source files in " + path);

sourcefiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(path)
               for name in files
               if name.endswith((".c", ".h")) and root.find("ftn-custom") < 0 and root.find("examples") < 0]

#
# Iterate over all source files and collect function names in dictionary (key-type: function name, value-type: filename where function is defined)
#

print("Stage 2: Building function dictionary...")
function_dict = defaultdict(set);

for file in sourcefiles:
  f = open(file, "r");
  for line in f.readlines():
    m = re.match('^PetscErrorCode\s*([^\s]*)\(', line)
    if m:
      function_dict[m.group(1)] = file;
    else:
      m = re.match(' void\s*([^\s]*)\(', line)
      if m:
        function_dict[m.group(1)] = file;

#
# Consistency check: There might be function names used multiple times.
#

# TODO





#
# Iterate over all source files and scan for the use of any of the registered functions
#

print("Stage 3: Building function calling dictionary (this might take a while)...")
function_calling_dict = defaultdict(set);  # Dictionary which records all calls to a function

for file in sourcefiles:
  f = open(file, "r");
  for line in f.readlines():
    if line.find(".seealso") >= 0:
      continue;

    for funcname in function_dict.keys():
      if line.find(' ' + funcname + '(') >= 0 or line.find("=" + funcname + '(') >= 0:    #Note: Might not have perfect accuracy, but is much faster then regular expressions.
        function_calling_dict[funcname].add(file);
#        print line;


#
# Now extract all functions which are only used in one file
#

static_functions_for_file = defaultdict(list);

for func in function_calling_dict.keys():
  if len(function_calling_dict[func]) < 2:
    static_functions_for_file[function_calling_dict[func].pop()].append(func);


#
# Output 'static' functions per file:
#

print("#")
print("# Functions only used in one file:")
print("#")

for filename in static_functions_for_file.keys():
  print(filename + ": ");
  for func in static_functions_for_file[filename]:
    print("  " + func);

#  print func + ": " + str(len(function_calling_dict[func])) + " uses";




#print "Function dictionary:"
#print function_dict;

#print "Function-calling dictionary:"
#print function_calling_dict;


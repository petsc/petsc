#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $ 
#
# change python to whatever is needed on your system to invoke python
#
#  Reads classes.data and prints the language d classes
#
#  Crude as all hack!
#
#  Calling sequence: 
#      d.py
##
import os
import re
from exceptions import *
import sys
from string import *
import pickle

def replace(enums,senums,structs,aliases,classes,k):
  k = k.replace("*","")
  for l in aliases: k = k.replace(l,aliases[l])
  k = k.replace("char[]","String").replace("char*","String").replace("char","String");
  for l in senums: k = k.replace(l,"String")
  k = k.replace("void","int")
  k = k.replace("PetscTruth","boolean")
  k = k.replace("*","")
  k = k.replace("unsigned","")
  k = k.replace("const","")
  k = k.replace("ushort","short")
  k = k.replace("const ","").replace("unsigned long","ulong")
  return k

def main(args):
  file = open('classes.data')
  enums   = pickle.load(file)
  senums  = pickle.load(file)
  structs = pickle.load(file)    
  aliases = pickle.load(file)  
  classes = pickle.load(file)
  if not os.path.isdir('src/java'): os.mkdir('src/java')
  outfile = open('src/java/petsc.java','w')

  notclasses = ['PetscMalloc','PetscStr','PetscDLLibrary','PetscFList']
  for i in enums:
    outfile.write("enum "+i+"\n")
    outfile.write("{\n")
    cnt = 0
    for j in enums[i]:
      outfile.write("    "+j.replace("=","(")+")")
      cnt = cnt + 1
      if not cnt == len(enums[i]): outfile.write(",")
      else: outfile.write(";")      
      outfile.write("\n")
    outfile.write("    private final int value;\n")
    outfile.write("    "+i+"(int v) { this.value = v; }\n")
    outfile.write("}\n")      
  outfile.write("\n")

  for i in structs:
    outfile.write("class "+i+"\n")
    outfile.write("{\n")
    for k in structs[i]:
      k = replace(enums,senums,structs,aliases,classes,k)
      outfile.write("    "+k+";\n")
    outfile.write("};\n")      
  outfile.write("\n")

  for i in classes:
    if i in notclasses: continue
    outfile.write("class "+i+"\n")
    outfile.write("{\n")
    for j in classes[i]:
      if not classes[i][j] or not classes[i][j][0] == i:
        outfile.write("  static ")      
      outfile.write("native void _"+j+"();\n")
    for j in classes[i]:
      if not classes[i][j] or not classes[i][j][0] == i:
        outfile.write("  static ")      
      outfile.write("    void "+j+"(")
      cnt = 0
      for k in classes[i][j]:
        k = replace(enums,senums,structs,aliases,classes,k)
        if cnt > 0:
          outfile.write(k)
          outfile.write(" a"+str(cnt))
          if cnt < len(classes[i][j])-1: outfile.write(",")
        cnt = cnt + 1
      outfile.write("){_"+j+"(")
      outfile.write(");}\n")      
    outfile.write("    int ptr;\n")
    if i == "Petsc":
      outfile.write('static {System.loadLibrary("petsc");}\n')
    outfile.write("}\n")
  outfile.close()
  
  os.system("cd src/java; javac petsc.java")
  import re
  for i in classes:
    if i in notclasses: continue
    os.system("cd src/java; javah "+i+";cp "+i+".h "+i+".c")
    f = open('src/java/'+i+".c","r")
    w = f.read()
    f.close()
#    w = w.replace("  (JNIEnv *, jobject);","  (JNIEnv *a1, jobject a2) {;}; ")
#    w = w.replace("  (JNIEnv *, jclass);","  (JNIEnv *a1, jclass a2) {;}; ")    
    w = w.replace("#include <jni.h>",'#include "'+i+'.h"')
    w = w.replace("#ifndef _Included_","#ifndef _string_that_does_not_exist_")
    w = w.replace("#define _Included_","#define _another_string_that_does_not_exist_")
    w = w.replace("#include <jni.h>","#include <JavaVM/jni.h>")
    for j in classes[i]:    
#      w = re.sub('JNIEXPORT void JNICALL Java_'+i+'([_0-9]*)'+j+'\n','JNIEXPORT void JNICALL Java_'+i+'\\1'+j+' ',w)
      w = re.sub('JNIEXPORT void JNICALL Java_'+i+'([_0-9]*)'+j+'\n [ ]* \(JNIEnv \*, ([a-z]*)\)','JNIEXPORT void JNICALL Java_'+i+'\\1'+j+' (JNIEnv *a1,\\2 a2) {'+i+j+'();}',w)
    f = open('src/java/'+i+".c","w")    
    f.write(w)
    f.close()
    
    f = open('src/java/'+i+".h","r")
    w = f.read()
    f.close()
    w = w.replace("#include <jni.h>","#include <JavaVM/jni.h>")
    f = open('src/java/'+i+".h","w")    
    f.write(w)
    f.close()
    os.system("cd src/java; gcc -c -fPIC "+i+".c")

  os.system("cd src/java; gcc -dynamiclib -single_module -multiply_defined suppress -undefined dynamic_lookup -framework JavaVM -o libpetsc.dylib *.o ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetscts.dylib")    

    
  
    
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  main(sys.argv[1:])


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
import sys
from string import *
import pickle

def replace(enums,senums,structs,aliases,classes,k):
  k = k.replace("*","")
  for l in aliases: k = k.replace(l,aliases[l])
  k = k.replace("char[]","String").replace("char*","String").replace("char","String");
  for l in senums: k = k.replace(l,"String")
  k = k.replace("void","int")
  k = k.replace("PetscBool","boolean")
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

  notclasses = ['PetscMalloc','PetscStr','PetscDLLibrary','PetscFunctionList']
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
      else:
        outfile.write("native void _"+j+"(int ptr")
        cnt = 0
        for k in classes[i][j]:
          if cnt > 0:
            k = replace(enums,senums,structs,aliases,classes,k)
            if k in classes: k = 'int'
            outfile.write(","+k+" a"+str(cnt))
          cnt = cnt + 1
        outfile.write(");\n")

    outfile.write("  static native int _"+i+"Create();\n")
    outfile.write("    "+i+"() {this.ptr = _"+i+"Create();}\n")

    for j in classes[i]:
      if not classes[i][j] or not classes[i][j][0] == i:
        outfile.write("  static ")
      outfile.write("    void "+j+"(")
      cnt = 0
      for k in classes[i][j]:
        k = replace(enums,senums,structs,aliases,classes,k)
        if cnt > 0:
          outfile.write(k+" a"+str(cnt))
          if cnt < len(classes[i][j])-1: outfile.write(",")
        cnt = cnt + 1
      outfile.write("){_"+j+"(")
      if classes[i][j] and classes[i][j][0] == i:
        outfile.write("this.ptr")
        for l in range(1,cnt):
          if replace(enums,senums,structs,aliases,classes,classes[i][j][l]) in classes: outfile.write(",a"+str(l)+".ptr")
          else: outfile.write(",a"+str(l))
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
    w = w.replace("#include <jni.h>",'#include "'+i+'.h"')
    w = w.replace("#ifndef _Included_","#ifndef _string_that_does_not_exist_")
    w = w.replace("#define _Included_","#define _another_string_that_does_not_exist_")
    w = w.replace("#include <jni.h>","#include <JavaVM/jni.h>")
    for j in classes[i]:
      t = re.search('JNIEXPORT void JNICALL Java_'+i+'[_0-9]*'+j+'\n [ ]* \(JNIEnv \*([, A-Za-z]*)\)',w)
      if t:
        t = t.group(0).split('\n')[1].replace('(JNIEnv *,','').replace(')','').split(',')
        cnt = 1
        t1  = ''
        t2  = 'a2'
        for k in t:
          t1 = t1 + ','+k+' a'+str(cnt)
          if cnt > 2:
            if k.strip() == 'jstring':
              t2 = t2 +', (*a0)->GetStringUTFChars(a0, a'+str(cnt)+',0)'
            else:
              t2 = t2 +', a'+str(cnt)
          cnt = cnt + 1
        if cnt == 2:
          w = re.sub('JNIEXPORT void JNICALL Java_'+i+'([_0-9]*)'+j+'\n [ ]* \(JNIEnv \*, ([a-z]*)\)','JNIEXPORT void JNICALL Java_'+i+'\\1'+j+' (JNIEnv *a0,\\2 a1) {'+i+j+'();}',w)
        else:
          w = re.sub('JNIEXPORT void JNICALL Java_'+i+'([_0-9]*)'+j+'\n [ ]* \(JNIEnv \*([, A-Za-z]*)\)','JNIEXPORT void JNICALL Java_'+i+'\\1'+j+' (JNIEnv *a0'+t1+') {'+i+j+'('+t2+');}',w)

    w = re.sub('JNIEXPORT jint JNICALL Java_'+i+'([_0-9]*)'+i+'Create\n [ ]* \(JNIEnv \*, ([a-z]*)\)','JNIEXPORT jint JNICALL Java_'+i+'\\1'+i+'Create(JNIEnv *a1,\\2 a2) {int ptr; '+i+'Create(0,&ptr); return ptr;}',w)
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


/* $Id: petscclfe.cpp,v 1.1 2001/03/06 23:58:18 buschelm Exp $ */
#include <iostream>
#include <stdlib.h>
#include "petscclfe.h"

using namespace PETScFE;

cl::cl() {
  OutputFlag = 0;
}

void cl::GetArgs(int argc,char *argv[]) {
  compiler::GetArgs(argc,argv);
  if (!verbose) compilearg[0] = compilearg[0] + " -nologo";
  linkarg[0]="-link";
}

void cl::Parse(void) {
  compiler::Parse();
  /* Fix Output flag for compile or link (-Fe or -Fo) */ 
  FixFx();
}

void cl::Compile(void) {
  compiler::Squeeze();

  string compile = compilearg[0];
  Merge(compile,compilearg,1);
  /* Execute each compilation one at a time */ 
  for (int i=0;i<file.size();i++) {
    /* Make default output a .o not a .obj */
    string outfile;
    if (OutputFlag==0) {
      outfile = "-Fo" + file[i];
      int n = outfile.find_last_of(".");
      string filebase = outfile.substr(0,n);
      outfile = filebase + ".o";
    }
    if (file[i]!="") {
      string compileeach = compile + " " + file[i] + " " + outfile;
      if (verbose) cout << compileeach << endl;
      system(compileeach.c_str());
    }
  }
}

void cl::FoundD(int &loc,string temp) {
  string::size_type i,j;
  i = temp.find("\"");
  if (i!=string::npos) {
    temp = temp.substr(0,i+1)+"\\\""+temp.substr(i+1,string::npos);
    j = temp.rfind("\"");
    if (j!=i+2) {
      temp = temp.substr(0,j)+"\\\""+temp.substr(j,string::npos);
    }
  }
  compilearg[loc]=temp;  
}

void cl::FoundL(int &loc,string temp) {
  linkarg[loc] = "-libpath:" + temp.substr(2);
  ReplaceSlashWithBackslash(linkarg[loc]);
  compilearg[loc] = "";
}

void cl::Foundl(int &loc,string temp) { 
  file[loc] = "lib" + temp.substr(2) + ".lib";
  compilearg[loc] = "";
} 

void cl::Foundo(int &loc,string temp){ 
  /* Set Flag then fix later based on compilation or link */
  OutputFlag = loc;
  compilearg[loc] = "-Fx" + arg[loc+1];
  arg[++loc] = "";
  /* Should perform some error checking ... */
}   

void cl::FixFx(void) {
  if (OutputFlag!=0) {
    if (linkarg[0]=="-c") {
      compilearg[OutputFlag][2]='o';
    } else {
      compilearg[OutputFlag][2]='e';
    }
  }
}

void df::Foundo(int &loc,string temp) {
  if (temp == "-o") {
    cl::Foundo(loc,temp);
  } else {
    compilearg[loc] = temp;
  }
}

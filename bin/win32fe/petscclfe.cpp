/* $Id:$ */
#include <iostream>
#include <stdlib.h>
#include "petscclfe.h"

using namespace PETScFE;

#define UNKNOWN '*'

cl::cl() {
  OutputFlag = 0;
  OptionTags = "DILclo";
  Options['D'] = &PETScFE::cl::FoundD;
  Options['L'] = &PETScFE::cl::FoundL;
  Options['l'] = &PETScFE::cl::Foundl;
  Options['o'] = &PETScFE::cl::Foundo;
}

void cl::GetArgs(int argc,char *argv[]) {
  compiler::GetArgs(argc,argv);
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
      string compileeach = compile + " -c " + file[i] + " " + outfile;
      if (!quiet) cout << compileeach << endl;
      system(compileeach.c_str());
    }
  }
}

void cl::FoundD(int &loc,string temp) {
  string::size_type i,j;
  i = temp.find("\"");
  if (i!=string::npos) {
    if (i!=1) {
      temp = temp.substr(0,i)+"\\"+temp.substr(i,string::npos);
    } else {
      temp = "\\" + temp;
    }
    j = temp.rfind("\"");
    if (j!=i+1) {
      temp = temp.substr(0,j)+"\\"+temp.substr(j,string::npos);
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

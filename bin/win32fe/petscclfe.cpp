/* $Id: clfe.cpp,v 1.11 2001/04/17 20:51:59 buschelm Exp buschelm $ */
#include <stdlib.h>
#include "petscclfe.h"
#include "Windows.h"

using namespace PETScFE;

cl::cl() {
  OutputFlag = compilearg.end();
}

void cl::GetArgs(int argc,char *argv[]) {
  compiler::GetArgs(argc,argv);
  if (!verbose) {
    string temp = compilearg.front();
    compilearg.pop_front();
    compilearg.push_front("-nologo");
    compilearg.push_front(temp);
  }
  linkarg.push_front("-link");
}

void cl::Parse(void) {
  compiler::Parse();
  /* Fix Output flag for compile or link (-Fe or -Fo) */ 
  FixFx();
}

void cl::Help(void) {
  tool::Help();
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

void cl::Compile(void) {
  LI i = compilearg.begin();
  string compile = *i++;
  Merge(compile,compilearg,i);
  /* Execute each compilation one at a time */ 
  i = file.begin();
  while (i != file.end()) {
    /* Make default output a .o not a .obj */
    string outfile;
    if (OutputFlag==compilearg.end()) {
      outfile = "-Fo" + *i;
      int n = outfile.find_last_of(".");
      string filebase = outfile.substr(0,n);
      outfile = filebase + ".o";
    }
    string compileeach = compile + " " + *i++ + " " + outfile;
    if (verbose) cout << compileeach << endl;
    system(compileeach.c_str());
  }
}

void cl::FoundD(LI &i) {
  string::size_type a,b;
  string temp = *i;
  ProtectQuotes(temp);
  compilearg.push_back(temp);  
}

void cl::FoundL(LI &i) {
  string temp = i->substr(2);
  ReplaceSlashWithBackslash(temp);
  GetShortPath(temp);
  temp = "-libpath:"+temp;
  linkarg.push_back(temp);
}

void cl::Foundl(LI &i) {
  string temp = *i;
  file.push_back("lib" + temp.substr(2) + ".lib");
} 

void cl::Foundo(LI &i){ 
  i++;
  arg.pop_front();
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  ProtectQuotes(temp);
  /* Set Flag then fix later based on compilation or link */
  compilearg.push_back("-Fx" + temp);
  OutputFlag = --compilearg.end();
  /* Should perform some error checking ... */
}   

void cl::FixFx(void) {
  if (OutputFlag!=compilearg.end()) {
    string temp = *OutputFlag;
    compilearg.erase(OutputFlag);
    if (linkarg.front()=="-c") {
      temp[2]='o';
    } else {
      temp[2]='e';
    }
    compilearg.push_back(temp);
  }
}

void df::Foundo(LI &i) {
  if (*i == "-o") {
    cl::Foundo(i);
  } else {
    compilearg.push_back(*i);
  }
}

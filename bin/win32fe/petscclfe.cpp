/* $Id: petscclfe.cpp,v 1.12 2001/04/17 21:17:31 buschelm Exp $ */
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
    compilearg.push_front("-nologo");
  }
  linkarg.push_front("-link");
}

void cl::Parse(void) {
  compiler::Parse();
  /* Fix Output flag for compile or link (-Fe or -Fo) */ 
  FixFx();
}

void cl::Help(void) {
  compiler::Help();
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

void cl::Compile(void) {
  compileoutflag = "-Fo";
  compiler::Compile();
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

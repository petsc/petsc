/* $Id: petscclfe.cpp,v 1.12 2001/04/17 21:17:31 buschelm Exp $ */
#include <stdlib.h>
#include "petscclfe.h"
#include "Windows.h"

using namespace PETScFE;

cl::cl() {
  compileoutflag = "-Fo";
  linkoutflag    = "-Fe";

  OutputFlag = compilearg.end();
}

void cl::Parse(void) {
  linkarg.push_front("-link");
  compiler::Parse();
  if (!verbose) {
    compilearg.push_back("-nologo");
  }
}

void cl::Help(void) {
  compiler::Help();
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

void cl::FoundL(LI &i) {
  string temp = i->substr(2);
  ReplaceSlashWithBackslash(temp);
  GetShortPath(temp);
  temp = "-libpath:"+temp;
  linkarg.push_back(temp);
}

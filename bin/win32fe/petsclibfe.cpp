/* $Id: petsclibfe.cpp,v 1.3 2001/03/23 19:31:16 buschelm Exp $ */
#include <fstream>
#include "petsclibfe.h"

using namespace PETScFE;
  
void lib::Execute(void) {
  if (!helpfound) {
    string temp = *file.begin();
    { /* Open file stream */ 
      ifstream LibraryExists(temp.c_str());
      if (temp.substr(temp.rfind("."))==".lib") {
        if (!LibraryExists) temp = "-out:" + *file.begin();
      } else {
        temp = "-out:" + *file.begin();
        if (LibraryExists) temp = temp + " " + *file.begin();
      }
    } /* Close file stream */ 
    file.pop_front();
    file.push_front(temp);
    if (!verbose) {
      string libexe = *archivearg.begin();
      libexe += " -nologo";
      archivearg.pop_front();
      archivearg.push_front(libexe);
    }
  }
  archiver::Execute();
}

void lib::Help(void) {
  tool::Help();
  string help = *archivearg.begin();
  help = help.substr(0,help.find(" -nologo"));
  help += " -? 2>&1"; 
  system(help.c_str());
}

/* $Id: petsclibfe.cpp,v 1.1 2001/03/06 23:58:18 buschelm Exp $ */
#include <fstream>
#include "petsclibfe.h"

using namespace PETScFE;
  
void lib::Execute(void) {
  {
    string temp = *file.begin();
    ifstream LibraryExists(temp.c_str());
    if (temp.substr(temp.rfind("."))==".lib") {
      if (!LibraryExists) temp = "-out:" + *file.begin();
    } else {
      temp = "-out:" + *file.begin();
      if (LibraryExists) temp = temp + " " + *file.begin();
    }
    file.pop_front();
    file.push_front(temp);
  }
  if (!verbose) {
    string thelib = *archivearg.begin();
    archivearg.pop_front();
    archivearg.push_front(thelib + " -nologo");
  }
  archiver::Execute();
}

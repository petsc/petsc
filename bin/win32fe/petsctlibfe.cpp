/* $Id: petsctlibfe.cpp,v 1.1 2001/03/06 23:58:18 buschelm Exp $ */
#include <fstream>
#include "Windows.h"
#include "petsctlibfe.h"

using namespace PETScFE;

void tlib::Execute() {
  archiver::Execute();
  string temp;
  temp = *file.begin();
  temp = temp.substr(0,temp.rfind("."));
  temp = temp.substr(1) + ".BAK";
  bool deleteme = FALSE;
  {
    ifstream LibraryExists(temp.c_str());
    if (LibraryExists) deleteme=TRUE;
  }
  if (deleteme) {
    if (verbose) cout << "del \"" << temp << "\"" << endl;
    DeleteFile(temp.c_str());
  }
}

void tlib::FoundFile(LI &i) {
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  if (temp[0]!='\"') {
    file.push_back("\"" + temp + "\"");
  } else {
    file.push_back(temp);
  }
}

void tlib::FoundFlag(LI &i) {
  string temp = *i;
  temp[0] = '/';
  archivearg.push_back(temp);
}

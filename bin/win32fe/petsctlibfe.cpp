/* $Id: petsctlibfe.cpp,v 1.2 2001/03/23 19:31:16 buschelm Exp buschelm $ */
#include <fstream>
#include <iostream>
#include "Windows.h"
#include "petsctlibfe.h"

using namespace PETScFE;
using namespace std;

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

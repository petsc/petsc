/* $Id: petsctlibfe.cpp,v 1.8 2001/04/17 21:15:54 buschelm Exp $ */
#include <fstream>
#include "Windows.h"
#include "petsctlibfe.h"

using namespace PETScFE;
using namespace std;

void tlib::Execute() {
  archiver::Execute();
  if (!helpfound) {
    string temp = file.front();
    temp = temp.substr(0,temp.rfind("."));
    temp = temp.substr(1) + ".BAK";
    bool deleteme = FALSE;
    { /* Open file stream */ 
      ifstream LibraryExists(temp.c_str());
      if (LibraryExists) deleteme=TRUE;
    } /* Close file stream */
    if (deleteme) {
      if (verbose) cout << "del \"" << temp << "\"" << endl;
      DeleteFile(temp.c_str());
    }
  }
}

void tlib::Help(void) {
  archiver::Help();
  cout << "tlib specific help:" << endl;
  cout << "  Note: tlib operators +, -+, *, etc. are not supported by win32fe." << endl;
  cout << "        Instead, use the flags -a, -u, -e, etc. accordingly." << endl << endl;
  cout << "=========================================================================" << endl << endl;
  string help = archivearg.front();
  system(help.c_str());
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
  if (temp == "-help") {
    helpfound = -1;
  } else {
    temp[0] = '/';
    archivearg.push_back(temp);
  }
}

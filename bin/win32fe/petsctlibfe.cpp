/* $Id: petsctlibfe.cpp,v 1.8 2001/04/17 21:15:54 buschelm Exp $ */
#include "Windows.h"
#include "petsctlibfe.h"

using namespace PETScFE;
using namespace std;

void tlib::Execute() {
  archiver::Execute();
  Archive();
  if (!helpfound) {
    string backup = file.front();
    backup = backup.substr(0,backup.rfind("."));
    backup = backup + ".BAK";
    string temp=backup;
    if (GetShortPath(temp)) {
      if (verbose) cout << "del \"" << backup << "\"" << endl;
      DeleteFile(backup.c_str());
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

void tlib::FoundFlag(LI &i) {
  string temp = *i;
  if (temp == "-help") {
    helpfound = -1;
  } else {
    temp[0] = '/';
    archivearg.push_back(temp);
  }
}

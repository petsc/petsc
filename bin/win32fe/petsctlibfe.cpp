/* $Id: petsctlibfe.cpp,v 1.11 2001/05/05 02:16:22 buschelm Exp buschelm $ */
#include "Windows.h"
#include "petsctlibfe.h"

using namespace PETScFE;
using namespace std;

void tlib::Execute() {
  archiver::Execute();
  if ((!helpfound) || (!versionfound)) {
    Archive();
    string backup = file.front();
    backup = backup.substr(1,backup.rfind(".")-1);
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
  cout << "  win32fe requires the use of - to denote flags instead of /." << endl;
  cout << "        The translation from - to / for tlib is managed automatically." << endl;
  cout << "  Note: tlib operators +, -+, *, etc. are not supported by win32fe." << endl;
  cout << "        Instead, use the arguments -a, -u, -e, etc. accordingly." << endl << endl;
  cout << "=========================================================================" << endl << endl;
  string help = archivearg.front();
  system(help.c_str());
}

void tlib::FoundFlag(LI &i) {
  string temp = *i;
  if (temp == "-help") {
    helpfound = TRUE;
  } else {
    temp[0] = '/';
    archivearg.push_back(temp);
  }
}

void tlib::FoundFile(LI &i) {
  tool::FoundFile(i);
  string temp=file.back();
  file.pop_back();
  file.push_back("\"" + temp + "\"");
}

/* $Id: petsclibfe.cpp,v 1.7 2001/04/17 21:16:12 buschelm Exp $ */
#include "petsclibfe.h"

using namespace PETScFE;
  
void lib::Execute(void) {
  archiver::Execute();
  if (!helpfound) {
    string temp, archivename = file.front();
    file.pop_front();
    archivearg.push_back("-out:" + archivename);
    temp = archivename;
    if (GetShortPath(temp)) {
      file.push_front(archivename);
    }
  }
  Archive();
}

void lib::Archive(void) {
  LI li = archivearg.begin();
  string header = *li++;
  Merge(header,archivearg,li);
//    PrintListString(archivearg);
  li = file.begin();
  string archivename = file.front();
  while (li != file.end()) {
    string archive = header;
    Merge(archive,file,li);
    if (verbose) cout << archive << endl;
    system(archive.c_str());
    if (archivearg.back()!=archivename) {
      archivearg.push_back(archivename);
      Merge(header,archivearg,--archivearg.end());
    }
  }
}

void lib::Help(void) {
  archiver::Help();
  cout << "lib specific help:" << endl;
  cout << "  Note: win32fe will check to see if the library exists, and will modify" << endl;
  cout << "        the arguments to lib accordingly.  As a result, specifying" << endl;
  cout << "        -out:<libname> is unnecessary and may lead to unexpected results." << endl << endl;
  cout << "=========================================================================" << endl << endl;
  
  string help = archivearg.front();
  help += " -? 2>&1"; 
  system(help.c_str());
}

void lib::Parse(void) {
  archiver::Parse();
  if (!verbose) {
    archivearg.push_back("-nologo");
  }
}

/* $Id: libfe.cpp,v 1.6 2001/04/17 20:51:59 buschelm Exp buschelm $ */
#include <fstream>
#include "petsclibfe.h"

using namespace PETScFE;
  
void lib::Execute(void) {
  tool::Execute();
  if (!helpfound) {
    if (!verbose) {
      string libexe = archivearg.front();
      archivearg.pop_front();
      archivearg.push_front("-nologo");
      archivearg.push_front(libexe);
    }
    string archivename = file.front();
    file.pop_front();
    archivearg.push_back("-out:" + archivename);
    { /* Open file stream */ 
      ifstream ArchiveExists(archivename.c_str());
      if (ArchiveExists) archivearg.push_back(archivename);
    } /* Close file stream */
    LI li = archivearg.begin();
    string header = *li++;
    Merge(header,archivearg,li);
    li = file.begin();
    string archive;
    while (li != file.end()) {
      archive = header;
      Merge(archive,file,li);
      if (verbose) cout << archive << endl;
      system(archive.c_str());
      if (archivearg.back()!=archivename)
        Merge(header,archivearg,archivearg.insert(archivearg.end(),archivename));
    }
  }
}

void lib::Help(void) {
  tool::Help();
  cout << "  Note: win32fe will check to see if the library exists, and will modify" << endl;
  cout << "        the arguments to lib accordingly.  As a result, specifying" << endl;
  cout << "        -out:<libname> is unnecessary and may lead to unexpected results." << endl << endl;
  cout << "  =======================================================================" << endl << endl;
  
  string help = archivearg.front();
  help += " -? 2>&1"; 
  system(help.c_str());
}

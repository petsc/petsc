/* $Id: petscarchivefe.cpp,v 1.4 2001/03/23 19:31:16 buschelm Exp $ */
#include <stdlib.h>
#include "petscarchiverfe.h"

using namespace PETScFE;

void archiver::GetArgs(int argc,char *argv[]) {
  tool::GetArgs(argc,argv);
  LI i = arg.begin();
  archivearg.push_front(*i);
  arg.pop_front();
}

void archiver::Parse(void) {
  tool::Parse();
  LI i = arg.begin();
  while (i !=arg.end()) {
    string temp = *i;
    if (temp[0]=='-') {
      FoundFlag(i);
    } else {
      FoundFile(i);
    }
    i++;
    arg.pop_front();
  }
}

void archiver::Execute(void) {
  tool::Execute();
  if (!helpfound) {
    LI i = archivearg.begin();
    string archive = *i++;
    Merge(archive,archivearg,i);
    Merge(archive,file,file.begin());
    if (verbose) cout << archive << endl;
    system(archive.c_str());
  }
}

void archiver::Help(void) {
  tool::Help();
  string help = *archivearg.begin();
  help += " -help";
  system(help.c_str());
}

void archiver::FoundFile(LI &i) {
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  file.push_back(temp);
}

void archiver::FoundFlag(LI &i) {
  string temp = *i;
  if (temp == "-help") {
    helpfound = -1;
  } else {
    archivearg.push_back(*i);
  }
}

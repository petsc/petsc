/* $Id: petscarchivefe.cpp,v 1.1 2001/03/06 23:58:18 buschelm Exp $ */
#include <iostream>
#include <stdlib.h>
#include "petscfe.h"

using namespace PETScFE;

void archiver::GetArgs(int argc,char *argv[]) {
  tool::GetArgs(argc,argv);
  LI i = arg.begin();
  archivearg.push_front(*i);
  arg.pop_front();
}

void archiver::Parse(void) {
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
  LI i = archivearg.begin();
  string archive = *i++;
  Merge(archive,archivearg,i);
  Merge(archive,file,file.begin());
  if (verbose) cout << archive << endl;
  system(archive.c_str());
}

void archiver::FoundFile(LI &i) {
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  file.push_back(temp);
}

void archiver::FoundFlag(LI &i) {
  archivearg.push_back(*i);
}

/* $Id:$ */
#include <iostream>
#include <stdlib.h>
#include "petscfe.h"

using namespace PETScFE;

void archiver::GetArgs(int argc,char *argv[]) {
  tool::GetArgs(argc,argv);
  archivearg.resize(argc-1);
  archivearg[0]=arg[0];
  file.resize(argc-1);
}

void archiver::Parse(void) {
  for (int i=1;i<arg.size();i++) {
    string temp = arg[i];
    if (temp[0]=='-') {
      FoundFlag(i,temp);
    } else {
      FoundFile(i,temp);
    }
  }
  Squeeze();
}

void archiver::Execute(void) {
  tool::Execute();
  string archive=archivearg[0];
  int i;
  for (i=1;i<archivearg.size();i++) {
    if (archivearg[i]=="") break;
    archive += " " + archivearg[i];
  }
  for (i=0;i<file.size();i++) {
    if (file[i]=="") break;
    archive += " " + file[i];
  }
  if (!quiet) cout << archive << endl;
  system(archive.c_str());
}

void archiver::FoundFile(int &loc,string temp) {
  ReplaceSlashWithBackslash(temp);
  file[loc] = temp;
}

void archiver::FoundFlag(int &loc,string temp) {
  archivearg[loc] = temp;
}

void archiver::Squeeze(void) {
  tool::Squeeze(archivearg);
  tool::Squeeze(file);
}

/* $Id: petscarchiverfe.cpp,v 1.10 2001/04/18 01:00:36 buschelm Exp $ */
#include <stdlib.h>
#include <process.h>
#include "petscarchiverfe.h"
#include <string.h>

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
    string archive;
    archivearg.push_back(file.front());
    file.pop_front();
    LI li = archivearg.begin();
    string header = *li++;
    Merge(header,archivearg,li);
    li = file.begin();
    while (li != file.end()) {
      /* Invoke archiver several times to limit arg length <512 chars */
      archive = header;
      Merge(archive,file,li);
      if (verbose)
        cout << archive << endl;
      system(archive.c_str());
    }
  }
}

void archiver::Help(void) {
  tool::Help();
  string help = archivearg.front();
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

void archiver::Merge(string &str,list<string> &liststr,LI &i) {
  string::size_type len=str.length();
  string::size_type maxlen=512;
  while (i!=liststr.end()) {
    string trystr=*i;
    if ((len+trystr.length()+1) < maxlen) {
      str += " " + trystr;
      len = str.length();
      i++;
    } else {
      break;
    }
  }
}

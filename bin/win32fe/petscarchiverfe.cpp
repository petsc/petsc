/* $Id: archiverfe.cpp,v 1.1 2001/04/17 15:21:14 buschelm Exp buschelm $ */
#include <stdlib.h>
#include <process.h>
#include "archiverfe.h"
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
    int lenarchive,lenfiles;
    LI li = archivearg.begin();
    string archive = *li++;
    string files,callarchive;
    Merge(archive,archivearg,li);
    li = file.begin();
    while (li != file.end()) {
      /* Invoke archiver several times to limit arg length <1024 chars */
      Merge(files,file,li);
      callarchive = archive + " " +files;
      files = "";
      if (verbose) {
        cout << callarchive << endl;
      }
      system(callarchive.c_str());
    }
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

void archiver::Merge(string &str,list<string> &liststr,LI &i) {
  int len = str.length();
  string tryfile = *i;
  while (((len+tryfile.length()+1)<512) && (i!=liststr.end())) {
    i++;
    str += " " + tryfile;
    len = str.length();
    tryfile = *i;
  }
}

/* $Id: petscarchiverfe.cpp,v 1.5 2001/04/11 07:48:16 buschelm Exp buschelm $ */
#include <stdlib.h>
#include <process.h>
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
    int len;
    LI li;
    const char **args; 
    if (verbose) {
      li = archivearg.begin();
      string archive = *li++;
      Merge(archive,archivearg,li);
      Merge(archive,file,file.begin());
      cout << archive << endl;
      cout.flush();
    }
    /*      system(archive.c_str()); */
    len = archivearg.size();
    len += file.size();
    args = (const char **)malloc((len+1)*sizeof(char *));
    int i=0;
    for (li=archivearg.begin();li!=archivearg.end();i++,li++) {
      args[i] = (*li).c_str();
    }
    for (li=file.begin();li!=file.end();i++,li++) {
      args[i] = (*li).c_str();
    }
    args[len+1] = NULL;
    /*      _execvp(args[0],args); */ 
    _spawnvp(_P_WAIT,args[0],args);
    free(args);
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

/* $Id: petscarchiverfe.cpp,v 1.15 2001/05/04 00:39:01 buschelm Exp $ */
#include <stdlib.h>
#include <process.h>
#include "petscarchiverfe.h"
#include <string.h>

using namespace PETScFE;

void archiver::Parse(void) {
  tool::Parse();
  LI i = arg.begin();
  archivearg.push_front(*i++);
  arg.pop_front();
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
    string archivename = file.front();
    file.pop_front();
    string::size_type n = archivename.find_last_of("\\");
    if (n!=string::npos) {
      string dir = archivename.substr(0,n);
      if (GetShortPath(dir)) {
        archivename = dir + archivename.substr(n);
      } else {
        cerr << "win32fe: Directory not found: ";
        cerr << archivename.substr(0,n);
      }
    }
    file.push_front(archivename);
  }
}

void archiver::Archive(void) {
  LI li = archivearg.begin();
  string header = *li++;
  Merge(header,archivearg,li);
  li = file.begin();
  string archive;
  while (li != file.end()) {
    /* Invoke archiver several times to limit arg length <512 chars */
    archive = header;
    Merge(archive,file,li);
    if (verbose)
      cout << archive << endl;
    system(archive.c_str());
  }
}

void archiver::Help(void) {
  tool::Help();
  cout << "For archivers:" << endl;
  cout << "  The first file specified will be the archive name." << endl;
  cout << "  All subsequent files will be inserted into the archive." << endl << endl;
  cout << "Ex: win32fe tlib -u libfoo.lib foo.o bar.o" << endl << endl;
  cout << "=========================================================================" << endl << endl;
  
}

void archiver::FoundFlag(LI &i) {
  string temp = *i;
  if (temp == "-help") {
    helpfound = true;
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

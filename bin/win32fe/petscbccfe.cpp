/* $Id: petscbccfe.cpp,v 1.16 2001/05/03 20:04:15 buschelm Exp buschelm $ */
#include <vector>
#include <stdlib.h>
#include <Windows.h>
#include "petscbccfe.h"

using namespace PETScFE;

bcc::bcc() {
  compileoutflag = "-o";
  linkoutflag    = "-e";

  OutputFlag = compilearg.end();
}

void bcc::Parse(void) {
  linkarg.push_front("**");
  compiler::Parse();
  if (!verbose) {
    compilearg.push_back("-q");
  }
}

void bcc::Link(void) {
  if (OutputFlag==compilearg.end()) {
    linkarg.pop_front();
    string tempstr = linkoutflag + file.front();
    tempstr.replace(tempstr.rfind("."),string::npos,".exe");
    linkarg.push_front(tempstr);
  }

  /* Copy file.o's to /tmp/file.obj's */ 
  int i,max_buffsize;
  char path[MAX_PATH];
  LI f;
  max_buffsize = MAX_PATH*sizeof(char);
  GetTempPath(max_buffsize,path);  /* Win32 Specific */
  vector<string> ext(file.size(),"");
  vector<string> temp(file.size(),"");
  for (i=0,f=file.begin();f!=file.end();i++,f++) {
    int n = (*f).find_last_of(".");
    ext[i] = (*f).substr(n,string::npos);
    if (ext[i] == ".o") {
      temp[i]=*f;
      string outfile = (string)path + (*f).substr(0,n) + ".obj";
      string copy = "copy " + temp[i] + " " + outfile;
      if (verbose) {
        cout << copy << endl;
      }
      CopyFile(temp[i].c_str(),outfile.c_str(),FALSE); /* Win32 Specific */
//        system(copy.c_str());
      f = file.erase(f);
      f = file.insert(f,outfile);
    }
  }

  /* Link, note linkargs before files */
  f = compilearg.begin();
  string link = *f++;
  Merge(link,compilearg,f);
  f = linkarg.begin();
  Merge(link,linkarg,f);
  f = file.begin();
  Merge(link,file,f);
  if (verbose) {
    cout << link << endl;
  }
  system(link.c_str());

  /* Remove /tmp/file.obj's */
  for (i=0,f=file.begin();f!=file.end();i++,f++) {
    if (ext[i] == ".o") {
      string del = "del " + *f;
      if (verbose) {
        cout << del << endl;
      }
//        system(del.c_str());
      DeleteFile((*f).c_str()); /* Win32 Specific */
      f = file.erase(f);
      f = file.insert(f,temp[i]);
    }
  }
}

void bcc::Help(void) {
  compiler::Help();
  cout << "bcc32 specific help:" << endl;
  cout << "  Note: The bcc32 option -l conflicts with the win32fe use of -l." << endl;
  cout << "        The following additional options are enabled for bcc32." << endl << endl;
  cout << "  -l:<flag>    enables <flag> for the linker, ilink32.exe" << endl;
  cout << "  -l:-<flag>   disables <flag> for the linker, ilink32.exe" << endl << endl;
  cout << "=========================================================================" << endl << endl;
  string help = compilearg.front();
  system(help.c_str());
}

void bcc::Foundl(LI &i) {
  string temp = *i;
  if (temp[2]==':') {
    linkarg.push_back("-l" + temp.substr(3));
  } else {
    compiler::Foundl(i);
  }
} 

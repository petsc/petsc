/* $Id: petscbccfe.cpp,v 1.12 2001/04/17 21:17:47 buschelm Exp $ */
#include <vector>
#include <stdlib.h>
#include <Windows.h>
#include "petscbccfe.h"

using namespace PETScFE;

bcc::bcc() {
  OutputFlag = compilearg.end();
}

void bcc::GetArgs(int argc,char *argv[]) {
  compiler::GetArgs(argc,argv);
  if (!verbose) {
    compilearg.push_front("-q");
  }
  linkarg.push_front("**");
}
void bcc::Parse(void) {
  compiler::Parse();
  FixOutput();
}

void bcc::Compile(void) {
  compileoutflag = "-o";
  compiler::Compile();
}

void bcc::Link(void) {
  if (OutputFlag==compilearg.end()) {
    linkarg.pop_front();
    string tempstr = "-e" + file.front();
    tempstr.replace(tempstr.rfind("."),string::npos,".exe");
    linkarg.push_front(tempstr);
  }

  /* Copy file.o's to /tmp/file.obj's */ 
  int i,max_buffsize;
  char path[256];
  LI f;
  max_buffsize = 256*sizeof(char);
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
      if (verbose) cout << copy << endl;
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
  Merge(link,linkarg,linkarg.begin());
  Merge(link,file,file.begin());
  if (verbose) cout << link << endl;
  system(link.c_str());

  /* Remove /tmp/file.obj's */
  for (i=0,f=file.begin();f!=file.end();i++,f++) {
    if (ext[i] == ".o") {
      string del = "del " + *f;
      if (verbose) cout << del << endl;
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

void bcc::FoundD(LI &i) {
  string temp = *i;
  ProtectQuotes(temp);
  compilearg.push_back(temp);  
}

void bcc::Foundl(LI &i) {
  string temp = *i;
  if (temp[2]==':') {
    linkarg.push_back("-l" + temp.substr(3));
  } else {
    file.push_back("lib" + temp.substr(2) + ".lib");
  }
} 

void bcc::Foundo(LI &i){ 
  i++;
  arg.pop_front();
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  /* Set Flag then fix later based on compilation or link */
  compilearg.push_back("-x" + temp);
  OutputFlag = --compilearg.end();
  /* Should perform some error checking ... */
}   

void bcc::FixOutput(void) {
  if (OutputFlag!=compilearg.end()) {
    string temp = *OutputFlag;
    if (linkarg.front()=="-c") {
      temp[1] = 'o';
    } else {
      temp[1] = 'e';
      linkarg.pop_front();
      linkarg.push_front(temp);
      compilearg.erase(OutputFlag);
    }
  }
}

/* $Id: petscbccfe.cpp,v 1.7 2001/03/28 21:03:32 buschelm Exp $ */
#include <iostream>
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
    string temp = *compilearg.begin();
    compilearg.pop_front();
    temp += " -q";
    compilearg.push_front(temp);
  }
  linkarg.push_front("**");
}
void bcc::Parse(void) {
  compiler::Parse();
  FixOutput();
}

void bcc::Compile(void) {
  LI i = compilearg.begin();
  string compile = *i++;
  Merge(compile,compilearg,i);
  /* Execute each compilation one at a time */
  i = file.begin();
  while (i!=file.end()) {
    /* Make default output a .o not a .obj */
    string outfile;
    if (OutputFlag==compilearg.end()) {
      outfile = "-o" + *i;
      int n = outfile.find_last_of(".");
      string filebase = outfile.substr(0,n);
      outfile = filebase + ".o";
    }
    string compileeach = compile + " " + outfile + " " + *i++;
    if (verbose) cout << compileeach << endl;
    system(compileeach.c_str());
  }
}

void bcc::Link(void) {
  if (OutputFlag==compilearg.end()) {
    linkarg.pop_front();
    string tempstr = "-e" + *file.begin();
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
    if (*linkarg.begin()=="-c") {
      temp[1] = 'o';
    } else {
      temp[1] = 'e';
      linkarg.pop_front();
      linkarg.push_front(temp);
      compilearg.erase(OutputFlag);
    }
  }
}

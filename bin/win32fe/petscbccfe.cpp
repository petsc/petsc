/* $Id: petscbccfe.cpp,v 1.1 2001/03/06 23:58:18 buschelm Exp $ */
#include <iostream>
#include <stdlib.h>
#include <Windows.h>
#include "petscbccfe.h"

using namespace PETScFE;

bcc::bcc() {
  OutputFlag = 0;
}

void bcc::GetArgs(int argc,char *argv[]) {
  compiler::GetArgs(argc,argv);
  if (!verbose) compilearg[0] = compilearg[0] + " -q";
  linkarg[0]="**";
}
void bcc::Parse(void) {
  compiler::Parse();
  FixOutput();
}

void bcc::Compile(void) {
  compiler::Squeeze();

  string compile = compilearg[0];
  Merge(compile,compilearg,1);
  /* Execute each compilation one at a time */ 
  for (int i=0;i<file.size();i++) {
    /* Make default output a .o not a .obj */
    string outfile;
    if (OutputFlag==0) {
      outfile = "-o" + file[i];
      int n = outfile.find_last_of(".");
      string filebase = outfile.substr(0,n);
      outfile = filebase + ".o";
    }
    if (file[i]!="") {
      string compileeach = compile + " " + outfile + " " + file[i];
      if (verbose) cout << compileeach << endl;
      system(compileeach.c_str());
    }
  }
}

void bcc::Link(void) {
  compiler::Squeeze();
  if (!OutputFlag) {
    linkarg[0] = "-e"+file[0];
    linkarg[0].replace(linkarg[0].rfind("."),string::npos,".exe");
  }

  /* Copy file.o's to /tmp/file.obj's */ 
  int i,max_buffsize,pathlength;
  char path[128];
  max_buffsize = 128*sizeof(char);
  pathlength = GetTempPath(max_buffsize,path);  /* Win32 Specific */
  vector<string> ext(file.size(),"");
  vector<string> temp(file.size(),"");
  for (i=0;i<file.size();i++) {
    if (file[i]=="") break;
    int n = file[i].find_last_of(".");
    ext[i] = file[i].substr(n,string::npos);
    if (ext[i] == ".o") {
      temp[i]=file[i];
      string outfile = (string)path + file[i].substr(0,n) + ".obj";
      string copy = "copy " + file[i] + " " + outfile;
      if (verbose) cout << copy << endl;
      system(copy.c_str());
      file[i] = outfile;
    }
  }

  /* Link, note linkargs before files */
  string link = compilearg[0];
  Merge(link,compilearg,1);
  Merge(link,linkarg,0);
  Merge(link,file,0);
  if (verbose) cout << link << endl;
  system(link.c_str());

  /* Remove /tmp/file.obj's */
  for (i=0;i<file.size();i++) {
    if (file[i]=="") break;
    if (ext[i] == ".o") {
      string del = "del " + file[i];
      if (verbose) cout << del << endl;
      system(del.c_str());
      file[i] = temp[i];
    }
  }
}
void bcc::FoundD(int &loc,string temp) {
  string::size_type i,j;
  i = temp.find("\"");
  if (i!=string::npos) {
    temp = temp.substr(0,i+1)+"\\\""+temp.substr(i+1,string::npos);
    j = temp.rfind("\"");
    if (j!=i+2) {
      temp = temp.substr(0,j)+"\\\""+temp.substr(j,string::npos);
    }
  }
  compilearg[loc]=temp;  
}


void bcc::FoundI(int &loc,string temp) {
  string str = temp.substr(2,string::npos);
  str = "\"" + str + "\"";
  compilearg[loc] = temp.substr(0,2) + str;
  ReplaceSlashWithBackslash(compilearg[loc]);
}

void bcc::FoundL(int &loc,string temp) {
  string str = temp.substr(2,string::npos);
  str = "\"" + str + "\"";
  linkarg[loc] = temp.substr(0,2) + str;
  ReplaceSlashWithBackslash(linkarg[loc]);
  compilearg[loc] = "";
}

void bcc::Foundl(int &loc,string temp) { 
  file[loc] = "lib" + temp.substr(2) + ".lib";
  compilearg[loc] = "";
} 

void bcc::Foundo(int &loc,string temp){ 
  /* Set Flag then fix later based on compilation or link */
  OutputFlag = loc;
  compilearg[loc] = "-x" + arg[loc+1];
  arg[++loc] = "";
  /* Should perform some error checking ... */
}   

void bcc::FixOutput(void) {
  if (OutputFlag!=0) {
    if (linkarg[0]=="-c") {
      compilearg[OutputFlag][1]='o';
    } else {
      compilearg[OutputFlag][1]='e';
      linkarg[0]=compilearg[OutputFlag];
      compilearg[OutputFlag]="";
    }
  }
}

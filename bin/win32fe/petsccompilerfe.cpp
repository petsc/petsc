/* $Id:$ */
#include <iostream>
#include <stdlib.h>
#include "petscfe.h"

using namespace PETScFE;

#define UNKNOWN '*'

compiler::compiler() {
  OptionTags = "DILclo";
  Options['D'] = &PETScFE::compiler::FoundD;
  Options['I'] = &PETScFE::compiler::FoundI;
  Options['L'] = &PETScFE::compiler::FoundL;
  Options['c'] = &PETScFE::compiler::Foundc;
  Options['l'] = &PETScFE::compiler::Foundl;
  Options['o'] = &PETScFE::compiler::Foundo;
  Options[UNKNOWN] = &PETScFE::compiler::FoundUnknown;
}

void compiler::GetArgs(int argc,char *argv[]) {
  tool::GetArgs(argc,argv);
  compilearg.resize(argc-1);
  compilearg[0] = arg[0];
  file.resize(argc-1);
  linkarg.resize(argc-1);
}

void compiler::Parse(void) {
  for (int i=1;i<arg.size();i++) {
    string temp = arg[i];
    if (temp[0]!='-') {
      FoundFile(i,temp);
    } else {
      char flag = temp[1];
      if (OptionTags.find(flag)==string::npos) {
        (this->*Options[UNKNOWN])(i,temp);
      } else {
        (this->*Options[flag])(i,temp);
      }
    }
  }
}

void compiler::Execute(void) {
  tool::Execute();
  if (linkarg[0]=="-c") {
    Compile();
  } else {
    Link();
  }
}

void compiler::Compile(void) {
  Squeeze();

  string compile=compilearg[0];
  Merge(compile,compilearg,1);
  Merge(compile,file,0);
  if (!quiet) cout << compile << endl;
  system(compile.c_str());
}

void compiler::Link(void) {
  Squeeze();

  string link = compilearg[0];
  Merge(link,compilearg,1);
  Merge(link,file,0);
  Merge(link,linkarg,0);
  if (!quiet) cout << link << endl;
  system(link.c_str());
}

void compiler::FoundFile(int &loc,string temp) {
  file[loc]=temp;
  ReplaceSlashWithBackslash(file[loc]);
  compilearg[loc] = "";
}

void compiler::FoundD(int &loc,string temp) {
  compilearg[loc] = temp;
}

void compiler::FoundI(int &loc,string temp) {
  compilearg[loc] = temp;
  ReplaceSlashWithBackslash(compilearg[loc]);
}

void compiler::FoundL(int &loc,string temp) {
  linkarg[loc] = temp;
  ReplaceSlashWithBackslash(linkarg[loc]);
  compilearg[loc] = "";
}

void compiler::Foundc(int &loc,string temp) {
  compilearg[loc] = temp;
  linkarg[0] = temp;
}

void compiler::Foundl(int &loc,string temp) { 
  file[loc] = temp;
  compilearg[loc] = "";
} 

void compiler::Foundo(int &loc,string temp){ 
  compilearg[loc] = temp + " " + arg[loc+1];
  arg[++loc] = "";
  /* Should perform some error checking ... */
}   

void compiler::FoundUnknown(int &loc,string temp){
  compilearg[loc] = temp;
}

void compiler::Squeeze(void) {
  tool::Squeeze(compilearg);
  tool::Squeeze(file);
  tool::Squeeze(linkarg);
}

/* $Id: petsccompilefe.cpp,v 1.1 2001/03/06 23:58:18 buschelm Exp $ */
#include <iostream>
#include <stdlib.h>
#include "petscfe.h"

using namespace PETScFE;

#define UNKNOWN '*'

compiler::compiler() {
  OptionTags = "DILclo";
  Options['D'] = &compiler::FoundD;
  Options['I'] = &compiler::FoundI;
  Options['L'] = &compiler::FoundL;
  Options['c'] = &compiler::Foundc;
  Options['l'] = &compiler::Foundl;
  Options['o'] = &compiler::Foundo;
  Options[UNKNOWN] = &compiler::FoundUnknown;
}

void compiler::GetArgs(int argc,char *argv[]) {
  tool::GetArgs(argc,argv);
  LI i = arg.begin();
  compilearg.push_front(*i);
  arg.pop_front();
}

void compiler::Parse(void) {
  LI i = arg.begin();
  while (i != arg.end()) {
    string temp = *i;
    if (temp[0]!='-') {
      FoundFile(i);
    } else {
      char flag = temp[1];
      if (OptionTags.find(flag)==string::npos) {
        (this->*Options[UNKNOWN])(i);
      } else {
        (this->*Options[flag])(i);
      }
    }
    i++;
    arg.pop_front();
  }
}

void compiler::Execute(void) {
  tool::Execute();
  LI i=linkarg.begin();
  if (*i == "-c") {
    Compile();
  } else {
    Link();
  }
}

void compiler::Compile(void) {
  LI i = compilearg.begin();
  string compile = *i++;
  Merge(compile,compilearg,i);
  Merge(compile,file,file.begin());
  if (verbose) cout << compile << endl;
  system(compile.c_str());
}

void compiler::Link(void) {
  LI i = compilearg.begin();
  string link = *i++;
  Merge(link,compilearg,i);
  Merge(link,file,file.begin());
  Merge(link,linkarg,linkarg.begin());
  if (verbose) cout << link << endl;
  system(link.c_str());
}

void compiler::FoundFile(LI &i) {
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  ProtectQuotes(temp);
  file.push_back(temp);
}

void compiler::FoundD(LI &i) {
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  ProtectQuotes(temp);
  compilearg.push_back(temp);
}

void compiler::FoundI(LI &i) {
  string temp = i->substr(2);
  ReplaceSlashWithBackslash(temp);
  GetShortPath(temp);
  temp = "-I"+temp;
  compilearg.push_back(temp);
}

void compiler::FoundL(LI &i) {
  string temp = i->substr(2);
  ReplaceSlashWithBackslash(temp);
  GetShortPath(temp);
  temp = "-I"+temp;
  linkarg.push_back(temp);
}

void compiler::Foundc(LI &i) {
  string temp = *i;
  compilearg.push_back(temp);
  linkarg.push_front(temp);
}

void compiler::Foundl(LI &i) { 
  file.push_back(*i);
} 

void compiler::Foundo(LI &i){
  compilearg.push_back(*i);
  i++;
  arg.pop_front();
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  ProtectQuotes(temp);
  compilearg.push_back(temp);
  /* Should perform some error checking ... */
}   

void compiler::FoundUnknown(LI &i){
  string temp = *i;
  compilearg.push_back(temp);
}

/* $Id:$ */
#include "petscfe.h"
#include <iostream>
#include <string>

using namespace PETScFE;

tool::tool(void) {
  tool::OptionTags= "use.quiet.arg";
  tool::Options["use"] = &tool::FoundUse;
  tool::Options["quiet"] = &tool::FoundQuiet;
  tool::Options["unknown"] = &tool::FoundArg;

  quiet = 0;
}
  
void tool::GetArgs(int argc,char *argv[]) {
  if (argc>2) { 
    arg.resize(argc-1); /* Skip argv[0] */ 
    for (int i=1;i<argc;i++) arg[i-1] = argv[i];
    tool::Parse(argc,argv);
    ReplaceSlashWithBackslash(arg[0]);
    Squeeze(arg);
  } else {
    cout << "Not enough arguments." << endl;
    cout << "Error: 2" << endl;
  }
}

void tool::Parse(int argc,char *argv[]) {
  /* argv[0] = "petscfe"  -- not parsed with args */
  /* argv[1] = The tool exe -- gotten in GetArgs */ 
  for (int i=1;i<argc-1;i++) {
    string temp = argv[i+1];
    if (temp.substr(0,2)!="--") {
      tool::FoundArg(i,temp);
    } else {
      string flag = temp.substr(2);
      if (tool::OptionTags.find(flag)==string::npos) {
        (this->*tool::Options["unknown"])(i,temp);
      } else {
        (this->*tool::Options[flag])(i,temp);
      }
    }
  }
}

void tool::Execute(void) {
 if (!quiet) cout << "PETSc Front End" << endl;
}

void tool::FoundArg(int &loc,string temp) {
  arg[loc] = temp;
}

void tool::FoundUse(int &loc,string temp) {
  arg[loc] = "";
  arg[0] = arg[loc+1];
  arg[++loc] = "";
}

void tool::FoundQuiet(int &loc,string temp) {
  quiet = -1;
  arg[loc] = "";
}

void tool::ReplaceSlashWithBackslash(string &name) {
  for (int i=0;i<name.length();i++)
    if (name[i]=='/') name[i]='\\';
}

void tool::PrintStringVector(vector<string> &strvec) {
  if (!quiet) cout << "Printing..." << endl;
  int size = strvec.size()-1;
  for (int i=0;i<size;i++) if (!quiet) cout << strvec[i] + " ";
  if (!quiet) cout << strvec[size] << endl;
}

void tool::Squeeze(vector<string> &strvec) {
  for (int i=0,current = 0;i<strvec.size();i++) {
    if (strvec[i]!="") {
      if (current!=i) {
        strvec[current++]=strvec[i];
        strvec[i]="";
      } else {
        current++;
      }
    }
  }
}

void tool::Merge(string &str,vector<string> &strvec,int start) {
  for (int i=start;i<strvec.size();i++) {
    /* if can be eliminated if use lists i/o vectors */
    if (strvec[i]=="") break;
    str += " " + strvec[i];
  }
}

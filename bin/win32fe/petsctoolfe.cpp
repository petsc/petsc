/* $Id: toolfe.cpp,v 1.8 2001/04/17 20:51:59 buschelm Exp buschelm $ */
#include "Windows.h"
#include "petsctoolfe.h"

using namespace PETScFE;

tool::tool(void) {
  tool::OptionTags= "uvh";
  tool::Options['u'] = &tool::FoundUse;
  tool::Options['v'] = &tool::FoundVerbose;
  tool::Options['h'] = &tool::FoundHelp;
  verbose   = 0;
  helpfound = 0;
}
  
void tool::GetArgs(int argc,char *argv[]) {
    for (int i=2;i<argc;i++) arg.push_back(argv[i]);
    if (argc == 2) arg.push_back("-help");
}

void tool::Parse() {
  LI i = arg.begin();
  while (i!=arg.end()) {
    string temp = *i;
    if (temp.substr(0,2)=="--") {
      char flag = temp[2];
      if (tool::OptionTags.find(flag)!=string::npos) {
        (this->*tool::Options[flag])(i);
      }
    } else {
      i++;
    }
  }
}

void tool::Execute(void) {
  if (verbose) cout << "PETSc Front End v1.0" << endl;
  if (helpfound) Help();
}

void tool::Help(void) {
  cout << "\tPETSc Front End v1.0" << endl << endl;  
  cout << "Usage: win32fe <tool> --<win32fe options> -<tool options> <files>" << endl;
  cout << "  <tool> must follow win32fe" << endl << endl;
  cout << "<tool>: {cl,df,f90,bcc32,lib,tlib}" << endl;
  cout << "  cl:    Microsoft 32-bit C/C++ Optimizing Compiler" << endl;
  cout << "  icl:   Intel C/C++ Optimizing Compiler" << endl;
  cout << "  df:    Compaq Visual Fortran Optimizing Compiler" << endl;
  cout << "  f90:   Compaq Visual Fortran90 Optimizing Compiler" << endl;
  cout << "  ifl:   Intel Fortran Optimizing Compiler" << endl;
  cout << "  bcc32: Borland C++ for Win32" << endl;
  cout << "  lib:   Microsoft Library Manager" << endl;
  cout << "  tlib:  Borland Library Manager" << endl << endl;
  cout << "<win32fe options>: {use,verbose,help}" << endl;
  cout << "  --use <arg>: <arg> Specifies the variant of <tool> to use" << endl;
  cout << "  --verbose:   Echo to stdout the translated commandline" << endl;
  cout << "  --help:      Output this help message and help for <tool>" << endl << endl;
  cout << "For compilers:" << endl;
  cout << "  win32fe will map the following <tool options> to their native options:" << endl;
  cout << "    -c:          Compile Only, generates an object file with .o extension" << endl;
  cout << "    -l<library>: Link the file lib<library>.lib" << endl;
  cout << "    -o <file>:   Output=<file> context dependent" << endl;
  cout << "    -D<macro>:   Define <macro>" << endl;
  cout << "    -I<path>:    Add <path> to the include path" << endl;
  cout << "    -L<path>:    Add <path> to the link path" << endl;
  cout << "    -help:       <tool> specific help for win32fe" << endl << endl;
  cout << "Ex: win32fe cl -Zi -c foo.c --verbose -Iinclude" << endl << endl;
  cout << "For archivers:" << endl;
  cout << "  The first file specified will be the archive name." << endl;
  cout << "  All subsequent files will be inserted into the archive." << endl << endl;
  cout << "Ex: win32fe tlib -u libfoo.lib foo.o bar.o" << endl << endl;
  cout << "=========================================================================" << endl << endl;
}

void tool::FoundUse(LI &i) {
  if (*i=="--use") {
    i = arg.erase(i);
    if (i!=arg.end()) {
      ReplaceSlashWithBackslash(*i);
      arg.pop_front();
      arg.push_front(*i);
      i = arg.erase(i);
    } else {
      i--;
      arg.push_back("--help");
      i--;
    }
  }
}

void tool::FoundVerbose(LI &i) {
  if (*i == "--verbose") {
    verbose = -1;
    i = arg.erase(i);
  }
}

void tool::FoundHelp(LI &i) {
  if (*i == "--help") {
    helpfound = -1;
    arg.push_back("-help");
    i = arg.erase(i);
  }
}

void tool::ReplaceSlashWithBackslash(string &name) {
  for (int i=0;i<name.length();i++)
    if (name[i]=='/') name[i]='\\';
}

void tool::ProtectQuotes(string &name) {
  string::size_type a,b;
  a = name.find("\"");
  if (a!=string::npos) {
    string temp = name.substr(0,a+1);
    temp += "\\\"";
    temp += name.substr(a+1,string::npos);
    name = temp;
    b = name.rfind("\"");
    if (b!=a+2) {
      temp = name.substr(0,b);
      temp += "\\\"";
      temp += name.substr(b,string::npos);
      name = temp;
    }
  }
}
void tool::GetShortPath(string &name) {
  if (name[0]=='\''||name[0]=='\"')
    name=name.substr(1,name.length()-2);
  char shortpath[256];
  int length=256*sizeof(char);
  GetShortPathName(name.c_str(),shortpath,length);
  name=(string)shortpath;
}

void tool::PrintListString(list<string> &liststr) {
  cout << "Printing..." << endl;
  LI i = liststr.begin();
  while (i!=liststr.end()) cout << *i << " ";
  cout << endl;
}

void tool::Merge(string &str,list<string> &liststr,LI &i) {
  while (i!=liststr.end()) {
    str += " " + *i++;
  }
}

/* $Id: petsctoolfe.cpp,v 1.16 2001/05/05 02:16:22 buschelm Exp buschelm $ */
#include "Windows.h"
#include "petsctoolfe.h"

using namespace PETScFE;

tool::tool(void) {
  tool::OptionTags= "uvhpV";
  tool::Options['u'] = &tool::FoundUse;
  tool::Options['v'] = &tool::FoundVerbose;
  tool::Options['h'] = &tool::FoundHelp;
  tool::Options['p'] = &tool::FoundPath;
  tool::Options['w'] = &tool::FoundWoff;
  tool::Options['V'] = &tool::FoundVersion;
  tool::Options[UNKNOWN] = &tool::FoundUnknown;
  version   = "PETSc's Win32 Front End, version 1.1";
  woff      = false;
  verbose   = false;
  helpfound = false;
  versionfound = false;
  inpath    = false;
}
  
void tool::GetArgs(int argc,char *argv[]) {
  for (int i=2;i<argc;i++) arg.push_back(argv[i]);
  if (argc == 2) {
    if (!((arg.front()=="--Version") || (arg.front()=="--help"))) {
      arg.push_back("--help");
    }
  }
}

void tool::Parse() {
  LI i = arg.begin();
  while (i!=arg.end()) {
    string temp = *i;
    if (temp.substr(0,2)=="--") {
      char flag = temp[2];
      if (tool::OptionTags.find(flag)==string::npos) {
        (this->*tool::Options[UNKNOWN])(i);
      } else {
        (this->*tool::Options[flag])(i);
      }
    } else {
      i++;
    }
  }
  if (IsAKnownTool()) {
    AddSystemInfo();
  } else {
    if (!(helpfound || versionfound)) {
      if (arg.begin()!=arg.end()) {
        cerr << endl << "Error: win32fe: Unknown Tool: " << arg.front() << endl;
        cerr << "  Use --help for more information on win32fe options." << endl << endl;
      } else {
        cerr << endl << "Error: win32fe: Unknown Tool:" << endl;
        cerr << "  Use --help for more information on win32fe options." << endl << endl;
      }
    }
  }
}

void tool::AddSystemInfo(void) {
  FindInstallation();
  AddPaths();
}

void tool::FindInstallation(void) {
  /* Find location of tool */
  InstallDir = arg.front();
  bool ierr = false;
  
    /* First check if a full path was specified with --use */
  string::size_type n = InstallDir.find(":");
  if (n==string::npos) {
    char tmppath[MAX_PATH],*tmp;
    int length = MAX_PATH*sizeof(char);
    string extension = ".exe";
    if (SearchPath(NULL,InstallDir.c_str(),extension.c_str(),length,tmppath,&tmp)) {
      /*
        Note: There are 6 locations searched, the 6th of these is the PATH variable.
        But, in reality, the first 5 don't make any sense for this tool, so it is 
        assumed that the tool was found in the PATH variable.
      */
      InstallDir = (string)tmppath;
    } else {
      ierr = true;
    }
  }
  InstallDir = InstallDir.substr(0,InstallDir.find_last_of("\\"));
  InstallDir = InstallDir.substr(0,InstallDir.find_last_of("\\")+1);
  if (GetShortPath(InstallDir)) {
    inpath = true;
  } else {
    ierr = true;
  }
  if (ierr) {
    string tool=arg.front();
    cerr << endl << "Error: win32fe: Tool Not Found: " << tool << endl;
    cerr << "  Specify the complete path to " << tool << " with --use" << endl;
    cerr << "  Use --help for more information on win32fe options." << endl << endl;
  }
}

void tool::AddPaths(void) {
  if (!inpath) {
    arg.push_back("--path");
    LI i = arg.end();
    i--;
    arg.push_back(InstallDir);
    FoundPath(i);
  }
}

void tool::Execute(void) {
  if (helpfound) {
    Help();
    return;
  }
  if (versionfound) {
    DisplayVersion();
    return;
  }
  if (verbose) cout << "PETSc Front End v1.1" << endl;
}

void tool::Help(void) {
  cout << endl << "PETSc's Win32 Tool Front End v1.0" << endl << endl;  
  cout << "Usage: win32fe <tool> --<win32fe options> -<tool options> <files>" << endl;
  cout << "  <tool> must be the first argument to win32fe" << endl << endl;
  cout << "<tool>: {cl,icl,df,f90,ifl,bcc32,lib,tlib}" << endl;
  cout << "  cl:    Microsoft 32-bit C/C++ Optimizing Compiler" << endl;
  cout << "  icl:   Intel C/C++ Optimizing Compiler" << endl;
  cout << "  df:    Compaq Visual Fortran Optimizing Compiler" << endl;
  cout << "  f90:   Compaq Visual Fortran90 Optimizing Compiler" << endl;
  cout << "  ifl:   Intel Fortran Optimizing Compiler" << endl;
  cout << "  bcc32: Borland C++ for Win32" << endl;
  cout << "  lib:   Microsoft Library Manager" << endl;
  cout << "  tlib:  Borland Library Manager" << endl << endl;
  cout << "<win32fe options>: {help,path,use,verbose,Version,woff}" << endl;
  cout << "  --help:       Output this help message and help for <tool>" << endl << endl;
  cout << "  --path <arg>: <arg> specifies an addition to the PATH that is required" << endl;
  cout << "                (ex. the location of a required .dll)" << endl;
  cout << "  --use <arg>:  <arg> specifies the variant of <tool> to use" << endl;
  cout << "  --verbose:    Echo to stdout the translated commandline" << endl;
  cout << "  --Version:    Output Version info for win32fe and <tool>" << endl;
  cout << "  --woff:       Suppress win32fe specific warning messages" << endl << endl;
  cout << "=========================================================================" << endl << endl;
}

void tool::FoundUnknown(LI &i) {
  i++;
}

void tool::FoundPath(LI &i) {
  if (*i=="--path") {
    i = arg.erase(i);
    if (i!=arg.end()) {
      int length = 1024*sizeof(char);
      char buff[1024];
      string path="PATH";
      GetEnvironmentVariable(path.c_str(),buff,length);
      string newpath = (string)buff;
      if (verbose) cout << "win32fe: Adding to path: " << *i << endl;
      newpath = *i + ";" + newpath;
      SetEnvironmentVariable(path.c_str(),newpath.c_str());
      i = arg.erase(i);
    } else {
      i--;
      arg.push_back("--help");
      i--;
    }
  }
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

void tool::FoundWoff(LI &i) {
  if (*i == "--woff") {
    woff = true;
    i = arg.erase(i);
    cout << endl;
  } else {
    i++;
  }
}

void tool::FoundVerbose(LI &i) {
  if (*i == "--verbose") {
    verbose = true;
    i = arg.erase(i);
    cout << endl;
  } else {
    i++;
  }
}

void tool::FoundVersion(LI &i) {
  if (*i == "--Version") {
    versionfound=true;
    i = arg.erase(i);
  } else {
    i++;
  }
}

void tool::FoundHelp(LI &i) {
  if (*i == "--help") {
    helpfound = true;
    i = arg.erase(i);
  } else {
    i++;
  }
}

void tool::FoundFile(LI &i) {
  string temp = *i;
  ReplaceSlashWithBackslash(temp);
  string::size_type n = temp.find_last_of("\\");
  if (n!=string::npos) {
    string dir = temp.substr(0,n);
    if (GetShortPath(dir)) {
      temp = dir + temp.substr(n);
    } else {
      cerr << "Error: win32fe: File Not Found: ";
      cerr << temp << endl;
      return;
    }
  }
  file.push_back(temp);
}

void tool::ReplaceSlashWithBackslash(string &name) {
  for (string::size_type i=0;i<name.length();i++)
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

int tool::GetShortPath(string &name) {
  if (name[0]=='\''||name[0]=='\"')
    name=name.substr(1,name.length()-2);
  char shortpath[MAX_PATH];
  int length=MAX_PATH*sizeof(char);
  int size = GetShortPathName(name.c_str(),shortpath,length);
  name=(string)shortpath;
  return(size);
}

void tool::PrintListString(list<string> &liststr) {
  cout << "Printing..." << endl;
  LI i = liststr.begin();
  while (i!=liststr.end()) cout << *i++ << " ";
  cout << endl;
}

void tool::Merge(string &str,list<string> &liststr,LI &i) {
  while (i!=liststr.end()) {
    str += " " + *i++;
  }
}

void tool::DisplayVersion(void) {
  cout << version << endl;
}

bool tool::IsAKnownTool(void) {
  return(false);
}

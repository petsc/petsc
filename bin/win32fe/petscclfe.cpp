/* $Id: petscclfe.cpp,v 1.21 2001/05/06 07:44:17 buschelm Exp buschelm $ */
#include <stdlib.h>
#include "petscclfe.h"
#include "Windows.h"

using namespace PETScFE;

cl::cl() {
  compileoutflag = "-Fo";
  linkoutflag    = "-Fe";

  OutputFlag = compilearg.end();
}

void cl::Parse(void) {
  linkarg.push_front("-link");
  compiler::Parse();
  if (!verbose) {
    compilearg.push_back("-nologo");
  }
}

void cl::FindInstallation(void) {
  tool::FindInstallation();
  string::size_type n = InstallDir.length()-1;
  VisualStudioDir = InstallDir.substr(0,n);
  n = VisualStudioDir.rfind("\\");
  VisualStudioDir = VisualStudioDir.substr(0,n+1);
  if (verbose) {
    string longpath;
    GetLongPath(VisualStudioDir,longpath);
    cout << "win32fe: Visual Studio Installation: " << longpath << endl;
  }
  VSVersion = InstallDir.substr(0,InstallDir.length()-1);
  VSVersion = VSVersion.substr(VisualStudioDir.length());
  if (verbose) {
    cout << "win32fe: Visual C++ Version: " << VSVersion << endl;
  }
}

void cl::AddPaths(void) {
  /* Find required .dll's */
  string addpath;
  bool KnownVersion=false;
  if (VSVersion=="VC98") {
    addpath = VisualStudioDir + "Common\\MSDev98\\Bin";
    KnownVersion=true;
  } else if (VSVersion=="VC7") {
    addpath = VisualStudioDir + "Common7\\IDE";
    KnownVersion=true;
  } else {
    if (!woff) {
      cerr << "Warning: win32fe Visual Studio version not recognized." << endl;
    }
  }
  if (KnownVersion) {
    arg.push_back("--path");
    LI i = arg.end();
    i--;
    GetShortPath(addpath);
    arg.push_back(addpath);
    FoundPath(i);
  }
}

void cl::Help(void) {
  compiler::Help();
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

void cl::FoundL(LI &i) {
  string temp = i->substr(2);
  ReplaceSlashWithBackslash(temp);
  if (GetShortPath(temp)) {
    temp = "-libpath:"+temp;
    linkarg.push_back(temp);
  } else {
    if (!woff) {
      cerr << "Warning: win32fe Library Path Not Found:" << i->substr(2) << endl;
    }
  }
}

void df::Help(void) {
  compiler::Help();
  cout << "df specific help:" << endl;
  cout << "  For mixed language C/FORTRAN programming in PETSc, the location of the" << endl;
  cout << "      C system libraries must also be specified in the LIB environment" << endl;
  cout << "      variable or with -L<dir> since the installed location of these" << endl;
  cout << "      libraries may be independent of the FORTRAN installation." << endl;
  cout << "      If installed, the Visual Studio 6, C system libraries are" << endl;
  cout << "      automatically found and added to the path." << endl << endl;
  cout << "=========================================================================" << endl << endl;
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

void df::AddPaths(void) {
  string addpath1,addpath2,DFVersion;
  addpath1 = DFVersion = InstallDir.substr(0,InstallDir.length()-1);
  DFVersion = DFVersion.substr(DFVersion.rfind("\\")+1);
  addpath2 = addpath1 = addpath1.substr(0,addpath1.rfind("\\")+1);

  if (DFVersion=="DF98") {
    addpath1 += "Common\\MSDev98\\Bin";
    addpath2 += "VC98\\Bin";
    GetShortPath(addpath1);
    GetShortPath(addpath2);
    string addpath = addpath1 + ";" + addpath2;
    arg.push_back("--path");
    LI i = arg.end();
    i--;
    arg.push_back(addpath);
    FoundPath(i);
  } else {
    if (!woff) {
      cerr << "Warning: win32fe df version not recognized." << endl;
    }
  }
}

void df::AddSystemLib(void) {
  compiler::AddSystemLib();
  string::size_type len_m_1 = InstallDir.length()-1;
  string libdir,DFVersion;
  libdir = DFVersion = InstallDir.substr(0,len_m_1);
  DFVersion = DFVersion.substr(DFVersion.rfind("\\")+1,len_m_1);
  libdir = libdir.substr(0,libdir.rfind("\\")+1);

  if (DFVersion=="DF98") {
    libdir += "VC98\\Lib";
    GetShortPath(libdir);
    if (verbose) {
      string longpath;
      GetLongPath(libdir,longpath);
      cout << "win32fe: Adding Flag: -L" << longpath << endl;
    }
    libdir = "-L" + libdir;
    LI i = arg.end();
    arg.push_back(libdir);
    i--;
    FoundL(i);
  } else {
    if (!woff) {
      cerr << "Warning: win32fe df version not recognized." << endl;
    }
  }
}
    

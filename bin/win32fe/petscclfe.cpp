/* $Id: petscclfe.cpp,v 1.20 2001/05/05 02:16:22 buschelm Exp buschelm $ */
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
  n = VisualStudioDir.find_last_of("\\");
  VisualStudioDir = VisualStudioDir.substr(0,n+1);
  VSVersion = InstallDir.substr(0,InstallDir.length()-1);
  VSVersion = VSVersion.substr(VisualStudioDir.length());
}

void cl::AddPaths(void) {
  /* Find required .dll's */
  string addpath;
  /* This is ugly and perhaps each version should have their own class */
  bool KnownVersion=false;
  if (VSVersion=="VC98") {
    addpath = VisualStudioDir + "Common\\MSDev98\\Bin";
    KnownVersion=true;
  } else if (VSVersion=="VC7") {
    addpath = VisualStudioDir + "Common7\\IDE";
    KnownVersion=true;
  } else {
    cerr << "Warning: win32fe Visual Studio version not recognized." << endl;
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
  cout << "cl specific help:" << endl;
  cout << "  Note: Different versions of Visual Studio require the use of additional" << endl;
  cout << "        .dll's which may require the use of --path <arg> to specify." << endl;
  cout << "  Alternatively, you can add the required location through the Windows" << endl;
  cout << "        Start Menu->Control Panel->System folder, or by invoking win32fe" << endl;
  cout << "        and cl from a specialized command prompt provided with cl." << endl << endl;
  cout << "=========================================================================" << endl << endl;
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

void cl::FoundL(LI &i) {
  string temp = i->substr(2);
  ReplaceSlashWithBackslash(temp);
  GetShortPath(temp);
  temp = "-libpath:"+temp;
  linkarg.push_back(temp);
}

void df::Help(void) {
  compiler::Help();
  cout << "df specific help:" << endl;
  cout << "  df requires the Microsoft linker, link.exe, which must be found in" << endl;
  cout << "      the PATH.  The <directory> containing link.exe can alternatively be" << endl;
  cout << "      given to win32fe with --path <directory>" << endl;
  cout << "  For mixed language C/FORTRAN programming in PETSc, the location of the" << endl;
  cout << "      C system libraries must also be specified in the LIB environment" << endl;
  cout << "      variable or with -L<dir> since the installed location of these" << endl;
  cout << "      libraries may be independent of the FORTRAN installation." << endl << endl;
  cout << "=========================================================================" << endl << endl;
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

void df::AddPaths(void) {
  string::size_type len_m_1 = InstallDir.length()-1;
  string addpath1,addpath2,DFVersion;
  addpath1 = DFVersion = InstallDir.substr(0,len_m_1);
  DFVersion = DFVersion.substr(DFVersion.find_last_of("\\")+1,len_m_1);
  addpath2 = addpath1 = addpath1.substr(0,addpath1.find_last_of("\\")+1);

  bool KnownVersion=false;
  if (DFVersion=="DF98") {
    addpath1 += "Common\\MSDev98\\Bin";
    addpath2 += "VC98\\Bin";
    KnownVersion=true;
  } else {
    cerr << "Warning: win32fe df version not recognized." << endl;
  }
  if (KnownVersion) {
    GetShortPath(addpath1);
    GetShortPath(addpath2);
    string addpath = addpath1 + ";" + addpath2;
    arg.push_back("--path");
    LI i = arg.end();
    i--;
    arg.push_back(addpath);
    FoundPath(i);
  }
}

void df::AddSystemLib(void) {
  compiler::AddSystemLib();
  string::size_type len_m_1 = InstallDir.length()-1;
  string libdir,DFVersion;
  libdir = DFVersion = InstallDir.substr(0,len_m_1);
  DFVersion = DFVersion.substr(DFVersion.find_last_of("\\")+1,len_m_1);
  libdir = libdir.substr(0,libdir.find_last_of("\\")+1);

  bool KnownVersion=false;
  if (DFVersion=="DF98") {
    libdir += "VC98\\Lib";
    KnownVersion=true;
  } else {
    cerr << "Warning: win32fe df version not recognized." << endl;
  }
  if (KnownVersion) {
    GetShortPath(libdir);
    libdir = "-L" + libdir;
    LI i = arg.end();
    arg.push_back(libdir);
    i--;
    FoundL(i);
  }
}
    

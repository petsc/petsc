/* $Id: petscclfe.cpp,v 1.12 2001/04/17 21:17:31 buschelm Exp $ */
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
  /* Find required .dll's */
  cout << "Install Dir = " << InstallDir << endl;
  string::size_type len_m_1 = InstallDir.length()-1;
  string addpath,VCVersion;
  addpath = VCVersion = InstallDir.substr(0,len_m_1);
  VCVersion = VCVersion.substr(VCVersion.find_last_of("\\")+1,len_m_1);
  addpath = addpath.substr(0,addpath.find_last_of("\\")+1);
  bool KnownVersion=FALSE;
  if(VCVersion=="VC98") {
    addpath += "Common\\MSDev98\\Bin";
    KnownVersion=TRUE;
  } else if (VCVersion=="VC7") {
    addpath += "Common7\\IDE";
    KnownVersion=TRUE;
  } else {
    cerr << "Warning: win32fe cl version not recognized.";
  }
  if (KnownVersion) {
    arg.push_back("--path");
    LI i = arg.end();
    i--;
    cout << "Adding to path " << addpath << endl;
    arg.push_back(addpath);
    FoundPath(i);
  }
}

void cl::Help(void) {
  compiler::Help();
  cout << "cl specific help:" << endl;
  cout << "  Note: Different versions of cl require the use of additional .dll's" << endl;
  cout << "        which may require the use of --path <arg> to specify the location" << endl;
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

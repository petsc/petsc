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

void icl::Help(void) {
  compiler::Help();
  cout << "icl specific help:" << endl;
  cout << "  icl requires the use of the Microsoft Linker (link.exe), which is" << endl;
  cout << "      invoked seamlessly.  However certain .dll's are required which" << endl;
  cout << "      may not be found in the PATH.  The needed path can be specified" << endl;
  cout << "      with --path <directory>" << endl;
  cout << "  icl also requires the use of the Microsoft Compiler (cl) header files" << endl;
  cout << "      and system libraries.  However, these system files are installed" << endl;
  cout << "      independently from icl, and must be specified either in the" << endl;
  cout << "      environment variables INCLUDE and LIB, or with -I<dir> -L<dir>" << endl;
  cout << "  Alternatively, win32fe and icl can be invoked within a command prompt" << endl;
  cout << "      provided with the icl installation." << endl << endl;
  cout << "=========================================================================" << endl << endl;
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}
 
void ifl::Help(void) {
  compiler::Help();
  cout << "ifl specific help:" << endl;
  cout << "  ifl requires the use of the Microsoft Linker (link.exe), which is" << endl;
  cout << "      invoked seamlessly.  However certain .dll's are required which" << endl;
  cout << "      may not be found in the PATH.  The needed path can be specified" << endl;
  cout << "      with --path <directory>" << endl << endl;
  cout << "=========================================================================" << endl << endl;
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

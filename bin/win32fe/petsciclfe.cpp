/* $Id:$ */
#include <stdlib.h>
#include "petsciclfe.h"
#include "Windows.h"

using namespace PETScFE;

void icl::Parse(void) {
  linkarg.push_front("-link");
  compiler::Parse();
  if (!verbose) {
    compilearg.push_back("-nologo");
  }
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

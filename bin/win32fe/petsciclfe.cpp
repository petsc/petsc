/* $Id:$ */
#include <stdlib.h>
#include <Windows.h>
#include "petsciclfe.h"

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
  cout << "      invoked seamlessly.  However certain .dll's may be required which" << endl;
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
 
void icl::FindInstallation(void) {
  compiler::FindInstallation();

  /* Intel Compiler is currently only compatible with VS6.0, Visual C/C++ 98 */
  VSVersion = "VC98";

  /* We must also find the Visual Studio Installation */
  /* Microsoft stashes this info in the registry */
  HKEY key;
  string vskey="Software\\Microsoft\\VisualStudio\\6.0";
  long ierr = RegOpenKeyEx(HKEY_LOCAL_MACHINE,vskey.c_str(),NULL,KEY_QUERY_VALUE,&key);
  if (ierr==ERROR_SUCCESS) {
    unsigned long length=MAX_PATH*sizeof(char);
    unsigned char buff[MAX_PATH];
    VisualStudioDir = "InstallDir";
    long ierr = RegQueryValueEx(key,VisualStudioDir.c_str(),NULL,NULL,buff,&length);
    if (ierr==ERROR_SUCCESS) {
      VisualStudioDir = (string)((char *)buff);
      for (int i=0;i<3;i++) {
        string::size_type n = VisualStudioDir.find_last_of("\\");
        VisualStudioDir = VisualStudioDir.substr(0,n);
      }
      VisualStudioDir += "\\";
      GetShortPath(VisualStudioDir);
    } else {
      cout << "Warning: win32fe Cannot Find Visual Studio Installation" << endl;
      cout << "         Registry Key could not be queried." << endl;
    }
    RegCloseKey(key);
  } else {
    cout << "Warning: win32fe Cannot Find Visual Studio Installation" << endl;
    cout << "         Registry Key could not be found." << endl;
  }
}

void icl::AddSystemInclude(void) {
  compiler::AddSystemInclude();
  /* Also requires Visual C/C++ headers */
  arg.push_back("-I" + VisualStudioDir + "VC98\\include");
  LI i = arg.end();
  FoundI(--i);
  arg.pop_back();
}

void icl::AddSystemLib(void) {
  compiler::AddSystemLib();
  /* Also requires Visual C/C++ libraries */
  arg.push_back("-L" + VisualStudioDir + "VC98\\lib");
  LI i = arg.end();
  FoundL(--i);
  arg.pop_back();
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

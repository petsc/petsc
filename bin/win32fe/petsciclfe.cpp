/* $Id: petsciclfe.cpp,v 1.2 2001/05/04 22:02:13 buschelm Exp buschelm $ */
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
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}
 
void icl::FindInstallation(void) {
  compiler::FindInstallation();

  /* Intel Compiler is currently only compatible with VS6.0, Visual C/C++ 98 */
  VSVersion = "VC98";
  if (verbose) {
    cout << "win32fe: Visual C++ Version: " << VSVersion << endl;
  }

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
      /* buff = VisualStudioDir\Common\IDE\IDE98 */
      VisualStudioDir = (string)((char *)buff);
      for (int i=0;i<3;i++) {
        string::size_type n = VisualStudioDir.rfind("\\");
        VisualStudioDir = VisualStudioDir.substr(0,n);
      }
      VisualStudioDir += "\\";
      GetShortPath(VisualStudioDir);
      if (verbose) {
        string longpath;
        GetLongPath(VisualStudioDir,longpath);
        cout << "win32fe: Visual Studio Installation: " << longpath << endl;
      }
    } else {
      if (!woff) {
        cout << "Warning: win32fe Cannot Find Visual Studio Installation" << endl;
        cout << "         Registry Key could not be queried." << endl;
      }
    }
    RegCloseKey(key);
  } else {
    if (!woff) {
      cout << "Warning: win32fe Cannot Find Visual Studio Installation" << endl;
      cout << "         Registry Key could not be found." << endl;
    }
  }
}

void icl::AddSystemInclude(void) {
  compiler::AddSystemInclude();
  /* Also requires Visual C/C++ headers */
  string include = VisualStudioDir + "VC98\\include";
  if (verbose) {
    string longpath;
    GetLongPath(include,longpath);
    cout << "win32fe: Adding Flag -I" << longpath << endl;
  }
  arg.push_back("-I" + include);
  LI i = arg.end();
  FoundI(--i);
  arg.pop_back();
}

void icl::AddSystemLib(void) {
  compiler::AddSystemLib();
  /* Also requires Visual C/C++ libraries */
  string lib = VisualStudioDir + "VC98\\lib";
  if (verbose) {
    string longpath;
    GetLongPath(lib,longpath);
    cout << "win32fe: Adding Flag -L" << longpath << endl;
  }
  arg.push_back("-L" + lib);
  LI i = arg.end();
  FoundL(--i);
  arg.pop_back();
}

void ifl::Help(void) {
  compiler::Help();
  string help = compilearg.front();
  help += " -? 2>&1";
  system(help.c_str());
}

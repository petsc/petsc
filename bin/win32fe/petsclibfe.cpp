/* $Id: petsclibfe.cpp,v 1.10 2001/05/05 02:16:22 buschelm Exp buschelm $ */
#include "petsclibfe.h"

using namespace PETScFE;
  
void lib::Execute(void) {
  archiver::Execute();
  if (!helpfound) {
    string temp, archivename = file.front();
    file.pop_front();
    archivearg.push_back("-out:" + archivename);
    temp = archivename;
    if (GetShortPath(temp)) {
      file.push_front(archivename);
    }
  }
  Archive();
}

void lib::Archive(void) {
  LI li = archivearg.begin();
  string header = *li++;
  Merge(header,archivearg,li);
//    PrintListString(archivearg);
  li = file.begin();
  string archivename = file.front();
  while (li != file.end()) {
    string archive = header;
    Merge(archive,file,li);
    if (verbose) cout << archive << endl;
    system(archive.c_str());
    if (archivearg.back()!=archivename) {
      archivearg.push_back(archivename);
      Merge(header,archivearg,--archivearg.end());
    }
  }
}

void lib::Help(void) {
  archiver::Help();
  cout << "lib specific help:" << endl;
  cout << "  Note: win32fe will check to see if the library exists, and will modify" << endl;
  cout << "        the arguments to lib accordingly.  As a result, specifying" << endl;
  cout << "        -out:<libname> is unnecessary and may lead to unexpected results." << endl << endl;
  cout << "=========================================================================" << endl << endl;
  
  string help = archivearg.front();
  help += " -? 2>&1"; 
  system(help.c_str());
}

void lib::Parse(void) {
  archiver::Parse();
  if (!verbose) {
    archivearg.push_back("-nologo");
  }
}

void lib::FindInstallation(void) {
  tool::FindInstallation();
  string::size_type n = InstallDir.length()-1;
  VisualStudioDir = InstallDir.substr(0,n);
  n = VisualStudioDir.find_last_of("\\");
  VisualStudioDir = VisualStudioDir.substr(0,n+1);
  VSVersion = InstallDir.substr(0,InstallDir.length()-1);
  VSVersion = VSVersion.substr(VisualStudioDir.length());
}

void lib::AddPaths(void) {
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

/* $Id: petsccreate.cpp,v 1.2 2001/03/07 21:48:16 buschelm Exp $ */
#include <map>
#include <string>
#include <iostream>
#include "petscfe.h"
#include "petscclfe.h"
#include "petscbccfe.h"
#include "petsclibfe.h"
#include "petsctlibfe.h"

using namespace std;

namespace PETScFE {

  void CreateCL(tool *&Tool) {Tool = new cl;}
  void CreateDF(tool *&Tool) {Tool = new df;}
  void CreateBCC(tool *&Tool) {Tool = new bcc;}
  void CreateLIB(tool *&Tool) {Tool = new lib;}
  void CreateTLIB(tool *&Tool) {Tool = new tlib;}

  void tool::Create(tool *&Tool,string argv) {
    string KnownTools = "cl.icl.ifl.df.f90.bcc32.lib.tlib";
    map<string,void (*)(PETScFE::tool*&)> CreateTool;
    CreateTool["cl"] = PETScFE::CreateCL;
    CreateTool["icl"] = PETScFE::CreateCL;
    CreateTool["ifl"] = PETScFE::CreateCL;
    CreateTool["df"] = PETScFE::CreateDF;
    CreateTool["f90"] = PETScFE::CreateDF;
    CreateTool["bcc32"] = PETScFE::CreateBCC;
    CreateTool["lib"] = PETScFE::CreateLIB;
    CreateTool["tlib"] = PETScFE::CreateTLIB;
      
    if (KnownTools.find(argv)!=string::npos) {
      CreateTool[argv](Tool);
    } else {
      cout << "Unknown Tool" << endl;
      cout << "Error: 1" << endl;
      Tool = NULL;
    }
  }

};


/* $Id: petsccreate.cpp,v 1.11 2001/04/17 21:16:55 buschelm Exp $ */
#include "petscclfe.h"
#include "petsciclfe.h"
#include "petscbccfe.h"
#include "petsclibfe.h"
#include "petsctlibfe.h"

using namespace std;

namespace PETScFE {

  void CreateCL(  tool *&Tool) {Tool = new cl;}
  void CreateICL( tool *&Tool) {Tool = new icl;}
  void CreateIFL( tool *&Tool) {Tool = new ifl;}
  void CreateDF(  tool *&Tool) {Tool = new df;}
  void CreateBCC( tool *&Tool) {Tool = new bcc;}
  void CreateLIB( tool *&Tool) {Tool = new lib;}
  void CreateTLIB(tool *&Tool) {Tool = new tlib;}

  int tool::Create(tool *&Tool,string argv) {
    string KnownTools = "cl.icl.ifl.df.f90.bcc32.lib.tlib";
    map<string,void (*)(PETScFE::tool*&)> CreateTool;
    CreateTool["cl"] = PETScFE::CreateCL;
    CreateTool["icl"] = PETScFE::CreateICL;
    CreateTool["ifl"] = PETScFE::CreateIFL;
    CreateTool["df"] = PETScFE::CreateDF;
    CreateTool["f90"] = PETScFE::CreateDF;
    CreateTool["bcc32"] = PETScFE::CreateBCC;
    CreateTool["lib"] = PETScFE::CreateLIB;
    CreateTool["tlib"] = PETScFE::CreateTLIB;

    if (KnownTools.find(argv)==string::npos) {
      argv = "--help";
      Tool = new(tool);
    } else {
      CreateTool[argv](Tool);
    }
    (Tool->arg).push_front(argv);
    return(0);
  }

};


/* $Id: petscfe.cpp,v 1.7 2001/03/28 17:48:12 buschelm Exp $ */
#include <iostream>
#include "petsctoolfe.h"

using namespace std;

int main(int argc,char *argv[]) {
  PETScFE::tool *Tool;
  if (argc>1) {
    PETScFE::tool::Create(Tool,(string)(argv[1]));
  } else {
    PETScFE::tool::Create(Tool,"--help");
  }

  Tool->GetArgs(argc,argv);
  Tool->Parse();
  Tool->Execute();
    
  Tool->Destroy();
  return(0);
}

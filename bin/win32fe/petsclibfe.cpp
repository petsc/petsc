/* $Id:$ */
#include <fstream>
#include "petsclibfe.h"

using namespace PETScFE;

void lib::Execute(void) {
  {
    ifstream LibraryExists(file[0].c_str());
    string temp = file[0];
    if (file[0].substr(file[0].rfind("."))==".lib") {
      if (!LibraryExists) temp = "-out:" + file[0];
    } else {
      temp = "-out:" + file[0];
      if (LibraryExists) temp = temp + " " + file[0];
    }
    file[0] = temp;
  }
  archiver::Execute();
}

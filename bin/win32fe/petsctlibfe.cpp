/* $Id:$ */
#include "petsctlibfe.h"

using namespace PETScFE;

void tlib::FoundFile(int &loc,string temp) {
  ReplaceSlashWithBackslash(temp);
  file[loc] = "\"" + temp + "\"";
}

void tlib::FoundFlag(int &loc,string temp) {
  temp[0] = '/';
  archivearg[loc]=temp;
}

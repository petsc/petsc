
static char help[] = 
"Demonstrates how users may insert their own event logging.\n\n";

#include "petsc.h"
#include "plog.h"
#include <stdio.h>

#define USER_EVENT 75

int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,help);

  PLogEventRegister(USER_EVENT,"User event");
  PLogEventBegin(USER_EVENT,0,0,0,0);
  sleep(1);
  PLogEventEnd(USER_EVENT,0,0,0,0);
  PetscFinalize();
  return 0;
}
 

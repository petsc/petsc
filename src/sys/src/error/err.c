
#include "petsc.h"
#include <stdio.h>  /*I <stdio.h> I*/

/*@
    PetscErrorHandler - Handles error. Will eventually call a (possibly)
        user provided function.

  Input Parameters:
.  line,file - the linenumber and file the error was detected in
.  message - a text string usually just printed to the screen
.  number - the user provided error number.
@*/
int PetscErrorHandler(int line,char *file,char *message,int number)
{
  fprintf(stderr,"%s %d %s %d\n",file,line,message,number);
  return number;
}

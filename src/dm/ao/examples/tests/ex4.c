#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.1 1997/09/30 16:41:17 bsmith Exp bsmith $";
#endif

static char help[] = "Tests AOData loading\n\n";

#include "petsc.h"
#include "ao.h"
#include <math.h>

int main(int argc,char **argv)
{
  AOData      aodata;
  Viewer      binary;
  int         ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /*
        Load the database from the file
  */
  ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,"dataoutput",BINARY_RDONLY,&binary);CHKERRA(ierr);
  ierr = AODataLoadBasic(binary,&aodata);CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);
 
  ierr = AODataDestroy(aodata); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 



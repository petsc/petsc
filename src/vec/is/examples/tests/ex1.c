#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.9 1995/09/30 19:26:21 bsmith Exp bsmith $";
#endif

static char help[] = "Tests various IS routines\n\n";

#include "petsc.h"
#include "is.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int      n = 5, ierr,indices[5],mytid;
  IS       is;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create an index set */
  indices[0] = mytid + 1; 
  indices[1] = mytid + 2; 
  indices[2] = mytid + 3; 
  indices[3] = mytid + 4; 
  indices[4] = mytid + 5; 
  ierr = ISCreateSeq(MPI_COMM_SELF,n,indices,&is); CHKERRA(ierr);

  ierr = ISView(is,STDOUT_VIEWER_SELF); CHKERRA(ierr);
  ierr = ISDestroy(is); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 

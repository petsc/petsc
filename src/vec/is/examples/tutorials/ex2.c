
/*      "$Id: ex2.c,v 1.8 1999/03/19 21:17:34 bsmith Exp balay $"; */

static char help[] = "Demonstrates creating a stride index set.\n\n";

/*T
    Concepts: Index sets^Manipulating a stride index set;
    Routines: ISCreateStride(); ISDestroy(); ISView()
    Routines: ISGetIndices(); ISRestoreIndices(); ISStrideGetInfo()
    
    Comment: Creates an index set based on a stride. Views that index set
    and then destroys it.
T*/

/*
  Include is.h so we can use PETSc IS objects. Note that this automatically 
  includes petsc.h.
*/

#include "is.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int i, n, ierr,  *indices, first, step;
  IS  set;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);
      
  n     = 10;
  first = 3;
  step  = 2;

  /*
    Create stride index set, starting at 3 with a stride of 2
    Note each processor is generating its own index set 
    (in this case they are all identical)
  */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,first,step,&set);CHKERRA(ierr);
  ierr = ISView(set,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  /*
    Extract indices from set.
  */
  ierr = ISGetIndices(set,&indices);CHKERRA(ierr);
  printf("Printing indices directly\n");
  for (i=0; i<n; i++) {
    printf("%d\n",indices[i]);
  }

  ierr = ISRestoreIndices(set,&indices);CHKERRA(ierr);

  /*
      Determine information on stride
  */
  ierr = ISStrideGetInfo(set,&first,&step);CHKERRA(ierr);
  if (first != 3 || step != 2) SETERRA(1,0,"Stride info not correct!\n");
  ierr = ISDestroy(set);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}



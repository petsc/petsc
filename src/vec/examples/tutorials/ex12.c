/*$Id: ex12.c,v 1.6 1999/05/04 20:31:12 balay Exp bsmith $*/

/* Program usage:  mpirun ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates VecStrideScatter() and VecStrideGather().\n\n";

/*T
   Concepts: Vectors^Sub-vectors;
   Routines: VecCreate(); VecSetFromOptions(); VecSet(); VecSetBlockSize(); VecStrideScatter(), VecStrideGather(); 
   Processors: n
T*/

/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   is.h     - index sets
     sys.h    - system routines       viewer.h - viewers
*/

#include "vec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Vec      v,s;               /* vectors */
  int      n = 20, ierr, flg;
  Scalar   one = 1.0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);

  /* 
      Create multi-component vector with 2 components
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&v);CHKERRA(ierr);
  ierr = VecSetFromOptions(v);CHKERRA(ierr);
  ierr = VecSetBlockSize(v,2);CHKERRA(ierr);

  /* 
      Create single-component vector
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n/2,&s);CHKERRA(ierr);
  ierr = VecSetFromOptions(s);CHKERRA(ierr);

  /*
     Set the vectors to entries to a constant value.
  */
  ierr = VecSet(&one,v);CHKERRA(ierr);

  /*
     Get the first component from the multi-component vector to the single vector
  */
  ierr = VecStrideGather(v,0,s,INSERT_VALUES);CHKERRA(ierr);

  ierr = VecView(s,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /*
     Put the values back into the second component 
  */
  ierr = VecStrideScatter(s,1,v,ADD_VALUES);CHKERRA(ierr);

  ierr = VecView(v,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(v);CHKERRA(ierr);
  ierr = VecDestroy(s);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 

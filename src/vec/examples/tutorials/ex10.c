#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex10.c,v 1.5 1999/03/19 21:18:23 bsmith Exp balay $";
#endif

/* Program usage:  mpirun ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates the AMS Memory Snooper viewing.\n\n";

/*T
   Concepts: Vectors^Using basic vector routines;
   Routines: VecCreate(); VecSetFromOptions(); VecDuplicate(); VecSet(); VecValid(); 
   Routines: VecDot(); VecMDot(); VecScale(); VecNorm(); VecCopy(); VecAXPY(); 
   Routines: VecAYPX(); VecWAXPY(); VecPointwiseMult(); VecPointwiseDivide(); 
   Routines: VecSwap(); VecMAXPY(); VecDestroy(); VecDestroyVecs(); VecDuplicateVecs();
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
  Vec      x, y;
  int      n = 20, ierr, flg,i,row;
  Scalar   value;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* 
     Create a vector, specifying only its global dimension.
     When using VecCreate() and VecSetFromOptions(), the vector format (currently parallel,
     shared, or sequential) is determined at runtime.  Also, the parallel
     partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
        VecCreateShared() - parallel vector that uses shared memory
                            (available only on the SGI); otherwise,
                            is the same as VecCreateMPI()

     With VecCreate() and VecSetFromOptions() the option -vec_type mpi or -vec_type shared causes the 
     particular type of vector to be formed.

  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x); CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);

  /*
     Duplicate some work vector (of the same format and
     partitioning as the initial vector).
  */
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);

  PetscObjectPublish((PetscObject) x);CHKERRA(ierr);

  for ( i=0; i<1000; i++ ) {

    /*
       Set the vectors to entries to a constant value.
    */
    value = 1;
    row   = i % n;
    ierr = VecSetValues(x,1,&row,&value,ADD_VALUES); CHKERRA(ierr);
    ierr = VecAssemblyBegin(x); CHKERRA(ierr);
    ierr = VecAssemblyEnd(x); CHKERRA(ierr);


    ierr = PetscSleep(5);
  }


  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 

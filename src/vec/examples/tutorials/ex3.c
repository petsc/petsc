/*$Id: ex3.c,v 1.43 2000/08/17 04:51:19 bsmith Exp bsmith $*/

static char help[] = "Parallel vector layout.\n\n";

/*T
   Concepts: Vectors^setting values
   Concepts: Vectors^local access to
   Concepts: Vectors^drawing vectors;
   Processors: n
T*/

/* 
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscis.h     - index sets
     petscsys.h    - system routines       petscviewer.h - viewers
*/
#include "petscvec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        i,istart,iend,n = 6,ierr,rank,nlocal;
  Scalar     v,*array;
  Vec        x;
  Viewer     viewer;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  
  /* 
     Create a vector, specifying only its global dimension.
     When using VecCreate() and VecSetFromOptions(), the vector format (currently parallel
     or sequential) is determined at runtime.  Also, the parallel
     partitioning of the vector is determined by PETSc at runtime.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);

  /* 
     PETSc parallel vectors are partitioned by
     contiguous chunks of rows across the processors.  Determine
     which vector are locally owned. 
  */
  ierr = VecGetOwnershipRange(x,&istart,&iend);CHKERRA(ierr);

  /* -------------------------------------------------------------------- 
     Set the vector elements.
      - Always specify global locations of vector entries.
      - Each processor can insert into any location, even ones it does not own
      - In this case each processor adds values to all the entries,
         this is not practical, but is merely done as an example
   */
  for (i=0; i<n; i++) { 
    v = (double)(rank*i);
    ierr = VecSetValues(x,1,&i,&v,ADD_VALUES);CHKERRA(ierr);
  }

  /* 
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  /*
     Open an X-window viewer.  Note that we specify the same communicator
     for the viewer as we used for the distributed vector (PETSC_COMM_WORLD).
       - Helpful runtime option:
            -draw_pause <pause> : sets time (in seconds) that the
                  program pauses after DrawPause() has been called
                  (0 is default, -1 implies until user input).

  */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL,PETSC_NULL,0,0,300,300,&viewer);CHKERRA(ierr);
  ierr = ViewerPushFormat(viewer,VIEWER_FORMAT_DRAW_LG,"Line graph Plot");CHKERRA(ierr);
  /*
     View the vector
  */
  ierr = VecView(x,viewer);CHKERRA(ierr);

  /* --------------------------------------------------------------------
       Access the vector values directly. Each processor has access only 
    to its portion of the vector. For default PETSc vectors VecGetArray()
    does NOT involve a copy
  */
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);
  ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  for ( i=0; i<nlocal; i++) {
    array[i] = rank + 1;
  }
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);

  /*
     View the vector
  */
  ierr = VecView(x,viewer);CHKERRA(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 

#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.35 1999/01/12 23:13:48 bsmith Exp bsmith $";
#endif

static char help[] = "Displays a vector visually.\n\n";

/*T
   Concepts: Vectors^Drawing vectors;
   Routines: VecCreate(); VecSetFromOptions(); VecSetValues(); VecView(); VecDestroy(); 
   Routines: VecAssemblyBegin(); VecAssemblyEnd(); VecGetOwnershipRange();
   Routines: ViewerDrawOpen(); ViewerDestroy();
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
  int        i, istart, iend, n = 50, ierr, flg;
  Scalar     v;
  Vec        x;
  Viewer     viewer;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  /* 
     Create a vector, specifying only its global dimension.
     When using VecCreate() and VecSetFromOptions(), the vector format (currently parallel
     or sequential) is determined at runtime.  Also, the parallel
     partitioning of the vector is determined by PETSc at runtime.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x); CHKERRA(ierr);
  ierr = VecSetFromOptions(x); CHKERRA(ierr);

  /* 
     Currently, all PETSc parallel vectors are partitioned by
     contiguous chunks of rows across the processors.  Determine
     which vector are locally owned. 
  */
  ierr = VecGetOwnershipRange(x,&istart,&iend); CHKERRA(ierr);

  /* 
     Set the vector elements.
      - Always specify global locations of vector entries.
      - Each processor needs to insert only elements that it owns locally.
   */
  for ( i=istart; i<iend; i++ ) { 
    v = (double) i;
    ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
  }

  /* 
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  /*
     Open an X-window viewer.  Note that we specify the same communicator
     for the viewer as we used for the distributed vector (PETSC_COMM_WORLD).
       - Helpful runtime option:
            -draw_pause <pause> : sets time (in seconds) that the
                  program pauses after DrawPause() has been called
                  (0 is default, -1 implies until user input).

  */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL,PETSC_NULL,0,0,300,300,
                         &viewer); CHKERRA(ierr);

  /*
     View the vector
  */
  ierr = VecView(x,viewer); CHKERRA(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 

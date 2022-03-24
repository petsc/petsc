
static char help[] = "Parallel vector layout.\n\n";

/*T
   Concepts: vectors^setting values
   Concepts: vectors^local access to
   Concepts: vectors^drawing vectors;
   Processors: n
T*/

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscInt       i,istart,iend,n = 6,nlocal;
  PetscScalar    v,*array;
  Vec            x;
  PetscViewer    viewer;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /*
     Create a vector, specifying only its global dimension.
     When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
     the vector format (currently parallel or sequential) is
     determined at runtime.  Also, the parallel partitioning of
     the vector is determined by PETSc at runtime.
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));

  /*
     PETSc parallel vectors are partitioned by
     contiguous chunks of rows across the processors.  Determine
     which vector are locally owned.
  */
  CHKERRQ(VecGetOwnershipRange(x,&istart,&iend));

  /* --------------------------------------------------------------------
     Set the vector elements.
      - Always specify global locations of vector entries.
      - Each processor can insert into any location, even ones it does not own
      - In this case each processor adds values to all the entries,
         this is not practical, but is merely done as an example
   */
  for (i=0; i<n; i++) {
    v    = (PetscReal)(rank*i);
    CHKERRQ(VecSetValues(x,1,&i,&v,ADD_VALUES));
  }

  /*
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /*
     Open an X-window viewer.  Note that we specify the same communicator
     for the viewer as we used for the distributed vector (PETSC_COMM_WORLD).
       - Helpful runtime option:
            -draw_pause <pause> : sets time (in seconds) that the
                  program pauses after PetscDrawPause() has been called
                  (0 is default, -1 implies until user input).

  */
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,0,0,300,300,&viewer));
  CHKERRQ(PetscObjectSetName((PetscObject)viewer,"Line graph Plot"));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG));
  /*
     View the vector
  */
  CHKERRQ(VecView(x,viewer));

  /* --------------------------------------------------------------------
       Access the vector values directly. Each processor has access only
    to its portion of the vector. For default PETSc vectors VecGetArray()
    does NOT involve a copy
  */
  CHKERRQ(VecGetLocalSize(x,&nlocal));
  CHKERRQ(VecGetArray(x,&array));
  for (i=0; i<nlocal; i++) array[i] = rank + 1;
  CHKERRQ(VecRestoreArray(x,&array));

  /*
     View the vector
  */
  CHKERRQ(VecView(x,viewer));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&x));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

TEST*/

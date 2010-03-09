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
#include "petscvec.h"
#include "stdlib.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,istart,iend,n = 6,m,*indices;
  PetscScalar    *values;
  Vec            x;
  PetscTruth     set_option_negidx = PETSC_FALSE, set_values_negidx = PETSC_FALSE, get_values_negidx = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-set_option_negidx", &set_option_negidx, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-set_values_negidx", &set_values_negidx, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-get_values_negidx", &get_values_negidx, PETSC_NULL);CHKERRQ(ierr);
  
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /* If we want to use negative indices, set the option */
  ierr = VecSetOption(x, VEC_IGNORE_NEGATIVE_INDICES,set_option_negidx); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&istart,&iend);CHKERRQ(ierr);
  m = iend - istart;


  /* Set the vectors */
  
  ierr = PetscMalloc(n*sizeof(PetscScalar),&values);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&indices);CHKERRQ(ierr);

  for (i=istart; i<iend; i++) {
    values[i - istart]  = (rank + 1) * i * 2;
    if (set_values_negidx) {
        indices[i - istart] = (-1 + 2*(i % 2)) * i;
    }
    else {
        indices[i - istart] = i;
    }
  }

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: Setting values...\n", rank); CHKERRQ(ierr);
  for (i = 0; i<m; i++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, 
				   "%d: idx[%D] == %D; val[%D] == %f\n", 
				   rank, i, indices[i], i, PetscRealPart(values[i]));CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = VecSetValues(x, m, indices, values, INSERT_VALUES);CHKERRQ(ierr);

  /* 
     Assemble vector.
  */
  
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /*
     Extract values from the vector.
  */
  
  for (i=0; i<m; i++) {
    values[i] = -1.0;
    if (get_values_negidx) {
      indices[i] = (-1 + 2*((istart+i) % 2)) * (istart+i);
    }
    else {
        indices[i] = istart+i;
    }
  }

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: Fetching these values from vector...\n", rank);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: idx[%D] == %D\n", rank, i, indices[i]);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = VecGetValues(x, m, indices, values);CHKERRQ(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: Fetched values:\n", rank);CHKERRQ(ierr);
  for (i = 0; i<m; i++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: idx[%D] == %D; val[%D] == %f\n", 
				   rank, i, indices[i], i, PetscRealPart(values[i]));CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  /* 
     Free work space.
  */

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
 

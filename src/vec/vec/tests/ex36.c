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
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,istart,iend,n = 6,m,*indices;
  PetscScalar    *values;
  Vec            x;
  PetscBool      set_option_negidx = PETSC_FALSE, set_values_negidx = PETSC_FALSE, get_values_negidx = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL, "-set_option_negidx", &set_option_negidx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL, "-set_values_negidx", &set_values_negidx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL, "-get_values_negidx", &get_values_negidx, NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /* If we want to use negative indices, set the option */
  ierr = VecSetOption(x, VEC_IGNORE_NEGATIVE_INDICES,set_option_negidx);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&istart,&iend);CHKERRQ(ierr);
  m    = iend - istart;

  ierr = PetscMalloc1(n,&values);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&indices);CHKERRQ(ierr);

  for (i=istart; i<iend; i++) {
    values[i - istart] = (rank + 1) * i * 2;
    if (set_values_negidx) indices[i - istart] = (-1 + 2*(i % 2)) * i;
    else                   indices[i - istart] = i;
  }

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: Setting values...\n", rank);CHKERRQ(ierr);
  for (i = 0; i<m; i++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%d: idx[%" PetscInt_FMT "] == %" PetscInt_FMT "; val[%" PetscInt_FMT "] == %f\n",rank,i,indices[i],i,(double)PetscRealPart(values[i]));CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

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
    if (get_values_negidx) indices[i] = (-1 + 2*((istart+i) % 2)) * (istart+i);
    else                   indices[i] = istart+i;
  }

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: Fetching these values from vector...\n", rank);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: idx[%" PetscInt_FMT "] == %" PetscInt_FMT "\n", rank, i, indices[i]);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = VecGetValues(x, m, indices, values);CHKERRQ(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: Fetched values:\n", rank);CHKERRQ(ierr);
  for (i = 0; i<m; i++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d: idx[%" PetscInt_FMT "] == %" PetscInt_FMT "; val[%" PetscInt_FMT "] == %f\n",rank,i,indices[i],i,(double)PetscRealPart(values[i]));CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  /*
     Free work space.
  */

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -set_option_negidx -set_values_negidx -get_values_negidx

TEST*/

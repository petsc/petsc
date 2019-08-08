static const char help[] = "Test VEC_SUBSET_OFF_PROC_ENTRIES\n\n";

#include <petsc.h>
#include <petscvec.h>

/* Unlike most finite element applications, IBAMR does assembly on many cells
   that are not locally owned; in some cases the processor may own zero finite
   element cells but still do assembly on a small number of cells anyway. To
   simulate this, this code assembles a PETSc vector by adding contributions
   to every entry in the vector on every processor. This causes a deadlock
   when we save the communication pattern via

     VecSetOption(vec, VEC_SUBSET_OFF_PROC_ENTRIES, PETSC_TRUE).

   Contributed-by: David Wells <drwells@email.unc.edu>

  Petsc developers' notes: this test tests how Petsc knows it can reuse existing communication
  pattern. All processes must come to the same conclusion, otherwise deadlock may happen due
  to mismatched MPI_Send/Recv. It also tests changing VEC_SUBSET_OFF_PROC_ENTRIES back and forth.
*/
int main(int argc, char **argv)
{
  Vec            v;
  PetscInt       i, j, k, *ln, n, rstart;
  PetscBool      saveCommunicationPattern = PETSC_FALSE;
  PetscMPIInt    size, rank, p;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL, NULL, "-save_comm", &saveCommunicationPattern, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  ierr = PetscMalloc1(size, &ln);CHKERRQ(ierr);
  /* This bug is triggered when one of the local lengths is small. Sometimes in IBAMR this value is actually zero. */
  for (p=0; p<size; ++p) ln[p] = 10;
  ln[0] = 2;
  ierr  = PetscPrintf(PETSC_COMM_WORLD, "local lengths are:\n");CHKERRQ(ierr);
  ierr  = PetscIntView(1, &ln[rank], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  n     = ln[rank];
  ierr  = VecCreateMPI(MPI_COMM_WORLD, n, PETSC_DECIDE, &v);CHKERRQ(ierr);
  ierr  = VecGetOwnershipRange(v, &rstart, NULL);CHKERRQ(ierr);

  for (k=0; k<5; ++k) { /* 5 iterations of VecAssembly */
    PetscReal norm = 0.0;
    PetscBool flag  = (k == 2) ?  PETSC_FALSE : PETSC_TRUE;
    PetscInt  shift = (k < 2) ? 0 : (k == 2) ? 1 : 0; /* Used to change patterns */

    /* If saveCommunicationPattern, let's see what should happen in the 5 iterations:
      iter 0: flag is true, and this is the first assebmly, so petsc should keep the
              communication pattern built during this assembly.
      iter 1: flag is true, reuse the pattern.
      iter 2: flag is false, discard/free the pattern built in iter 0; rebuild a new
              pattern, but do not keep it after VecAssemblyEnd since the flag is false.
      iter 3: flag is true again, this is the new first assembly with a true flag. So
              petsc should keep the communication pattern built during this assembly.
      iter 4: flag is true, reuse the pattern built in iter 3.

      When the vector is destroyed, memory used by the pattern is freed. One can also do it early with a call
          VecSetOption(v, VEC_SUBSET_OFF_PROC_ENTRIES, PETSC_FALSE);
     */
    if (saveCommunicationPattern) {ierr = VecSetOption(v, VEC_SUBSET_OFF_PROC_ENTRIES, flag);CHKERRQ(ierr);}
    ierr = VecSet(v, 0.0);CHKERRQ(ierr);

    for (i=0; i<n; ++i) {
      PetscScalar val = 1.0;
      PetscInt    r   = rstart + i;

      ierr = VecSetValue(v, r, val, ADD_VALUES);CHKERRQ(ierr);
      /* do assembly on all other processors too (the 'neighbors') */
      {
        const PetscMPIInt neighbor = (i+shift) % size; /* Adjust communication patterns between iterations */
        const PetscInt    nn       = ln[neighbor];
        PetscInt          nrstart  = 0;

        for (p=0; p<neighbor; ++p) nrstart += ln[p];
        for (j=0; j<nn/4; j+= 3) {
          PetscScalar val = 0.01;
          PetscInt    nr  = nrstart + j;

          ierr = VecSetValue(v, nr, val, ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
    ierr = VecNorm(v, NORM_1, &norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "norm is %g\n", (double)norm);CHKERRQ(ierr);
  }
  ierr = PetscFree(ln);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 4
   test:
      suffix: 1_save
      args: -save_comm
      nsize: 4
      output_file: output/ex49_1.out
TEST*/

static char help[] = "Tests for norm caching\n";

#include <petscvec.h>
#include <petsc/private/petscimpl.h>  /* to gain access to the private PetscObjectStateIncrease() */

int main(int argc,char **argv)
{
  Vec            V,W;
  MPI_Comm       comm;
  PetscScalar    one=1,e=2.7181;
  PetscReal      nrm1,nrm2,nrm3,nrm4;
  PetscInt       ione=1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  comm = MPI_COMM_SELF;

  CHKERRQ(VecCreate(comm,&V));
  CHKERRQ(VecSetSizes(V,10,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(V));
  CHKERRQ(VecSetRandom(V,NULL));
  CHKERRQ(VecAssemblyBegin(V));
  CHKERRQ(VecAssemblyEnd(V));

  /*
   * Initial
   */
  /* display norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"Original: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /* display cached norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"cached: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /*
   * Alter an element
   */
  CHKERRQ(VecSetValues(V,1,&ione,&one,INSERT_VALUES));

  /* display norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"Altered: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /* display cached norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"recomputed: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /*
   * Scale the vector a little
   */
  CHKERRQ(VecScale(V,e));

  /* display updated cached norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"Scale: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /* display forced norm 1 & 2 */
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"recompute: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /*
   * Normalize the vector a little
   */
  CHKERRQ(VecNormalize(V,&nrm1));

  /* display updated cached norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"Normalize: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /* display forced norm 1 & 2 */
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"recompute: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /*
   * Copy to another vector
   */
  CHKERRQ(VecDuplicate(V,&W));
  CHKERRQ(VecCopy(V,W));

  /* display norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"Original: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /* display cached norm 1 & 2 */
  CHKERRQ(VecNorm(W,NORM_1,&nrm1));
  CHKERRQ(VecNorm(W,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"copied: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /*
   * Copy while data is invalid
   */
  CHKERRQ(VecSetValues(V,1,&ione,&one,INSERT_VALUES));
  CHKERRQ(VecCopy(V,W));

  /* display norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"Invalidated: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /* display norm 1 & 2 */
  CHKERRQ(VecNorm(W,NORM_1,&nrm1));
  CHKERRQ(VecNorm(W,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"copied: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /*
   * Constant vector
   */
  CHKERRQ(VecSet(V,e));

  /* display updated cached norm 1 & 2 */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"Constant: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /* display forced norm 1 & 2 */
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(V,NORM_2,&nrm2));
  CHKERRQ(PetscPrintf(comm,"recomputed: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2));

  /*
   * Swap vectors
   */
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(W,NORM_1,&nrm2));
  CHKERRQ(PetscPrintf(comm,"Orig: norm_V=%e,norm_W=%e\n",(double)nrm1,(double)nrm2));
  /* store inf norm */
  CHKERRQ(VecNorm(V,NORM_INFINITY,&nrm3));
  CHKERRQ(VecNorm(W,NORM_INFINITY,&nrm4));

  CHKERRQ(VecSwap(V,W));

  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  CHKERRQ(PetscObjectStateIncrease((PetscObject)W));
  CHKERRQ(VecNorm(V,NORM_1,&nrm1));
  CHKERRQ(VecNorm(W,NORM_1,&nrm2));
  CHKERRQ(PetscPrintf(comm,"swapped: norm_V=%e,norm_W=%e\n",(double)nrm2,(double)nrm1));
  CHKERRQ(PetscPrintf(comm,"orig: F-norm_V=%e,F-norm_W=%e\n",(double)nrm3,(double)nrm4));
  CHKERRQ(VecNorm(V,NORM_INFINITY,&nrm3));
  CHKERRQ(VecNorm(W,NORM_INFINITY,&nrm4));
  CHKERRQ(PetscPrintf(comm,"swapped: F-norm_V=%e,F-norm_W=%e\n",(double)nrm4,(double)nrm3));

  CHKERRQ(VecDestroy(&V));
  CHKERRQ(VecDestroy(&W));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/ex34_1.out
      test:
        suffix: standard
      test:
        requires: cuda
        args: -vec_type cuda
        suffix: cuda
      test:
        requires: viennacl
        args: -vec_type viennacl
        suffix: viennacl
      test:
        requires: kokkos_kernels
        args: -vec_type kokkos
        suffix: kokkos
      test:
        requires: hip
        args: -vec_type hip
        suffix: hip

TEST*/

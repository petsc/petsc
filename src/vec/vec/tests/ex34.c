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

  ierr = VecCreate(comm,&V);CHKERRQ(ierr);
  ierr = VecSetSizes(V,10,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(V);CHKERRQ(ierr);
  ierr = VecSetRandom(V,NULL);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(V);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(V);CHKERRQ(ierr);

  /*
   * Initial
   */
  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Original: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"cached: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /*
   * Alter an element
   */
  ierr = VecSetValues(V,1,&ione,&one,INSERT_VALUES);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Altered: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"recomputed: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /*
   * Scale the vector a little
   */
  ierr = VecScale(V,e);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Scale: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"recompute: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /*
   * Normalize the vector a little
   */
  ierr = VecNormalize(V,&nrm1);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Normalize: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"recompute: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /*
   * Copy to another vector
   */
  ierr = VecDuplicate(V,&W);CHKERRQ(ierr);
  ierr = VecCopy(V,W);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Original: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(W,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"copied: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /*
   * Copy while data is invalid
   */
  ierr = VecSetValues(V,1,&ione,&one,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecCopy(V,W);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Invalidated: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(W,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"copied: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /*
   * Constant vector
   */
  ierr = VecSet(V,e);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Constant: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"recomputed: norm1=%e,norm2=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);

  /*
   * Swap vectors
   */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_1,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Orig: norm_V=%e,norm_W=%e\n",(double)nrm1,(double)nrm2);CHKERRQ(ierr);
  /* store inf norm */
  ierr = VecNorm(V,NORM_INFINITY,&nrm3);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_INFINITY,&nrm4);CHKERRQ(ierr);

  ierr = VecSwap(V,W);CHKERRQ(ierr);

  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)W);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_1,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"swapped: norm_V=%e,norm_W=%e\n",(double)nrm2,(double)nrm1);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"orig: F-norm_V=%e,F-norm_W=%e\n",(double)nrm3,(double)nrm4);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_INFINITY,&nrm3);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_INFINITY,&nrm4);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"swapped: F-norm_V=%e,F-norm_W=%e\n",(double)nrm4,(double)nrm3);CHKERRQ(ierr);

  ierr = VecDestroy(&V);CHKERRQ(ierr);
  ierr = VecDestroy(&W);CHKERRQ(ierr);
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

TEST*/

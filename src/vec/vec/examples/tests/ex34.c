static char help[] = "Tests for norm caching\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec            V,W;
  PetscReal      nrm1,nrm2,nrm3,nrm4;
  MPI_Comm       comm;
  PetscScalar    one=1,e=2.7181;
  PetscInt       ione=1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = MPI_COMM_SELF;

  ierr = VecCreateSeq(comm,10,&V);CHKERRQ(ierr);
  ierr = VecSetRandom(V,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(V);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(V);CHKERRQ(ierr);

  /*
   * Initial
   */
  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Original: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"cached: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /*
   * Alter an element
   */
  ierr = VecSetValues(V,1,&ione,&one,INSERT_VALUES);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Altered: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"recomputed: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /*
   * Scale the vector a little
   */
  ierr = VecScale(V,e);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Scale: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"recompute: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /*
   * Normalize the vector a little
   */
  ierr = VecNormalize(V,&nrm1);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Normalize: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"recompute: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /*
   * Copy to another vector
   */
  ierr = VecDuplicate(V,&W);CHKERRQ(ierr);
  ierr = VecCopy(V,W);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Original: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(W,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"copied: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /*
   * Copy while data is invalid
   */
  ierr = VecSetValues(V,1,&ione,&one,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecCopy(V,W);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Invalidated: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(W,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"copied: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /*
   * Constant vector
   */
  ierr = VecSet(V,e);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Constant: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"recomputed: norm1=%e, norm2=%e\n",nrm1,nrm2);CHKERRQ(ierr);

  /*
   * Swap vectors
   */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_1,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Orig: norm V=%e, norm W=%e\n",nrm1,nrm2);CHKERRQ(ierr);
  /* store inf norm */
  ierr = VecNorm(V,NORM_INFINITY,&nrm3);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_INFINITY,&nrm4);CHKERRQ(ierr);

  ierr = VecSwap(V,W);CHKERRQ(ierr);

  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)W);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_1,&nrm2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"swapped: norm V=%e, norm W=%e\n",nrm2,nrm1);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"orig: F norm V=%e, F norm W=%e\n",nrm3,nrm4);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_INFINITY,&nrm3);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_INFINITY,&nrm4);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"swapped: F norm V=%e, F norm W=%e\n",nrm4,nrm3);CHKERRQ(ierr);

  ierr = VecDestroy(&V);CHKERRQ(ierr);
  ierr = VecDestroy(&W);CHKERRQ(ierr);

  PetscFinalize();
  PetscFunctionReturn(0);
}

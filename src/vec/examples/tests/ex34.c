/*
 * Test file for norm caching
 */

#include <stdlib.h>
#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec         V,W;
  PetscReal   nrm1,nrm2,nrm3,nrm4;
  MPI_Comm    comm;
  PetscScalar one=1,e=2.7181;
  int         ione=1, ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  comm = MPI_COMM_SELF;
  
  ierr = VecCreateSeq(comm,10,&V);CHKERRQ(ierr);
  ierr = VecSetRandom(PETSC_NULL,V);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(V);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(V);CHKERRQ(ierr);

  /*
   * Initial
   */
  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"Original: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"cached: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /* 
   * Alter an element
   */
  ierr = VecSetValues(V,1,&ione,&one,INSERT_VALUES);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"Altered: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"recomputed: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /*
   * Scale the vector a little
   */
  ierr = VecScale(&e,V);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"Scale: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectIncreaseState((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"recompute: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /*
   * Normalize the vector a little
   */
  ierr = VecNormalize(V,&nrm1);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"Normalize: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectIncreaseState((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"recompute: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /*
   * Copy to another vector
   */
  ierr = VecDuplicate(V,&W);CHKERRQ(ierr);
  ierr = VecCopy(V,W);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"Original: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /* display cached norm 1 & 2 */
  ierr = VecNorm(W,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"copied: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /*
   * Copy while data is invalid
   */
  ierr = VecSetValues(V,1,&ione,&one,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecCopy(V,W);CHKERRQ(ierr);

  /* display norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"Invalidated: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /* display norm 1 & 2 */
  ierr = VecNorm(W,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"copied: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /*
   * Constant vector
   */
  ierr = VecSet(&e,V);CHKERRQ(ierr);

  /* display updated cached norm 1 & 2 */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"Constant: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /* display forced norm 1 & 2 */
  ierr = PetscObjectIncreaseState((PetscObject)V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"recomputed: norm1=%e, norm2=%e\n",nrm1,nrm2);

  /*
   * Swap vectors
   */
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_1,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"Orig: norm V=%e, norm W=%e\n",nrm1,nrm2);
  /* store inf norm */
  ierr = VecNorm(V,NORM_INFINITY,&nrm3);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_INFINITY,&nrm4);CHKERRQ(ierr);

  ierr = VecSwap(V,W);CHKERRQ(ierr);

  ierr = PetscObjectIncreaseState((PetscObject)V);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)W);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_1,&nrm1);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_1,&nrm2);CHKERRQ(ierr);
  PetscPrintf(comm,"swapped: norm V=%e, norm W=%e\n",nrm2,nrm1);
  PetscPrintf(comm,"orig: F norm V=%e, F norm W=%e\n",nrm3,nrm4);
  ierr = VecNorm(V,NORM_INFINITY,&nrm3);CHKERRQ(ierr);
  ierr = VecNorm(W,NORM_INFINITY,&nrm4);CHKERRQ(ierr);
  PetscPrintf(comm,"swapped: F norm V=%e, F norm W=%e\n",nrm4,nrm3);

  ierr = VecDestroy(V);CHKERRQ(ierr);
  ierr = VecDestroy(W);CHKERRQ(ierr);

  PetscFinalize();
  PetscFunctionReturn(0);
}

/*
 * Test file for the PCILUSetShift routine or -pc_shift option.
 * The test matrix is the example from Kershaw's paper [J.Comp.Phys 1978]
 * of a positive definite matrix for which ILU(0) will give a negative pivot.
 * This means that the CG method will break down; the Manteuffel shift
 * repairs this.
 *
 * Contributed by Victor Eijkhout 2003.
 */

#include <stdlib.h>
#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec V; PetscReal nrm1,nrm2; MPI_Comm comm;
  PetscScalar one=1; int ione=1, ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,0,0); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-options_left",PETSC_NULL); CHKERRQ(ierr);
  comm = MPI_COMM_SELF;
  
  ierr = VecCreateSeq(comm,10,&V); CHKERRQ(ierr);
  ierr = VecSetRandom(PETSC_NULL,V); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(V); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(V); CHKERRQ(ierr);

  ierr = VecNorm(V,NORM_1,&nrm1); CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2); CHKERRQ(ierr);
  PetscPrintf(comm,"Vector created  : norm1=%e, norm2=%e\n",nrm1,nrm2);
  ierr = VecNorm(V,NORM_1,&nrm1); CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2); CHKERRQ(ierr);
  PetscPrintf(comm,"  reused values : norm1=%e, norm2=%e\n",nrm1,nrm2);

  ierr = VecSetValues(V,1,&ione,&one,INSERT_VALUES); CHKERRQ(ierr);

  ierr = VecNorm(V,NORM_1,&nrm1); CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2); CHKERRQ(ierr);
  PetscPrintf(comm,"Vector altered  : norm1=%e, norm2=%e\n",nrm1,nrm2);
  ierr = VecNorm(V,NORM_1,&nrm1); CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&nrm2); CHKERRQ(ierr);
  PetscPrintf(comm,"  reused values : norm1=%e, norm2=%e\n",nrm1,nrm2);

  PetscFinalize();
  PetscFunctionReturn(0);
}


static char help[] = "Tests DA interpolation for coarse DA on a subset of processors.\n\n";

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M = 14,dof = 1,s = 1,ratio = 2,dim = 2;
  PetscErrorCode ierr;
  DA             da_c,da_f;
  Vec            v_c,v_f;
  Mat            I;
  PetscScalar    one = 1.0;
  MPI_Comm       comm_f, comm_c;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscOptionsGetInt(PETSC_NULL,"-dim",&dim,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-sw",&s,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ratio",&ratio,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr);

  comm_f = PETSC_COMM_WORLD;
  ierr = DASplitComm2d(comm_f,M,M,s,&comm_c);CHKERRQ(ierr);
    
  /* Set up the array */ 
  if (dim == 2) {
    ierr = DACreate2d(comm_c,DA_NONPERIODIC,DA_STENCIL_BOX,M,M,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL,PETSC_NULL,&da_c);CHKERRQ(ierr);
    M    = ratio*(M-1) + 1;
    ierr = DACreate2d(comm_f,DA_NONPERIODIC,DA_STENCIL_BOX,M,M,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL,PETSC_NULL,&da_f);CHKERRQ(ierr);
  } else if (dim == 3) {
    ;
  }

  ierr = DACreateGlobalVector(da_c,&v_c);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da_f,&v_f);CHKERRQ(ierr);

  ierr = VecSet(v_c,one);CHKERRQ(ierr);
  ierr = DAGetInterpolation(da_c,da_f,&I,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatInterpolate(I,v_c,v_f);CHKERRQ(ierr);
  ierr = VecView(v_f,PETSC_VIEWER_STDOUT_(comm_f));CHKERRQ(ierr);
  ierr = MatRestrict(I,v_f,v_c);CHKERRQ(ierr);
  ierr = VecView(v_c,PETSC_VIEWER_STDOUT_(comm_c));CHKERRQ(ierr);

  ierr = MatDestroy(I);CHKERRQ(ierr);
  ierr = VecDestroy(v_c);CHKERRQ(ierr);
  ierr = DADestroy(da_c);CHKERRQ(ierr);
  ierr = VecDestroy(v_f);CHKERRQ(ierr);
  ierr = DADestroy(da_f);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 





static char help[] = "Tests DMDA interpolation for coarse DM on a subset of processors.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt       M = 14,dof = 1,s = 1,ratio = 2,dim = 2;
  PetscErrorCode ierr;
  DM             da_c,da_f;
  Vec            v_c,v_f;
  Mat            I;
  PetscScalar    one = 1.0;
  MPI_Comm       comm_f, comm_c;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-sw",&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ratio",&ratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);

  comm_f = PETSC_COMM_WORLD;
  ierr   = DMDASplitComm2d(comm_f,M,M,s,&comm_c);CHKERRQ(ierr);

  /* Set up the array */
  if (dim == 2) {
    ierr = DMDACreate2d(comm_c,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,M,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da_c);CHKERRQ(ierr);
    M    = ratio*(M-1) + 1;
    ierr = DMDACreate2d(comm_f,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,M,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da_f);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(da_c);CHKERRQ(ierr);
  ierr = DMSetUp(da_c);CHKERRQ(ierr)
  ierr = DMSetFromOptions(da_f);CHKERRQ(ierr);
  ierr = DMSetUp(da_f);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da_c,&v_c);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da_f,&v_f);CHKERRQ(ierr);

  ierr = VecSet(v_c,one);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(da_c,da_f,&I,NULL);CHKERRQ(ierr);
  ierr = MatInterpolate(I,v_c,v_f);CHKERRQ(ierr);
  ierr = VecView(v_f,PETSC_VIEWER_STDOUT_(comm_f));CHKERRQ(ierr);
  ierr = MatRestrict(I,v_f,v_c);CHKERRQ(ierr);
  ierr = VecView(v_c,PETSC_VIEWER_STDOUT_(comm_c));CHKERRQ(ierr);

  ierr = MatDestroy(&I);CHKERRQ(ierr);
  ierr = VecDestroy(&v_c);CHKERRQ(ierr);
  ierr = DMDestroy(&da_c);CHKERRQ(ierr);
  ierr = VecDestroy(&v_f);CHKERRQ(ierr);
  ierr = DMDestroy(&da_f);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}





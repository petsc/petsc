/*$Id: ex1.c,v 1.19 2001/08/10 03:34:58 bsmith Exp $*/

static char help[] = "Tests VecView()/VecLoad() for DA vectors (this tests DAGlobalToNatural()).\n\n";

#include "petscda.h"
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int            size,M=8,ierr,dof=1,stencil_width=1,P=5;
  int            N = 6,m=PETSC_DECIDE,n=PETSC_DECIDE,p=PETSC_DECIDE;
  PetscTruth     flg2,flg3;
  DAPeriodicType periodic = DA_NONPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_STAR;
  DA             da;
  Vec            global1,global2;
  PetscScalar    mone = -1.0;
  PetscReal      norm;
  PetscViewer    viewer;
  PetscRandom    rand;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-stencil_width",&stencil_width,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-periodic",(int*)&periodic,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-stencil_type",(int*)&stencil_type,PETSC_NULL);CHKERRQ(ierr); 

  ierr = PetscOptionsHasName(PETSC_NULL,"-1d",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-2d",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-3d",&flg3);CHKERRQ(ierr);
  if (flg2) {
    ierr = DACreate2d(PETSC_COMM_WORLD,periodic,stencil_type,M,N,m,n,dof,stencil_width,0,0,&da);CHKERRQ(ierr);
  } else if (flg3) {
    ierr = DACreate3d(PETSC_COMM_WORLD,periodic,stencil_type,M,N,P,m,n,p,dof,stencil_width,0,0,0,&da);CHKERRQ(ierr);
  }
  else {
    ierr = DACreate1d(PETSC_COMM_WORLD,periodic,M,dof,stencil_width,PETSC_NULL,&da);CHKERRQ(ierr);
  }

  ierr = DACreateGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rand);CHKERRQ(ierr);
  ierr = VecSetRandom(rand,global1);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);

  ierr = DACreateGlobalVector(da,&global2);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",PETSC_FILE_CREATE,&viewer);CHKERRQ(ierr);
  ierr = VecView(global1,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",PETSC_FILE_RDONLY,&viewer);CHKERRQ(ierr);
  ierr = VecLoadIntoVector(viewer,global2);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  ierr = VecAXPY(&mone,global1,global2);CHKERRQ(ierr);
  ierr = VecNorm(global2,NORM_MAX,&norm);CHKERRQ(ierr);
  if (norm != 0.0) {
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of difference %g should be zero\n",norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %d %d %d %d\n",M,N,P,dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %d stencil_type %d periodic %d\n",stencil_width,stencil_type,periodic);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3);CHKERRQ(ierr);
  }
   
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = VecDestroy(global1);CHKERRQ(ierr);
  ierr = VecDestroy(global2);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

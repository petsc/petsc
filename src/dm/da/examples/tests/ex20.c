
static char help[] = "Tests SDALocalToLocalxxx().\n\n";

#include "petscda.h"

/*
         For testing purposes this example also creates a 
   DA context. Actually codes using SDA routines will probably 
   not also work with DA contexts.
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscInt       M=8,dof=1,stencil_width=1,i,start,end,P=5,N = 6,m=PETSC_DECIDE,n=PETSC_DECIDE,p=PETSC_DECIDE,pt = 0,st = 0;
  PetscErrorCode ierr;
  PetscTruth     flg2,flg3,flg;
  DAPeriodicType periodic = DA_NONPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_STAR;
  DA             da;
  SDA            sda;
  Vec            local,global,local_copy;
  PetscScalar    value,*in,*out;
  PetscReal      norm,work;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  FILE           *file;


  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-stencil_width",&stencil_width,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-periodic",&pt,PETSC_NULL);CHKERRQ(ierr); 
  periodic = (DAPeriodicType) pt;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-stencil_type",&st,PETSC_NULL);CHKERRQ(ierr); 
  stencil_type = (DAStencilType) st;

  ierr = PetscOptionsHasName(PETSC_NULL,"-1d",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-2d",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-3d",&flg3);CHKERRQ(ierr);
  if (flg2) {
    ierr = DACreate2d(PETSC_COMM_WORLD,periodic,stencil_type,M,N,m,n,dof,stencil_width,0,0,&da);CHKERRQ(ierr);
    ierr = SDACreate2d(PETSC_COMM_WORLD,periodic,stencil_type,M,N,m,n,dof,stencil_width,0,0,&sda);CHKERRQ(ierr);
  } else if (flg3) {
    ierr = DACreate3d(PETSC_COMM_WORLD,periodic,stencil_type,M,N,P,m,n,p,dof,stencil_width,0,0,0,&da);CHKERRQ(ierr);
    ierr = SDACreate3d(PETSC_COMM_WORLD,periodic,stencil_type,M,N,P,m,n,p,dof,stencil_width,0,0,0,&sda);CHKERRQ(ierr);
  }
  else {
    ierr = DACreate1d(PETSC_COMM_WORLD,periodic,M,dof,stencil_width,PETSC_NULL,&da);CHKERRQ(ierr);
    ierr = SDACreate1d(PETSC_COMM_WORLD,periodic,M,dof,stencil_width,PETSC_NULL,&sda);CHKERRQ(ierr);
  }

  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = VecDuplicate(local,&local_copy);CHKERRQ(ierr);
  
  /* zero out vectors so that ghostpoints are zero */
  value = 0;
  ierr = VecSet(local,value);CHKERRQ(ierr);
  ierr = VecSet(local_copy,value);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    value = i + 1;
    ierr = VecSetValues(global,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(global);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(global);CHKERRQ(ierr);

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-same_array",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    /* test the case where the input and output array is the same */
    ierr = VecCopy(local,local_copy);CHKERRQ(ierr);
    ierr = VecGetArray(local_copy,&in);CHKERRQ(ierr);
    ierr = VecRestoreArray(local_copy,PETSC_NULL);CHKERRQ(ierr);
    ierr = SDALocalToLocalBegin(sda,in,INSERT_VALUES,in);CHKERRQ(ierr);
    ierr = SDALocalToLocalEnd(sda,in,INSERT_VALUES,in);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(local,&out);CHKERRQ(ierr);
    ierr = VecRestoreArray(local,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecGetArray(local_copy,&in);CHKERRQ(ierr);
    ierr = VecRestoreArray(local_copy,PETSC_NULL);CHKERRQ(ierr);
    ierr = SDALocalToLocalBegin(sda,out,INSERT_VALUES,in);CHKERRQ(ierr);
    ierr = SDALocalToLocalEnd(sda,out,INSERT_VALUES,in);CHKERRQ(ierr);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-save",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    sprintf(filename,"local.%d",rank);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer,&file);CHKERRQ(ierr);
    ierr = VecView(local,viewer);CHKERRQ(ierr);
    fprintf(file,"Vector with correct ghost points\n");
    ierr = VecView(local_copy,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }

  ierr = VecAXPY(local_copy,-1.0,local);CHKERRQ(ierr);
  ierr = VecNorm(local_copy,NORM_MAX,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&norm,1,MPIU_REAL,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (norm != 0.0) {
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of difference %G should be zero\n",norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %D %D %D %D\n",M,N,P,dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %D stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)periodic);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3);CHKERRQ(ierr);
  }
   
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = SDADestroy(sda);CHKERRQ(ierr);
  ierr = VecDestroy(local_copy);CHKERRQ(ierr);
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

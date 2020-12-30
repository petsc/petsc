
static char help[] = "Tests DMLocalToLocalxxx() for DMDA.\n\n";

#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt      rank;
  PetscInt         M=8,dof=1,stencil_width=1,i,start,end,P=5,N = 6,m=PETSC_DECIDE,n=PETSC_DECIDE,p=PETSC_DECIDE,pt = 0,st = 0;
  PetscErrorCode   ierr;
  PetscBool        flg = PETSC_FALSE,flg2,flg3;
  DMBoundaryType   periodic;
  DMDAStencilType  stencil_type;
  DM               da;
  Vec              local,global,local_copy;
  PetscScalar      value;
  PetscReal        norm,work;
  PetscViewer      viewer;
  char             filename[64];
  FILE             *file;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-periodic",&pt,NULL);CHKERRQ(ierr);

  periodic = (DMBoundaryType) pt;

  ierr = PetscOptionsGetInt(NULL,NULL,"-stencil_type",&st,NULL);CHKERRQ(ierr);

  stencil_type = (DMDAStencilType) st;

  ierr = PetscOptionsHasName(NULL,NULL,"-grid2d",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-grid3d",&flg3);CHKERRQ(ierr);
  if (flg2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,periodic,periodic,stencil_type,M,N,m,n,dof,stencil_width,NULL,NULL,&da);CHKERRQ(ierr);
  } else if (flg3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,periodic,periodic,periodic,stencil_type,M,N,P,m,n,p,dof,stencil_width,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,periodic,M,dof,stencil_width,NULL,&da);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = VecDuplicate(local,&local_copy);CHKERRQ(ierr);


  /* zero out vectors so that ghostpoints are zero */
  value = 0;
  ierr  = VecSet(local,value);CHKERRQ(ierr);
  ierr  = VecSet(local_copy,value);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    value = i + 1;
    ierr  = VecSetValues(global,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(global);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(global);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);


  ierr = DMLocalToLocalBegin(da,local,INSERT_VALUES,local_copy);CHKERRQ(ierr);
  ierr = DMLocalToLocalEnd(da,local,INSERT_VALUES,local_copy);CHKERRQ(ierr);


  ierr = PetscOptionsGetBool(NULL,NULL,"-save",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
    sprintf(filename,"local.%d",rank);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer,&file);CHKERRQ(ierr);
    ierr = VecView(local,viewer);CHKERRQ(ierr);
    fprintf(file,"Vector with correct ghost points\n");
    ierr = VecView(local_copy,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = VecAXPY(local_copy,-1.0,local);CHKERRQ(ierr);
  ierr = VecNorm(local_copy,NORM_MAX,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&norm,1,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of difference %g should be zero\n",(double)norm);CHKERRQ(ierr);

  ierr = VecDestroy(&local_copy);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 8
      args: -dof 3 -stencil_width 2 -M 50 -N 50 -periodic

   test:
      suffix: 2
      nsize: 8
      args: -dof 3 -stencil_width 2 -M 50 -N 50 -periodic -grid2d
      output_file: output/ex7_1.out

   test:
      suffix: 3
      nsize: 8
      args: -dof 3 -stencil_width 2 -M 50 -N 50 -periodic -grid3d
      output_file: output/ex7_1.out

TEST*/

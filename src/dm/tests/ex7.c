
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
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-periodic",&pt,NULL));

  periodic = (DMBoundaryType) pt;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_type",&st,NULL));

  stencil_type = (DMDAStencilType) st;

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-grid2d",&flg2));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-grid3d",&flg3));
  if (flg2) {
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,periodic,periodic,stencil_type,M,N,m,n,dof,stencil_width,NULL,NULL,&da));
  } else if (flg3) {
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,periodic,periodic,periodic,stencil_type,M,N,P,m,n,p,dof,stencil_width,NULL,NULL,NULL,&da));
  } else {
    CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,periodic,M,dof,stencil_width,NULL,&da));
  }
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));
  CHKERRQ(VecDuplicate(local,&local_copy));

  /* zero out vectors so that ghostpoints are zero */
  value = 0;
  CHKERRQ(VecSet(local,value));
  CHKERRQ(VecSet(local_copy,value));

  CHKERRQ(VecGetOwnershipRange(global,&start,&end));
  for (i=start; i<end; i++) {
    value = i + 1;
    CHKERRQ(VecSetValues(global,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(global));
  CHKERRQ(VecAssemblyEnd(global));

  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  CHKERRQ(DMLocalToLocalBegin(da,local,INSERT_VALUES,local_copy));
  CHKERRQ(DMLocalToLocalEnd(da,local,INSERT_VALUES,local_copy));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-save",&flg,NULL));
  if (flg) {
    CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    sprintf(filename,"local.%d",rank);
    CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer));
    CHKERRQ(PetscViewerASCIIGetPointer(viewer,&file));
    CHKERRQ(VecView(local,viewer));
    fprintf(file,"Vector with correct ghost points\n");
    CHKERRQ(VecView(local_copy,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  CHKERRQ(VecAXPY(local_copy,-1.0,local));
  CHKERRQ(VecNorm(local_copy,NORM_MAX,&work));
  CHKERRMPI(MPI_Allreduce(&work,&norm,1,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of difference %g should be zero\n",(double)norm));

  CHKERRQ(VecDestroy(&local_copy));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
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

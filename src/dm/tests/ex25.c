
static char help[] = "Tests DMLocalToGlobal() for dof > 1\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt       M = 6,N = 5,P = 4, m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE,i,j,k,is,js,ks,in,jen,kn;
  DM             da;
  Vec            local,global;
  PetscScalar    ****l;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,P,m,n,p,2,1,NULL,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));

  CHKERRQ(DMDAGetCorners(da,&is,&js,&ks,&in,&jen,&kn));
  CHKERRQ(DMDAVecGetArrayDOF(da,local,&l));
  for (i=is; i<is+in; i++) {
    for (j=js; j<js+jen; j++) {
      for (k=ks; k<ks+kn; k++) {
        l[k][j][i][0] = 2*(i + j*M + k*M*N);
        l[k][j][i][1] = 2*(i + j*M + k*M*N) + 1;
      }
    }
  }
  CHKERRQ(DMDAVecRestoreArrayDOF(da,local,&l));
  CHKERRQ(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  /* Free memory */
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

      test:
         filter: grep -v -i Process
         output_file: output/ex25_1.out

      test:
         suffix: 2
         nsize: 2
         filter: grep -v -i Process
         output_file: output/ex25_2.out

      test:
         suffix: 3
         nsize: 3
         filter: grep -v -i Process
         output_file: output/ex25_2.out

      test:
         suffix: 4
         nsize: 4
         filter: grep -v -i Process
         output_file: output/ex25_2.out

      test:
         suffix: 5
         nsize: 5
         filter: grep -v -i Process
         output_file: output/ex25_2.out

      test:
         suffix: 6
         nsize: 6
         filter: grep -v -i Process
         output_file: output/ex25_2.out

TEST*/

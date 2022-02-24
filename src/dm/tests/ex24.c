
static char help[] = "Tests DMLocalToGlobal() for dof > 1\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt       M = 6,N = 5,m = PETSC_DECIDE,n = PETSC_DECIDE,i,j,is,js,in,jen;
  PetscErrorCode ierr;
  DM             da;
  Vec            local,global;
  PetscScalar    ***l;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,3,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));

  CHKERRQ(DMDAGetCorners(da,&is,&js,0,&in,&jen,0));
  CHKERRQ(DMDAVecGetArrayDOF(da,local,&l));
  for (i=is; i<is+in; i++) {
    for (j=js; j<js+jen; j++) {
      l[j][i][0] = 3*(i + j*M);
      l[j][i][1] = 3*(i + j*M) + 1;
      l[j][i][2] = 3*(i + j*M) + 2;
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
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

      test:
         filter: grep -v -i Process
         output_file: output/ex24_1.out

      test:
         suffix: 2
         nsize: 2
         filter: grep -v -i Process
         output_file: output/ex24_2.out

      test:
         suffix: 3
         nsize: 3
         filter: grep -v -i Process
         output_file: output/ex24_2.out

      test:
         suffix: 4
         nsize: 4
         filter: grep -v -i Process
         output_file: output/ex24_2.out

      test:
         suffix: 5
         nsize: 5
         filter: grep -v -i Process
         output_file: output/ex24_2.out

      test:
         suffix: 6
         nsize: 6
         filter: grep -v -i Process
         output_file: output/ex24_2.out

TEST*/

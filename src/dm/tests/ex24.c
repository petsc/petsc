
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
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,3,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&is,&js,0,&in,&jen,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,local,&l);CHKERRQ(ierr);
  for (i=is; i<is+in; i++) {
    for (j=js; j<js+jen; j++) {
      l[j][i][0] = 3*(i + j*M);
      l[j][i][1] = 3*(i + j*M) + 1;
      l[j][i][2] = 3*(i + j*M) + 2;
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da,local,&l);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,ADD_VALUES,global);CHKERRQ(ierr);

  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free memory */
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
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

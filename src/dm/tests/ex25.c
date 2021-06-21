
static char help[] = "Tests DMLocalToGlobal() for dof > 1\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt       M = 6,N = 5,P = 4, m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE,i,j,k,is,js,ks,in,jen,kn;
  PetscErrorCode ierr;
  DM             da;
  Vec            local,global;
  PetscScalar    ****l;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* Create distributed array and get vectors */
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,P,m,n,p,2,1,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&is,&js,&ks,&in,&jen,&kn);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,local,&l);CHKERRQ(ierr);
  for (i=is; i<is+in; i++) {
    for (j=js; j<js+jen; j++) {
      for (k=ks; k<ks+kn; k++) {
        l[k][j][i][0] = 2*(i + j*M + k*M*N);
        l[k][j][i][1] = 2*(i + j*M + k*M*N) + 1;
      }
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

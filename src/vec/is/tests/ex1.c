
static char help[]= "Tests ISLocalToGlobalMappingCreateIS() for bs > 1.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               bs = 2,n = 3,ix[3] = {1,7,9},iy[2] = {0,2},mp[2];
  IS                     isx;
  ISLocalToGlobalMapping ltog;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,n,ix,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(isx,&ltog);CHKERRQ(ierr);

  ierr = PetscIntView(2,iy,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltog,2,iy,mp);CHKERRQ(ierr);
  ierr = PetscIntView(2,mp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscIntView(2,iy,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyBlock(ltog,2,iy,mp);CHKERRQ(ierr);
  ierr = PetscIntView(2,mp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  ierr = ISDestroy(&isx);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:

TEST*/

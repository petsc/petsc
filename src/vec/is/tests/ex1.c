
static char help[]= "Tests ISLocalToGlobalMappingCreateIS() for bs > 1.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt               bs = 2,n = 3,ix[3] = {1,7,9},iy[2] = {0,2},mp[2];
  IS                     isx;
  ISLocalToGlobalMapping ltog;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,n,ix,PETSC_COPY_VALUES,&isx));
  CHKERRQ(ISLocalToGlobalMappingCreateIS(isx,&ltog));

  CHKERRQ(PetscIntView(2,iy,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISLocalToGlobalMappingApply(ltog,2,iy,mp));
  CHKERRQ(PetscIntView(2,mp,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscIntView(2,iy,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISLocalToGlobalMappingApplyBlock(ltog,2,iy,mp));
  CHKERRQ(PetscIntView(2,mp,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISLocalToGlobalMappingDestroy(&ltog));
  CHKERRQ(ISDestroy(&isx));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/

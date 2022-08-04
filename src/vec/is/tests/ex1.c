
static char help[]= "Tests ISLocalToGlobalMappingCreateIS() for bs > 1.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt               bs = 2,n = 3,ix[3] = {1,7,9},iy[2] = {0,2},mp[2];
  IS                     isx;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,n,ix,PETSC_COPY_VALUES,&isx));
  PetscCall(ISLocalToGlobalMappingCreateIS(isx,&ltog));

  PetscCall(PetscIntView(2,iy,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISLocalToGlobalMappingApply(ltog,2,iy,mp));
  PetscCall(PetscIntView(2,mp,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscIntView(2,iy,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISLocalToGlobalMappingApplyBlock(ltog,2,iy,mp));
  PetscCall(PetscIntView(2,mp,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
  PetscCall(ISDestroy(&isx));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/

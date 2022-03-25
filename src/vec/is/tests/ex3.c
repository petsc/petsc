
static char help[]= "Tests ISSetBlockSize() on ISBlock().\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt               bs = 2,n = 3,ix[3] = {1,7,9};
  const PetscInt         *indices;
  IS                     is;
  PetscBool              broken = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-broken",&broken,NULL));
  PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,n,ix,PETSC_COPY_VALUES,&is));
  PetscCall(ISGetIndices(is,&indices));
  PetscCall(PetscIntView(bs*3,indices,NULL));
  PetscCall(ISRestoreIndices(is,&indices));
  if (broken) {
    PetscCall(ISSetBlockSize(is,3));
    PetscCall(ISGetIndices(is,&indices));
    PetscCall(PetscIntView(bs*3,indices,NULL));
    PetscCall(ISRestoreIndices(is,&indices));
  }
  PetscCall(ISDestroy(&is));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
     suffix: 2
     args:  -broken
     filter: Error: grep -o  "[0]PETSC ERROR: Object is in wrong state"

TEST*/

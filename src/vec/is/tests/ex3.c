
static char help[]= "Tests ISSetBlockSize() on ISBlock().\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               bs = 2,n = 3,ix[3] = {1,7,9};
  const PetscInt         *indices;
  IS                     is;
  PetscBool              broken = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-broken",&broken,NULL));
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,n,ix,PETSC_COPY_VALUES,&is));
  CHKERRQ(ISGetIndices(is,&indices));
  CHKERRQ(PetscIntView(bs*3,indices,NULL));
  CHKERRQ(ISRestoreIndices(is,&indices));
  if (broken) {
    CHKERRQ(ISSetBlockSize(is,3));
    CHKERRQ(ISGetIndices(is,&indices));
    CHKERRQ(PetscIntView(bs*3,indices,NULL));
    CHKERRQ(ISRestoreIndices(is,&indices));
  }
  CHKERRQ(ISDestroy(&is));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
     suffix: 2
     args:  -broken
     filter: Error: grep -o  "[0]PETSC ERROR: Object is in wrong state"

TEST*/


/*
       Formatted test for ISStride routines.
*/

static char help[] = "Tests IS stride routines.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt       i,n,start,stride;
  const PetscInt *ii;
  IS             is;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*
     Test IS of size 0
  */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,2,&is));
  CHKERRQ(ISGetSize(is,&n));
  PetscCheckFalse(n != 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISCreateStride");
  CHKERRQ(ISStrideGetInfo(is,&start,&stride));
  PetscCheckFalse(start != 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStrideGetInfo");
  PetscCheckFalse(stride != 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStrideGetInfo");
  CHKERRQ(PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStride");
  CHKERRQ(ISGetIndices(is,&ii));
  CHKERRQ(ISRestoreIndices(is,&ii));
  CHKERRQ(ISDestroy(&is));

  /*
     Test ISGetIndices()
  */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,10000,-8,3,&is));
  CHKERRQ(ISGetLocalSize(is,&n));
  CHKERRQ(ISGetIndices(is,&ii));
  for (i=0; i<10000; i++) {
    PetscCheckFalse(ii[i] != -8 + 3*i,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetIndices");
  }
  CHKERRQ(ISRestoreIndices(is,&ii));
  CHKERRQ(ISDestroy(&is));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     output_file: output/ex1_1.out

TEST*/

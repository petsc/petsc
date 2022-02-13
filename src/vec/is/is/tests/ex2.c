
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
  ierr = ISCreateStride(PETSC_COMM_SELF,0,0,2,&is);CHKERRQ(ierr);
  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  PetscCheckFalse(n != 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISCreateStride");
  ierr = ISStrideGetInfo(is,&start,&stride);CHKERRQ(ierr);
  PetscCheckFalse(start != 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStrideGetInfo");
  PetscCheckFalse(stride != 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStrideGetInfo");
  ierr = PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStride");
  ierr = ISGetIndices(is,&ii);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ii);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /*
     Test ISGetIndices()
  */
  ierr = ISCreateStride(PETSC_COMM_SELF,10000,-8,3,&is);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ii);CHKERRQ(ierr);
  for (i=0; i<10000; i++) {
    PetscCheckFalse(ii[i] != -8 + 3*i,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetIndices");
  }
  ierr = ISRestoreIndices(is,&ii);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     output_file: output/ex1_1.out

TEST*/

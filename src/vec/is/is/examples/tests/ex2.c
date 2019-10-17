
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
  if (n != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISCreateStride");
  ierr = ISStrideGetInfo(is,&start,&stride);CHKERRQ(ierr);
  if (start != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStrideGetInfo");
  if (stride != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStrideGetInfo");
  ierr = PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStride");
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
    if (ii[i] != -8 + 3*i) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetIndices");
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

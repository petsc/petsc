
static char help[] = "Saves 4by4 block matrix.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       i,j;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscViewer    fd;
  PetscScalar    values[16],one = 1.0;
  Vec            x;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,m"Can only run on one processor");

  /*
     Open binary file.  Note that we use FILE_MODE_WRITE to indicate
     writing to this file.
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"4by4",FILE_MODE_WRITE,&fd));

  CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_WORLD,4,12,12,0,0,&A));

  for (i=0; i<16; i++) values[i] = i;
  for (i=0; i<4; i++) values[4*i+i] += 5;
  i    = 0; j = 0;
  CHKERRQ(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;
  i    = 0; j = 2;
  CHKERRQ(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;
  i    = 1; j = 0;
  CHKERRQ(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;for (i=0; i<4; i++) values[4*i+i] += 6;
  i    = 1; j = 1;
  CHKERRQ(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;
  i    = 2; j = 0;
  CHKERRQ(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;for (i=0; i<4; i++) values[4*i+i] += 7;
  i    = 2; j = 2;
  CHKERRQ(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,fd));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,12,&x));
  CHKERRQ(VecSet(x,one));
  CHKERRQ(VecView(x,fd));
  CHKERRQ(VecDestroy(&x));

  CHKERRQ(PetscViewerDestroy(&fd));
  ierr = PetscFinalize();
  return ierr;
}

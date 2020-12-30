
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,m"Can only run on one processor");

  /*
     Open binary file.  Note that we use FILE_MODE_WRITE to indicate
     writing to this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"4by4",FILE_MODE_WRITE,&fd);CHKERRQ(ierr);

  ierr = MatCreateSeqBAIJ(PETSC_COMM_WORLD,4,12,12,0,0,&A);CHKERRQ(ierr);

  for (i=0; i<16; i++) values[i] = i;
  for (i=0; i<4; i++) values[4*i+i] += 5;
  i    = 0; j = 0;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRQ(ierr);

  for (i=0; i<16; i++) values[i] = i;
  i    = 0; j = 2;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRQ(ierr);

  for (i=0; i<16; i++) values[i] = i;
  i    = 1; j = 0;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRQ(ierr);

  for (i=0; i<16; i++) values[i] = i;for (i=0; i<4; i++) values[4*i+i] += 6;
  i    = 1; j = 1;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRQ(ierr);

  for (i=0; i<16; i++) values[i] = i;
  i    = 2; j = 0;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRQ(ierr);

  for (i=0; i<16; i++) values[i] = i;for (i=0; i<4; i++) values[4*i+i] += 7;
  i    = 2; j = 2;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,fd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_WORLD,12,&x);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecView(x,fd);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

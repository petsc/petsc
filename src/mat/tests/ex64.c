
static char help[] = "Saves 4by4 block matrix.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       i,j;
  PetscMPIInt    size;
  PetscViewer    fd;
  PetscScalar    values[16],one = 1.0;
  Vec            x;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,m"Can only run on one processor");

  /*
     Open binary file.  Note that we use FILE_MODE_WRITE to indicate
     writing to this file.
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"4by4",FILE_MODE_WRITE,&fd));

  PetscCall(MatCreateSeqBAIJ(PETSC_COMM_WORLD,4,12,12,0,0,&A));

  for (i=0; i<16; i++) values[i] = i;
  for (i=0; i<4; i++) values[4*i+i] += 5;
  i    = 0; j = 0;
  PetscCall(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;
  i    = 0; j = 2;
  PetscCall(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;
  i    = 1; j = 0;
  PetscCall(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;for (i=0; i<4; i++) values[4*i+i] += 6;
  i    = 1; j = 1;
  PetscCall(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;
  i    = 2; j = 0;
  PetscCall(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  for (i=0; i<16; i++) values[i] = i;for (i=0; i<4; i++) values[4*i+i] += 7;
  i    = 2; j = 2;
  PetscCall(MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,fd));
  PetscCall(MatDestroy(&A));

  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,12,&x));
  PetscCall(VecSet(x,one));
  PetscCall(VecView(x,fd));
  PetscCall(VecDestroy(&x));

  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(PetscFinalize());
  return 0;
}

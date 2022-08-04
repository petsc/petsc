
static char help[] = "Tests converting a SBAIJ matrix to BAIJ format with MatConvert. Modified from ex55.c\n\n";

#include <petscmat.h>
/* Usage: ./ex141 -mat_view */

int main(int argc,char **args)
{
  Mat            C,B;
  PetscInt       i,bs=2,mbs,m,block,d_nz=6,col[3];
  PetscMPIInt    size;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      equal,loadmat;
  PetscScalar    value[4];
  PetscInt       ridx[2],cidx[2];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&loadmat));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* input matrix C */
  if (loadmat) {
    /* Open binary file. Load a sbaij matrix, then destroy the viewer. */
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
    PetscCall(MatSetType(C,MATSEQSBAIJ));
    PetscCall(MatLoad(C,fd));
    PetscCall(PetscViewerDestroy(&fd));
  } else { /* Create a sbaij mat with bs>1  */
    mbs  =8;
    PetscCall(PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL));
    m    = mbs*bs;
    PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
    PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m,m));
    PetscCall(MatSetType(C,MATSBAIJ));
    PetscCall(MatSetFromOptions(C));
    PetscCall(MatSeqSBAIJSetPreallocation(C,bs,d_nz,NULL));
    PetscCall(MatSetUp(C));
    PetscCall(MatSetOption(C,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE));

    for (block=0; block<mbs; block++) {
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        PetscCall(MatSetValues(C,1,&i,3,col,value,INSERT_VALUES));
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;

      value[0]=-1.0; value[1]=4.0;
      PetscCall(MatSetValues(C,1,&i,2,col,value,INSERT_VALUES));

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;

      value[0]=4.0; value[1] = -1.0;
      PetscCall(MatSetValues(C,1,&i,2,col,value,INSERT_VALUES));
    }
    /* off-diagonal blocks */
    value[0]=-1.0; value[1] = -0.1; value[2] = 0.0; value[3] = -1.0; /* row-oriented */
    for (block=0; block<mbs-1; block++) {
      for (i=0; i<bs; i++) {
        ridx[i] = block*bs+i; cidx[i] = (block+1)*bs+i;
      }
      PetscCall(MatSetValues(C,bs,ridx,bs,cidx,value,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  }

  /* convert C to BAIJ format */
  PetscCall(MatConvert(C,MATSEQBAIJ,MAT_INITIAL_MATRIX,&B));
  PetscCall(MatMultEqual(B,C,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatConvert fails!");

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     output_file: output/ex141.out

TEST*/

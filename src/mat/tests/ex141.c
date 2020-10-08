
static char help[] = "Tests converting a SBAIJ matrix to BAIJ format with MatConvert. Modified from ex55.c\n\n";

#include <petscmat.h>
/* Usage: ./ex141 -mat_view */

int main(int argc,char **args)
{
  Mat            C,B;
  PetscErrorCode ierr;
  PetscInt       i,bs=2,mbs,m,block,d_nz=6,col[3];
  PetscMPIInt    size;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      equal,loadmat;
  PetscScalar    value[4];
  PetscInt       ridx[2],cidx[2];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&loadmat);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* input matrix C */
  if (loadmat) {
    /* Open binary file. Load a sbaij matrix, then destroy the viewer. */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
    ierr = MatSetType(C,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatLoad(C,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  } else { /* Create a sbaij mat with bs>1  */
    mbs  =8;
    ierr = PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL);CHKERRQ(ierr);
    m    = mbs*bs;
    ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
    ierr = MatSetType(C,MATSBAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(C);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(C,bs,d_nz,NULL);CHKERRQ(ierr);
    ierr = MatSetUp(C);CHKERRQ(ierr);
    ierr = MatSetOption(C,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);

    for (block=0; block<mbs; block++) {
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(C,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;

      value[0]=-1.0; value[1]=4.0;
      ierr    = MatSetValues(C,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;

      value[0]=4.0; value[1] = -1.0;
      ierr    = MatSetValues(C,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* off-diagonal blocks */
    value[0]=-1.0; value[1] = -0.1; value[2] = 0.0; value[3] = -1.0; /* row-oriented */
    for (block=0; block<mbs-1; block++) {
      for (i=0; i<bs; i++) {
        ridx[i] = block*bs+i; cidx[i] = (block+1)*bs+i;
      }
      ierr = MatSetValues(C,bs,ridx,bs,cidx,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* convert C to BAIJ format */
  ierr = MatConvert(C,MATSEQBAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatMultEqual(B,C,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatConvert fails!");

  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     output_file: output/ex141.out

TEST*/

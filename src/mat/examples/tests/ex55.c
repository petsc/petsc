
static char help[] = "Tests converting a matrix to another format with MatConvert().\n\n";

#include "src/mat/matimpl.h"
#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C,A,B,D; 
  PetscErrorCode ierr;
  PetscInt       i,j,ntypes = 3,bs,mbs,m,block,d_nz=3, o_nz=3,col[3],row,msglvl=1;
  PetscMPIInt    size,rank;
  /* const MatType  type[9] = {MATMPIAIJ,MATMPIBAIJ,MATMPIROWBS};*/ /* BlockSolve95 is required for MATMPIROWBS */
  const MatType  type[9] = {MATSEQAIJ,MATSEQBAIJ,MATSEQSBAIJ}; 
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscTruth     equal,flg_loadmat;
  PetscScalar    value[3];

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN-1,&flg_loadmat);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_SUP,"At present, this is a uniprocessor example!");

  /* input matrix C */
  if (flg_loadmat){
    /* Open binary file.  Note that we use PETSC_FILE_RDONLY to indicate
       reading from this file. */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_FILE_RDONLY,&fd);CHKERRQ(ierr);

    /* Load the matrix, then destroy the viewer. */
    ierr = MatLoad(fd,MATAIJ,&C);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);
    bs = 1;  /* assume the loaded matrix has block size 1 */
  } else { /* Convert a baij mat with bs>1 to other formats */
    bs = 2; mbs=8;
    ierr = PetscOptionsGetInt(PETSC_NULL,"-mbs",&mbs,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRQ(ierr);
    if (bs <= 1) SETERRQ(PETSC_ERR_ARG_WRONG," bs must be >1 in this case");
    m = mbs*bs;
    ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,m,m,d_nz,PETSC_NULL,o_nz,PETSC_NULL,&C);CHKERRQ(ierr);
    for (block=0; block<mbs; block++){
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(C,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);    
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
      value[0]=-1.0; value[1]=4.0;  
      ierr = MatSetValues(C,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr); 

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
      value[0]=4.0; value[1] = -1.0; 
      ierr = MatSetValues(C,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);  
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(mbs-1)*bs; i++){
      col[0]=i+bs;
      ierr = MatSetValues(C,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      col[0]=i; row=i+bs;
      ierr = MatSetValues(C,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  
    ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  
  /* convert C to other formats */
  for (i=0; i<ntypes; i++) {
    ierr = MatConvert(C,type[i],&A);CHKERRQ(ierr);
    for (j=0; j<ntypes; j++) { 
      if (j==i) continue; 
      if (msglvl>0 && !rank)
        ierr = PetscPrintf(PETSC_COMM_SELF," [%d] test conversion between %s and %s\n",rank,type[i],type[j]);

      ierr = MatConvert(A,type[j],&B);CHKERRQ(ierr);
      ierr = MatConvert(B,type[i],&D);CHKERRQ(ierr); 

      if (bs == 1){
        ierr = MatEqual(A,D,&equal);CHKERRQ(ierr);
        if (!equal){
          ierr = PetscPrintf(PETSC_COMM_SELF," A: %s\n",type[i]);
          MatView(A,PETSC_VIEWER_STDOUT_WORLD);
          ierr = PetscPrintf(PETSC_COMM_SELF," B: %s\n",type[j]);
          MatView(B,PETSC_VIEWER_STDOUT_WORLD);
          ierr = PetscPrintf(PETSC_COMM_SELF," D: %s\n",type[i]);
          MatView(D,PETSC_VIEWER_STDOUT_WORLD);
          SETERRQ2(1,"Error in conversion from %s to %s",type[i],type[j]);
        }
      } else { /* bs > 1 */
        ierr = MatMultEqual(A,B,10,&equal);CHKERRQ(ierr);
        if (!equal) SETERRQ2(PETSC_ERR_ARG_NOTSAMETYPE,"Error in conversion from %s to %s",type[i],type[j]);
      }
      ierr = MatDestroy(B);CHKERRQ(ierr);
      ierr = MatDestroy(D);CHKERRQ(ierr); 
    }
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  ierr = MatDestroy(C);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}












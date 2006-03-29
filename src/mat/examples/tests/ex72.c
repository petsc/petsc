
#if !defined(PETSC_USE_COMPLEX)

static char help[] = "Reads in a Symmetric matrix in MatrixMarket format. Writes\n\
it using the PETSc sparse format. It also adds a Vector set to random values to the\n\
output file. Input parameters are:\n\
  -fin <filename> : input file\n\
  -fout <filename> : output file\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  Vec            b;
  char           filein[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN],buf[PETSC_MAX_PATH_LEN];
  PetscInt       i,m,n,nnz,col,row;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscScalar    val;
  FILE*          file;
  PetscViewer    view;
  PetscRandom    r;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(1,"Uniprocessor Example only\n");

  /* Read in matrix and RHS */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",filein,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF,filein,"r",&file);CHKERRQ(ierr);

  /* Ignore the first line */
  /* while (getc(file) != '\n'); */
  fgets(buf,PETSC_MAX_PATH_LEN-1,file);
  printf("%s",buf);
  fscanf(file,"%d %d %d\n",&m,&n,&nnz);
  printf ("m = %d, n = %d, nnz = %d\n",m,n,nnz);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,20,0,&A);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
  ierr = VecSetRandom(b,r);CHKERRQ(ierr);

  for (i=0; i<nnz; i++) {
    fscanf(file,"%d %d %le\n",&row,&col,(double*)&val);
    row = row-1; col = col-1 ;
    ierr = MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
    if (row != col) {
      ierr = MatSetValues(A,1,&col,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  fclose(file);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Reading matrix completes.\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",fileout,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileout,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = VecView(b,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(view);CHKERRQ(ierr);

  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(r);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
#else
#include <stdio.h>
int main(int argc,char **args)
{
  fprintf(stdout,"This example does not work for complex numbers.\n");
  return 0;
}
#endif

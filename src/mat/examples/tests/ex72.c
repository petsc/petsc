
#include <petscmat.h>

static char help[] = "Read in a Symmetric matrix in MatrixMarket format (only the lower triangle). \n\
  Assemble it to a PETSc sparse SBAIJ (upper triangle) matrix. \n\
  Write it in a AIJ matrix (entire matrix) to a file. \n\
  Input parameters are:            \n\
    -fin <filename> : input file   \n\
    -fout <filename> : output file \n\n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  char           filein[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN],buf[PETSC_MAX_PATH_LEN];
  PetscInt       i,m,n,nnz;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscScalar    *VAL,zero=0.0;
  FILE           *file;
  PetscViewer    view;
  int            *I,*J,*rownz;

  PetscInitialize(&argc,&args,(char*)0,help);
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example does not work with complex numbers");
#else
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,1,"Uniprocessor Example only\n");

  /* Read in matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-fin",filein,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF,filein,"r",&file);CHKERRQ(ierr);

  /* process header with comments */
  do fgets(buf,PETSC_MAX_PATH_LEN-1,file);
  while (buf[0] == '%');

  /* The first non-comment line has the matrix dimensions */
  sscanf(buf,"%d %d %d\n",&m,&n,&nnz);
  ierr = PetscPrintf (PETSC_COMM_SELF,"m = %d, n = %d, nnz = %d\n",m,n,nnz);

  /* reseve memory for matrices */
  ierr = PetscMalloc4(nnz,&I,nnz,&J,nnz,&VAL,m,&rownz);CHKERRQ(ierr);
  for (i=0; i<m; i++) rownz[i] = 1; /* add 0.0 to diagonal entries */

  for (i=0; i<nnz; i++) {
    ierr = fscanf(file,"%d %d %le\n",&I[i],&J[i],(double*)&VAL[i]);
    if (ierr == EOF) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"i=%d, reach EOF\n",i);
    I[i]--; J[i]--;    /* adjust from 1-based to 0-based */
    rownz[J[i]]++;
  }
  fclose(file);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Read file completes.\n");CHKERRQ(ierr);

  /* Creat and asseble SBAIJ matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,1,0,rownz);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /* Add zero to diagonals, in case the matrix missing diagonals */
  for (i=0; i<m; i++){
    ierr = MatSetValues(A,1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=0; i<nnz; i++) {
    ierr = MatSetValues(A,1,&J[i],1,&I[i],&VAL[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Assemble SBAIJ matrix completes.\n");CHKERRQ(ierr);

  /* Write out matrix in AIJ format */
  ierr = PetscOptionsGetString(NULL,NULL,"-fout",fileout,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileout,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  ierr = PetscFree4(I,J,VAL,rownz);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
#endif
  return 0;
}



static char help[] = "Reads in a PETSc binary matrix and saves in Harwell-Boeing format.\n\
  -fout <output_file> : file to load.\n\
  -fin <input_file> : For a 5X5 example of the 5-pt. stencil,\n\
                       use the file petsc/src/mat/examples/matbinary.ex\n\n";

/*
  Include the private file (not included by most applications) so we have direct
  access to the matrix data structure.

  This code is buggy! What is it doing here?
*/
#include "../src/mat/impls/aij/seq/aij.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       n,m,i,*ai,*aj,nz;
  PetscMPIInt    size;
  Mat            A;
  Vec            x;
  char           bfile[PETSC_MAX_PATH_LEN],hbfile[PETSC_MAX_PATH_LEN]; 
  PetscViewer    fd;
  Mat_SeqAIJ     *a;
  PetscScalar    *aa,*xx;
  FILE           *file;
  char           head[81];

  PetscInitialize(&argc,&args,(char *)0,help);

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(1,"This example does not work with complex numbers");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(1,"Only runs on one processor");

  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",bfile,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",hbfile,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);

  /* Read matrix and RHS */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,bfile,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&A);CHKERRQ(ierr);
  ierr = VecLoad(fd,PETSC_NULL,&x);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  /* Format is in column storage so we print transpose matrix */
  ierr = MatTranspose(A,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);

  m = A->m;
  n = A->n;
  if (n != m) SETERRQ(1,"Only for square matrices");

  /* charrage returns \n may not belong below
    depends on what 80 character fixed format means to Fortran */

  file = fopen(hbfile,"w"); if (!file) SETERRQ(1,"Cannot open HB file");
  sprintf(head,"%-72s%-8s\n","Title","Key");
  fprintf(file,head);
  a  = (Mat_SeqAIJ*)A->data;
  aa = a->a;
  ai = a->i;
  aj = a->j;
  nz = a->nz;


  sprintf(head,"%14d%14d%14d%14d%14d%10s\n",3*m+1,m+1,nz,nz," ");
  fprintf(file,head);
  sprintf(head,"RUA%14d%14d%14d%14d%13s\n",m,m,nz," ");
  fprintf(file,head);

  fprintf(file,"Formats I don't know\n");

  for (i=0; i<m+1; i++) {
    fprintf(file,"%10d%70s\n",ai[i]," ");
  }
  for (i=0; i<nz; i++) {
    fprintf(file,"%10d%70s\n",aj[i]," ");
  }

  for (i=0; i<nz; i++) {
    fprintf(file,"%16.14e,%64s\n",aa[i]," ");
  }

  /* print the vector to the file */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    fprintf(file,"%16.14e%64s\n",xx[i]," ");
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);

  fclose(file);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

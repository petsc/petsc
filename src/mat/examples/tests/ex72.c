/*$Id: ex72.c,v 1.8 1999/10/24 14:02:39 bsmith Exp bsmith $*/

#if !defined(PETSC_USE_COMPLEX)

static char help[] = "Reads in a Symmetric matrix in MatrixMarket format and writes\n\
it using the PETSc sparse format. It also adds a Vector set to random values to the\n\
output file. Input parameters are:\n\
  -fin <filename> : input file\n\
  -fout <filename> : output file\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         A;
  Vec         b;
  char        filein[128],fileout[128],buf[128];
  int         i, m, n, nnz, ierr, size, col, row;
  Scalar      val;
  FILE*       file;
  Viewer      view;
  PetscRandom r;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size > 1) SETERRA(1,0,"Uniprocessor Example only\n");

  /* Read in matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-fin",filein,127,PETSC_NULL);CHKERRA(ierr);
  if (!(file = PetscFOpen(PETSC_COMM_SELF,filein,"r"))) {
    SETERRA(1,0,"cannot open file\n");
  }

  /* Ignore the first line */
  /* while (getc(file) != '\n') ; */
  fgets(buf,128,file);
  printf("%s",buf);
  fscanf(file,"%d %d %d\n",&m,&n,&nnz);
  printf ("m = %d, n = %d, nnz = %d\n",m,n,nnz);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,20,0,&A);CHKERRA(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&b);CHKERRA(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRA(ierr);
  ierr = VecSetRandom(r,b);CHKERRA(ierr);

  for (i=0; i<nnz; i++) {
    fscanf(file,"%d %d %le\n",&row,&col,&val);
    row = row-1; col = col-1 ;
    ierr = MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES);CHKERRA(ierr);
    if (row != col) {
      ierr = MatSetValues(A,1,&col,1,&row,&val,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  fclose(file);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Reading matrix completes.\n");CHKERRA(ierr);
  ierr = OptionsGetString(PETSC_NULL,"-fout",fileout,127,PETSC_NULL);CHKERRA(ierr);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,fileout,BINARY_CREATE,&view);CHKERRA(ierr);
  ierr = MatView(A,view);CHKERRA(ierr);
  ierr = VecView(b,view);CHKERRA(ierr);
  ierr = ViewerDestroy(view);CHKERRA(ierr);

  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = PetscRandomDestroy(r);CHKERRA(ierr);

  PetscFinalize();
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

/*$Id: ex9.c,v 1.14 2000/01/11 21:01:03 bsmith Exp balay $*/

static char help[] = "Tests MPI parallel matrix creation.\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat        C; 
  MatType    type;
  MatInfo    info;
  int        i,j,m = 3,n = 2,rank,size,low,high,iglobal;
  int        I,J,ierr,ldim;
  PetscTruth set,flg;
  Scalar     v,one = 1.0;
  Vec        u,b;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  n = 2*size;

  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,0,&type,&set);CHKERRA(ierr);
  if (type == MATMPIBDIAG || type == MATSEQBDIAG) {
    int bs,ndiag,diag[7];  bs = 1,ndiag = 5;
    diag[0] = n;
    diag[1] = 1;
    diag[2] = 0;
    diag[3] = -1;
    diag[4] = -n;
    if (size>1) {ndiag = 7; diag[5] = 2; diag[6] = -2;}
    ierr = MatCreateMPIBDiag(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,m*n,
           ndiag,bs,diag,PETSC_NULL,&C);CHKERRA(ierr);
  } else if (type == MATMPIDENSE || type == MATSEQDENSE) {
    ierr = MatCreateMPIDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
           m*n,m*n,PETSC_NULL,&C);CHKERRA(ierr);
  } else if (type == MATMPIAIJ || type == MATSEQAIJ) {
    ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
           m*n,m*n,5,PETSC_NULL,5,PETSC_NULL,&C);CHKERRA(ierr);
  } else SETERRA(1,0,"Invalid matrix type for this example.");

  /* Create the matrix for the five point stencil, YET AGAIN */
  for (i=0; i<m; i++) { 
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  I = j + n*i;
      if (i>0)   {J = I - n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (i<m-1) {J = I + n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (j>0)   {J = I - 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (j<n-1) {J = I + 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }

  /* Add extra elements (to illustrate variants of MatGetInfo) */
  I = n; J = n-2; v = 100.0;
  ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);
  I = n-2; J = n; v = 100.0;
  ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Form vectors */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,&u);CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr);
  ierr = VecGetLocalSize(u,&ldim);
  ierr = VecGetOwnershipRange(u,&low,&high);CHKERRA(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = one*((double)i) + 100.0*rank;
    ierr = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRA(ierr);
  ierr = VecAssemblyEnd(u);CHKERRA(ierr);

  ierr = MatMult(C,u,b);CHKERRA(ierr);

  ierr = VecView(u,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = VecView(b,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-view_info",&flg);CHKERRA(ierr);
  if (flg)  {ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr);}
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix information (global sums):\n\
     nonzeros = %d, allocated nonzeros = %d\n",(int)info.nz_used,(int)info.nz_allocated);CHKERRA(ierr);
  ierr = MatGetInfo (C,MAT_GLOBAL_MAX,&info);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix information (global max):\n\
     nonzeros = %d, allocated nonzeros = %d\n",(int)info.nz_used,(int)info.nz_allocated);CHKERRA(ierr);

  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

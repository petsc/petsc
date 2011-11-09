
static char help[] = "Reads a matrix from PETSc binary file. Use for view or investigating matrix data structure. \n\n"; 
/*
 Example: 
      ./ex16 -f <matrix file> -a_mat_view_draw -draw_pause -1
      ./ex16 -f <matrix file> -a_mat_view_info
 */

#include <petscmat.h>
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                   A,Asp;          
  PetscViewer           fd;               /* viewer */
  char                  file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscErrorCode        ierr;
  PetscInt              m,n,rstart,rend;
  PetscBool             flg;
  PetscInt             row,ncols,j,nrows,nnzA=0,nnzAsp=0;
  const PetscInt       *cols;
  const PetscScalar    *vals;
  PetscReal            norm,percent,val,dtol=1.e-16;
  PetscMPIInt          rank;
  MatInfo              matinfo;
  PetscInt             Dnnz,Onnz;
  

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Determine files from which we read the linear systems. */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");

  /* Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /* Load the matrix; then destroy the viewer. */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetInfo(A,MAT_LOCAL,&matinfo);CHKERRQ(ierr);
  //printf("matinfo.nz_used %g\n",matinfo.nz_used);

  /* Get a sparse matrix Asp by dumping zero entries of A */
  ierr = MatCreate(PETSC_COMM_WORLD,&Asp);CHKERRQ(ierr);
  ierr = MatSetSizes(Asp,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(Asp,"asp_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(Asp);CHKERRQ(ierr);
  Dnnz  = (PetscInt)matinfo.nz_used/m + 1;
  Onnz  = Dnnz/2;
  printf("Dnnz %d %d\n",Dnnz,Onnz);
  ierr = MatSeqAIJSetPreallocation(Asp,Dnnz,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Asp,Dnnz,PETSC_NULL,Onnz,PETSC_NULL);CHKERRQ(ierr);
 
  /* Check zero rows */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  nrows = 0;
  for (row=rstart; row<rend; row++){
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    nnzA += ncols;
    norm = 0.0;
    for (j=0; j<ncols; j++){
      val = PetscAbsScalar(vals[j]);
      if (norm < val) norm = norm;
      if (val > dtol){
        ierr = MatSetValues(Asp,1,&row,1,&cols[j],&vals[j],INSERT_VALUES);CHKERRQ(ierr);
        nnzAsp++;
      }
    }
    if (!norm) nrows++;
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Asp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Asp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
 
  percent=(PetscReal)nnzA*100/(m*n);
  ierr = PetscPrintf(PETSC_COMM_SELF," [%d] Matrix A local size %d,%d; nnzA %d, %g percent; No. of zero rows: %d\n",rank,m,n,nnzA,percent,nrows);
  percent=(PetscReal)nnzAsp*100/(m*n);
  ierr = PetscPrintf(PETSC_COMM_SELF," [%d] Matrix Asp nnzAsp %d, %g percent\n",rank,nnzAsp,percent);

  /* investigate matcoloring for Asp */
  PetscBool     Asp_coloring = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-Asp_color",&Asp_coloring);CHKERRQ(ierr);
  if (Asp_coloring){
    ISColoring    iscoloring;
    MatFDColoring matfdcoloring;
    ierr = PetscPrintf(PETSC_COMM_WORLD," Create coloring of Asp...\n");
    ierr = MatGetColoring(Asp,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(Asp,iscoloring,&matfdcoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
    //ierr = MatFDColoringView(matfdcoloring,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringDestroy(&matfdcoloring);CHKERRQ(ierr);
  }

  /* Write Asp in binary for study - see ~petsc/src/mat/examples/tests/ex124.c */
  PetscBool Asp_write = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-Asp_write",&Asp_write);CHKERRQ(ierr);
  if (Asp_write){
    PetscViewer    viewer;
    ierr = PetscPrintf(PETSC_COMM_SELF,"Write Asp into file Asp.dat ...\n");
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Asp.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = MatView(Asp,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Asp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


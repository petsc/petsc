
static char help[] = "Reads a matrix from PETSc binary file. Use for view or investigating matrix data structure. \n\n";
/*
 Example:
      ./ex16 -f <matrix file> -a_mat_view draw -draw_pause -1
      ./ex16 -f <matrix file> -a_mat_view ascii::ascii_info
 */

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat               A,Asp;
  PetscViewer       fd;                        /* viewer */
  char              file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscInt          m,n,rstart,rend;
  PetscBool         flg;
  PetscInt          row,ncols,j,nrows,nnzA=0,nnzAsp=0;
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscReal         norm,percent,val,dtol=1.e-16;
  PetscMPIInt       rank;
  MatInfo           matinfo;
  PetscInt          Dnnz,Onnz;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Determine files from which we read the linear systems. */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file. */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  /* Load the matrix; then destroy the viewer. */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetOptionsPrefix(A,"a_"));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatGetSize(A,&m,&n));
  PetscCall(MatGetInfo(A,MAT_LOCAL,&matinfo));
  /*printf("matinfo.nz_used %g\n",matinfo.nz_used);*/

  /* Get a sparse matrix Asp by dumping zero entries of A */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&Asp));
  PetscCall(MatSetSizes(Asp,m,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetOptionsPrefix(Asp,"asp_"));
  PetscCall(MatSetFromOptions(Asp));
  Dnnz = (PetscInt)matinfo.nz_used/m + 1;
  Onnz = Dnnz/2;
  printf("Dnnz %d %d\n",Dnnz,Onnz);
  PetscCall(MatSeqAIJSetPreallocation(Asp,Dnnz,NULL));
  PetscCall(MatMPIAIJSetPreallocation(Asp,Dnnz,NULL,Onnz,NULL));
  /* The allocation above is approximate so we must set this option to be permissive.
   * Real code should preallocate exactly. */
  PetscCall(MatSetOption(Asp,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));

  /* Check zero rows */
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  nrows = 0;
  for (row=rstart; row<rend; row++) {
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    nnzA += ncols;
    norm  = 0.0;
    for (j=0; j<ncols; j++) {
      val = PetscAbsScalar(vals[j]);
      if (norm < val) norm = norm;
      if (val > dtol) {
        PetscCall(MatSetValues(Asp,1,&row,1,&cols[j],&vals[j],INSERT_VALUES));
        nnzAsp++;
      }
    }
    if (!norm) nrows++;
    PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
  }
  PetscCall(MatAssemblyBegin(Asp,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Asp,MAT_FINAL_ASSEMBLY));

  percent=(PetscReal)nnzA*100/(m*n);
  PetscCall(PetscPrintf(PETSC_COMM_SELF," [%d] Matrix A local size %d,%d; nnzA %d, %g percent; No. of zero rows: %d\n",rank,m,n,nnzA,percent,nrows));
  percent=(PetscReal)nnzAsp*100/(m*n);
  PetscCall(PetscPrintf(PETSC_COMM_SELF," [%d] Matrix Asp nnzAsp %d, %g percent\n",rank,nnzAsp,percent));

  /* investigate matcoloring for Asp */
  PetscBool Asp_coloring = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL,NULL,"-Asp_color",&Asp_coloring));
  if (Asp_coloring) {
    MatColoring   mc;
    ISColoring    iscoloring;
    MatFDColoring matfdcoloring;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Create coloring of Asp...\n"));
    PetscCall(MatColoringCreate(Asp,&mc));
    PetscCall(MatColoringSetType(mc,MATCOLORINGSL));
    PetscCall(MatColoringSetFromOptions(mc));
    PetscCall(MatColoringApply(mc,&iscoloring));
    PetscCall(MatColoringDestroy(&mc));
    PetscCall(MatFDColoringCreate(Asp,iscoloring,&matfdcoloring));
    PetscCall(MatFDColoringSetFromOptions(matfdcoloring));
    PetscCall(MatFDColoringSetUp(Asp,iscoloring,matfdcoloring));
    /*PetscCall(MatFDColoringView(matfdcoloring,PETSC_VIEWER_STDOUT_WORLD));*/
    PetscCall(ISColoringDestroy(&iscoloring));
    PetscCall(MatFDColoringDestroy(&matfdcoloring));
  }

  /* Write Asp in binary for study - see ~petsc/src/mat/tests/ex124.c */
  PetscBool Asp_write = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL,NULL,"-Asp_write",&Asp_write));
  if (Asp_write) {
    PetscViewer viewer;
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Write Asp into file Asp.dat ...\n"));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Asp.dat",FILE_MODE_WRITE,&viewer));
    PetscCall(MatView(Asp,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Asp));
  PetscCall(PetscFinalize());
  return 0;
}

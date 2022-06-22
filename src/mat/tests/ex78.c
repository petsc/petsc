
static char help[] = "Reads in a matrix in ASCII MATLAB format (I,J,A), read in vectors rhs and exact_solu in ASCII format.\n\
Writes them using the PETSc sparse format.\n\
Note: I and J start at 1, not 0, use -noshift if indices in file start with zero!\n\
Input parameters are:\n\
  -Ain  <filename> : input matrix in ascii format\n\
  -rhs  <filename> : input rhs in ascii format\n\
  -solu  <filename> : input true solution in ascii format\n\\n";

/*
Example: ./ex78 -Ain Ain -rhs rhs -solu solu -noshift -mat_view
 with the datafiles in the following format:
Ain (I and J start at 0):
------------------------
3 3 6
0 0 1.0
0 1 2.0
1 0 3.0
1 1 4.0
1 2 5.0
2 2 6.0

rhs
---
0 3.0
1 12.0
2 6.0

solu
----
0 1.0
0 1.0
0 1.0
*/

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A = NULL;
  Vec            b,u = NULL,u_tmp;
  char           Ain[PETSC_MAX_PATH_LEN],rhs[PETSC_MAX_PATH_LEN],solu[PETSC_MAX_PATH_LEN];
  int            m,n = 0,nz,dummy; /* these are fscaned so kept as int */
  PetscInt       i,col,row,shift = 1,sizes[3],nsizes;
  PetscScalar    val;
  PetscReal      res_norm;
  FILE           *Afile,*bfile,*ufile;
  PetscViewer    view;
  PetscBool      flg_A,flg_b,flg_u,flg;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Read in matrix, rhs and exact solution from ascii files */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-Ain",Ain,sizeof(Ain),&flg_A));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-noshift",&flg));
  if (flg) shift = 0;
  if (flg_A) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n Read matrix in ascii format ...\n"));
    PetscCall(PetscFOpen(PETSC_COMM_SELF,Ain,"r",&Afile));
    nsizes = 3;
    PetscCall(PetscOptionsGetIntArray(NULL,NULL,"-nosizesinfile",sizes,&nsizes,&flg));
    if (flg) {
      PetscCheck(nsizes == 3,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must pass in three m,n,nz as arguments for -nosizesinfile");
      m  = sizes[0];
      n  = sizes[1];
      nz = sizes[2];
    } else {
      PetscCheck(fscanf(Afile,"%d %d %d\n",&m,&n,&nz) == 3,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"m: %d, n: %d, nz: %d \n", m,n,nz));
    PetscCheck(m == n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Number of rows, cols must be same for this example");
    PetscCall(MatCreate(PETSC_COMM_SELF,&A));
    PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSeqAIJSetPreallocation(A,nz/m,NULL));
    PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

    for (i=0; i<nz; i++) {
      PetscCheck(fscanf(Afile,"%d %d %le\n",&row,&col,(double*)&val) == 3,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
      row -= shift; col -= shift;  /* set index set starts at 0 */
      PetscCall(MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    fclose(Afile);
  }

  PetscCall(PetscOptionsGetString(NULL,NULL,"-rhs",rhs,sizeof(rhs),&flg_b));
  if (flg_b) {
    PetscCall(VecCreate(PETSC_COMM_SELF,&b));
    PetscCall(VecSetSizes(b,PETSC_DECIDE,n));
    PetscCall(VecSetFromOptions(b));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n Read rhs in ascii format ...\n"));
    PetscCall(PetscFOpen(PETSC_COMM_SELF,rhs,"r",&bfile));
    for (i=0; i<n; i++) {
      PetscCheck(fscanf(bfile,"%d %le\n",&dummy,(double*)&val) == 2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
      PetscCall(VecSetValues(b,1,&i,&val,INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
    fclose(bfile);
  }

  PetscCall(PetscOptionsGetString(NULL,NULL,"-solu",solu,sizeof(solu),&flg_u));
  if (flg_u) {
    PetscCall(VecCreate(PETSC_COMM_SELF,&u));
    PetscCall(VecSetSizes(u,PETSC_DECIDE,n));
    PetscCall(VecSetFromOptions(u));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n Read exact solution in ascii format ...\n"));
    PetscCall(PetscFOpen(PETSC_COMM_SELF,solu,"r",&ufile));
    for (i=0; i<n; i++) {
      PetscCheck(fscanf(ufile,"%d  %le\n",&dummy,(double*)&val) == 2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
      PetscCall(VecSetValues(u,1,&i,&val,INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(u));
    PetscCall(VecAssemblyEnd(u));
    fclose(ufile);
  }

  /* Write matrix, rhs and exact solution in Petsc binary file */
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n Write matrix in binary to 'matrix.dat' ...\n"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF,"matrix.dat",FILE_MODE_WRITE,&view));
  PetscCall(MatView(A,view));

  if (flg_b) { /* Write rhs in Petsc binary file */
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n Write rhs in binary to 'matrix.dat' ...\n"));
    PetscCall(VecView(b,view));
  }
  if (flg_u) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n Write exact solution in binary to 'matrix.dat' ...\n"));
    PetscCall(VecView(u,view));
  }

  /* Check accuracy of the data */
  if (flg_A & flg_b & flg_u) {
    PetscCall(VecDuplicate(u,&u_tmp));
    PetscCall(MatMult(A,u,u_tmp));
    PetscCall(VecAXPY(u_tmp,-1.0,b));
    PetscCall(VecNorm(u_tmp,NORM_2,&res_norm));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n Accuracy of the reading data: | b - A*u |_2 : %g \n",res_norm));
    PetscCall(VecDestroy(&u_tmp));
  }

  PetscCall(MatDestroy(&A));
  if (flg_b) PetscCall(VecDestroy(&b));
  if (flg_u) PetscCall(VecDestroy(&u));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires:  !defined(PETSC_USE_64BIT_INDICES) double !complex

   test:
      requires: datafilespath
      args: -Ain ${DATAFILESPATH}/matrices/indefinite/afiro_A.dat

TEST*/

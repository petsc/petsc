
static char help[] = "Reads in a matrix in ASCII Matlab format (I,J,A), read in vectors rhs and exact_solu in ASCII format.\n\
Writes them using the PETSc sparse format.\n\
Note: I and J start at 1, not 0, use -noshift if indices in file start with zero!\n\
Input parameters are:\n\
  -Ain  <filename> : input matrix in ascii format\n\
  -rhs  <filename> : input rhs in ascii format\n\
  -solu  <filename> : input true solution in ascii format\n\\n";

/*
Example: ./ex78 -Ain Ain -rhs rhs -solu solu -noshift -mat_view
 with the datafiles in the followig format:
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

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  Vec            b,u,u_tmp;
  char           Ain[PETSC_MAX_PATH_LEN],rhs[PETSC_MAX_PATH_LEN],solu[PETSC_MAX_PATH_LEN]; 
  PetscErrorCode ierr;
  int            m,n,nz,dummy; /* these are fscaned so kept as int */
  PetscInt       i,col,row,shift = 1,sizes[3],nsizes;
  PetscScalar    val;
  PetscReal      res_norm;
  FILE           *Afile,*bfile,*ufile;
  PetscViewer    view;
  PetscTruth     flg_A,flg_b,flg_u,flg;
  PetscMPIInt    size;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* Read in matrix, rhs and exact solution from ascii files */
  ierr = PetscOptionsGetString(PETSC_NULL,"-Ain",Ain,PETSC_MAX_PATH_LEN-1,&flg_A);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-noshift",&flg);CHKERRQ(ierr);
  if (flg) shift = 0;
  if (flg_A){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read matrix in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,Ain,"r",&Afile);CHKERRQ(ierr); 
    nsizes = 3;
    ierr = PetscOptionsGetIntArray(PETSC_NULL,"-nosizesinfile",sizes,&nsizes,&flg);CHKERRQ(ierr);
    if (flg) {
      if (nsizes != 3) SETERRQ(1,"Must pass in three m,n,nz as arguments for -nosizesinfile");
      m = sizes[0];
      n = sizes[1];
      nz = sizes[2];
    } else {
      fscanf(Afile,"%d %d %d\n",&m,&n,&nz);
    }
    printf("m: %d, n: %d, nz: %d \n", m,n,nz);
    if (m != n) SETERRQ(PETSC_ERR_ARG_SIZ, "Number of rows, cols must be same for this example\n");
    ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,nz/m,PETSC_NULL);CHKERRQ(ierr);

    for (i=0; i<nz; i++) {
      fscanf(Afile,"%d %d %le\n",&row,&col,(double*)&val);
      row -= shift; col -= shift;  /* set index set starts at 0 */
      ierr = MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    fflush(stdout);
    fclose(Afile);  
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-rhs",rhs,PETSC_MAX_PATH_LEN-1,&flg_b);CHKERRQ(ierr);
  if (flg_b){
    ierr = VecCreate(PETSC_COMM_SELF,&b);CHKERRQ(ierr);
    ierr = VecSetSizes(b,PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(b);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read rhs in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,rhs,"r",&bfile);CHKERRQ(ierr); 
    for (i=0; i<n; i++) {      
      fscanf(bfile,"%d %le\n",&dummy,(double*)&val); 
      ierr = VecSetValues(b,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
    fflush(stdout);
    fclose(bfile);
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-solu",solu,PETSC_MAX_PATH_LEN-1,&flg_u);CHKERRQ(ierr);
  if (flg_u){
    ierr = VecCreate(PETSC_COMM_SELF,&u);CHKERRQ(ierr);
    ierr = VecSetSizes(u,PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(u);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read exact solution in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,solu,"r",&ufile);CHKERRQ(ierr); 
    for (i=0; i<n; i++) {
      fscanf(ufile,"%d  %le\n",&dummy,(double*)&val); 
      ierr = VecSetValues(u,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
    fflush(stdout);
    fclose(ufile);
  }
 
  /* Check accuracy of the data */
  if (flg_A & flg_b & flg_u){
    ierr = VecDuplicate(u,&u_tmp);CHKERRQ(ierr); 
    ierr = MatMult(A,u,u_tmp);CHKERRQ(ierr);
    ierr = VecAXPY(u_tmp,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRQ(ierr);
    printf("\n Accuracy of the reading data: | b - A*u |_2 : %g \n",res_norm); 

    /* Write the matrix, rhs and exact solution in Petsc binary file */
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Write matrix and rhs in binary to 'matrix.dat' ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"matrix.dat",FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = MatView(A,view);CHKERRQ(ierr);
    ierr = VecView(b,view);CHKERRQ(ierr);
    ierr = VecView(u,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(view);CHKERRQ(ierr);

    ierr = VecDestroy(b);CHKERRQ(ierr);
    ierr = VecDestroy(u);CHKERRQ(ierr);
    ierr = VecDestroy(u_tmp);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}


static char help[] = "Test MatMatMult() for AIJ and Dense matrices.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) 
{
  Mat            A,B,C,D;
  PetscInt       i,j,k,M=10,N=5;
  PetscScalar    *array,*a;
  PetscErrorCode ierr;
  PetscRandom    r;
  PetscTruth     equal=PETSC_FALSE;
  PetscReal      fill = 1.0;
  PetscInt       rstart,rend,nza,col,am,an,bm,bn;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-fill",&fill,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  /* create a aij matrix A */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,M);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  nza  = (PetscInt)(.3*M); /* num of nozeros in each row of A */
  ierr = MatSeqAIJSetPreallocation(A,nza,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,nza,PETSC_NULL,nza,PETSC_NULL);CHKERRQ(ierr);  
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc((nza+1)*sizeof(PetscScalar),&a);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) { 
    for (j=0; j<nza; j++) {
      ierr  = PetscRandomGetValue(r,&a[j]);CHKERRQ(ierr);
      col   = (PetscInt)(M*PetscRealPart(a[j]));
      ierr  = MatSetValues(A,1,&i,1,&col,&a[j],ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* create a dense matrix B */
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,am,N,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(B,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(B,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  ierr = MatGetArray(B,&array);CHKERRQ(ierr);
  k = 0;
  for (j=0; j<N; j++){ /* local column-wise entries */
    for (i=0; i<bm; i++){
      ierr = PetscRandomGetValue(r,&array[k++]);CHKERRQ(ierr); 
    }
  }
  ierr = MatRestoreArray(B,&array);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(r);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  /* Test MatMatMult() */
  ierr = MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);

  /*
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_MATLAB);
  ierr = MatView(C,0);CHKERRQ(ierr);
  ierr = MatView(B,0);CHKERRQ(ierr);
  ierr = MatView(A,0);CHKERRQ(ierr);
  */
  

  ierr = MatMatMultSymbolic(B,A,fill,&D);CHKERRQ(ierr);
  for (i=0; i<2; i++){    
    ierr = MatMatMultNumeric(B,A,D);CHKERRQ(ierr);
  }  
  ierr = MatEqual(C,D,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"C != D");

  ierr = MatDestroy(D);CHKERRQ(ierr); 
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  PetscFinalize();
  return(0);
}

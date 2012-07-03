
static char help[] = "Tests the vatious routines in MatMPIBAIJ format.\n";


#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  PetscInt       m=2,bs=1,M,row,col,start,end,i,j,k;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscScalar    data=100;
  PetscBool      flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Test MatSetValues() and MatGetValues() */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_size",&m,PETSC_NULL);CHKERRQ(ierr);

  M    = m*bs*size;
  ierr = MatCreateBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,M,M,PETSC_DECIDE,PETSC_NULL,PETSC_DECIDE,PETSC_NULL,&A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&start,&end);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-column_oriented",&flg);CHKERRQ(ierr);
  if (flg) { 
    ierr = MatSetOption(A,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); 
  }

  /* inproc assembly */
  for (row=start; row<end; row++) {
    for (col=start; col<end; col++,data+=1) {
      ierr = MatSetValues(A,1,&row,1,&col,&data,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* offproc assembly */
  data = 5.0;
  row = (M+start-1)%M;
  for (col=0; col<M; col++) {
    ierr = MatSetValues(A,1,&row,1,&col,&data,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatSetValuesBlocked() */
  ierr = PetscOptionsHasName(PETSC_NULL,"-test_setvaluesblocked",&flg);CHKERRQ(ierr);
  if (flg) { 
    PetscScalar *bval;
    row /= bs;
    col = start/bs;
    ierr = PetscMalloc(bs*bs*sizeof(PetscScalar),&bval);CHKERRQ(ierr);
    /* ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Set offproc blockvalues, blockrow %d, blockcol: %d\n",rank,row,col);CHKERRQ(ierr); */
    k = 1;
    /* row oriented - defalt */
    for (i=0; i<bs; i++){
      for (j=0; j<bs; j++){
        bval[i*bs+j] = (PetscScalar)k; k++;
      }
    }
    ierr = MatSetValuesBlocked(A,1,&row,1,&col,bval,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscFree(bval);CHKERRQ(ierr);
  }

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);
  ierr = PetscFinalize();
  return 0;
}
 

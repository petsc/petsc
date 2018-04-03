
static char help[] = "Tests the different MatColoring implementatons and ISColoringTestValid().\n\n";
#include <petscmat.h>

#if 0
#include <petscbt.h>

PetscErrorCode MatColoringVarify(Mat A,ISColoring iscoloring)
{
  PetscErrorCode ierr;
  PetscInt       nn,c,i,j,M,N,nc,nnz,col,row;
  const PetscInt *cia,*cja;
  IS             *isis;
  Mat            Aseq = A;
  MPI_Comm       comm;
  PetscMPIInt    size,rank;
  PetscBool      done;
  PetscBT        table;
  const PetscInt *cols;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,&nn,&isis);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    /* create a sequential iscoloring on all processors */
    ierr = MatGetSeqNonzeroStructure(A,&Aseq);CHKERRQ(ierr);
  }
  ierr = MatGetSize(Aseq,&M,NULL);CHKERRQ(ierr);
  printf("MatColoringVarify...nn %d, M %d\n",nn,M);

  ierr = MatGetColumnIJ(Aseq,0,PETSC_FALSE,PETSC_TRUE,&N,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = PetscBTCreate(M,&table);CHKERRQ(ierr);
  for (c=0; c<nn; c++) { /* for each color */
    ierr = ISGetSize(isis[c],&nc);CHKERRQ(ierr);
    if (nc <= 1) continue;

    ierr = PetscBTMemzero(M,table);CHKERRQ(ierr);
    //printf("\n %d ---- isis\n",c);

    ierr = ISGetIndices(isis[c],&cols);CHKERRQ(ierr);
    for (j=0; j<nc; j++) {
      //PetscInt nnz,col,row;
      col = cols[j];
      nnz = cia[col+1] - cia[col];
      //printf("    col %d, nnz %d, rows:\n",col,nnz);
      for (i=0; i<nnz; i++) {
        row = cja[cia[col]+i];
        //printf(" %d,",row);
        if (PetscBTLookupSet(table,row)) {
          SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"color %D, col %D: row %D already in this color",c,col,row);
        }
      }
      //printf(" \n");
    }
    ierr = ISRestoreIndices(isis[c],&cols);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&table);CHKERRQ(ierr);

  if (size > 1) {
    ierr = MatDestroy(&Aseq);CHKERRQ(ierr);
  }
  ierr = MatRestoreColumnIJ(Aseq,1,PETSC_FALSE,PETSC_TRUE,NULL,&cia,&cja,&done);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode FormJacobian(Vec input,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       vecsize,ownbegin,ownend,i,j;
  PetscScalar    dummy=0.0;

  PetscFunctionBeginUser;
  ierr = VecGetSize(input,&vecsize);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(input,&ownbegin,&ownend);CHKERRQ(ierr);

  for (i=ownbegin; i<ownend; i++) {
    for(j=i-3;j<i+3;j++) {
      if(j<vecsize){
        ierr = MatSetValues(A,1,&i,1,&j,&dummy,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  Mat            J;
  Vec            solution,residual;
  PetscMPIInt    size;
  PetscInt       M=8;
  ISColoring     iscoloring;
  MatColoring    coloring;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr= MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr= MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, M, M);CHKERRQ(ierr);
  ierr= MatSetFromOptions(J);CHKERRQ(ierr);
  ierr= MatSetUp(J);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&solution);CHKERRQ(ierr);
  ierr = VecSetSizes(solution,PETSC_DECIDE,M);CHKERRQ(ierr);
  ierr = VecSetFromOptions(solution);CHKERRQ(ierr);
  ierr = VecDuplicate(solution,&residual);CHKERRQ(ierr);

  ierr = FormJacobian(solution,J,J,NULL);CHKERRQ(ierr);
  ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
    Color the matrix, i.e. determine groups of columns that share no common
    rows. These columns in the Jacobian can all be computed simultaneously.
   */
  ierr = MatColoringCreate(J, &coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGGREEDY);CHKERRQ(ierr);
  ierr = MatColoringSetDistance(coloring,2);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring, &iscoloring);CHKERRQ(ierr);

  if (size == 1) {
    ierr = ISColoringTestValid(J,iscoloring);CHKERRQ(ierr);
  }

  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&solution);CHKERRQ(ierr);
  ierr = VecDestroy(&residual);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

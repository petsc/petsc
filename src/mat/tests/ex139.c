
const char help[] = "Test MatCreateLocalRef()\n\n";

#include <petscmat.h>

static PetscErrorCode GetLocalRef(Mat A,IS isrow,IS iscol,Mat *B)
{
  IS             istmp;

  PetscFunctionBegin;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Extracting LocalRef with isrow:\n"));
  CHKERRQ(ISOnComm(isrow,PETSC_COMM_WORLD,PETSC_COPY_VALUES,&istmp));
  CHKERRQ(ISView(istmp,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISDestroy(&istmp));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Extracting LocalRef with iscol (only rank=0 shown):\n"));
  CHKERRQ(ISOnComm(iscol,PETSC_COMM_WORLD,PETSC_COPY_VALUES,&istmp));
  CHKERRQ(ISView(istmp,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISDestroy(&istmp));

  CHKERRQ(MatCreateLocalRef(A,isrow,iscol,B));
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  PetscErrorCode         ierr;
  MPI_Comm               comm;
  Mat                    J,B;
  PetscInt               i,j,k,l,rstart,rend,m,n,top_bs,row_bs,col_bs,nlocblocks,*idx,nrowblocks,ncolblocks,*ridx,*cidx,*icol,*irow;
  PetscScalar            *vals;
  ISLocalToGlobalMapping brmap;
  IS                     is0,is1;
  PetscBool              diag,blocked;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  ierr = PetscOptionsBegin(comm,NULL,"LocalRef Test Options",NULL);CHKERRQ(ierr);
  {
    top_bs = 2; row_bs = 2; col_bs = 2; diag = PETSC_FALSE; blocked = PETSC_FALSE;
    CHKERRQ(PetscOptionsInt("-top_bs","Block size of top-level matrix",0,top_bs,&top_bs,NULL));
    CHKERRQ(PetscOptionsInt("-row_bs","Block size of row map",0,row_bs,&row_bs,NULL));
    CHKERRQ(PetscOptionsInt("-col_bs","Block size of col map",0,col_bs,&col_bs,NULL));
    CHKERRQ(PetscOptionsBool("-diag","Extract a diagonal black",0,diag,&diag,NULL));
    CHKERRQ(PetscOptionsBool("-blocked","Use block insertion",0,blocked,&blocked,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(MatCreate(comm,&J));
  CHKERRQ(MatSetSizes(J,6,6,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetBlockSize(J,top_bs));
  CHKERRQ(MatSetFromOptions(J));
  CHKERRQ(MatSeqBAIJSetPreallocation(J,top_bs,PETSC_DECIDE,0));
  CHKERRQ(MatMPIBAIJSetPreallocation(J,top_bs,PETSC_DECIDE,0,PETSC_DECIDE,0));
  CHKERRQ(MatSetUp(J));
  CHKERRQ(MatGetSize(J,&m,&n));
  CHKERRQ(MatGetOwnershipRange(J,&rstart,&rend));

  nlocblocks = (rend-rstart)/top_bs + 2;
  CHKERRQ(PetscMalloc1(nlocblocks,&idx));
  for (i=0; i<nlocblocks; i++) {
    idx[i] = (rstart/top_bs + i - 1 + m/top_bs) % (m/top_bs);
  }
  CHKERRQ(ISLocalToGlobalMappingCreate(comm,top_bs,nlocblocks,idx,PETSC_OWN_POINTER,&brmap));
  CHKERRQ(PetscPrintf(comm,"Block ISLocalToGlobalMapping:\n"));
  CHKERRQ(ISLocalToGlobalMappingView(brmap,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatSetLocalToGlobalMapping(J,brmap,brmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&brmap));

  /* Create index sets for local submatrix */
  nrowblocks = (rend-rstart)/row_bs;
  ncolblocks = (rend-rstart)/col_bs;
  CHKERRQ(PetscMalloc2(nrowblocks,&ridx,ncolblocks,&cidx));
  for (i=0; i<nrowblocks; i++) ridx[i] = i + ((i > nrowblocks/2) ^ !rstart);
  for (i=0; i<ncolblocks; i++) cidx[i] = i + ((i < ncolblocks/2) ^ !!rstart);
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,row_bs,nrowblocks,ridx,PETSC_COPY_VALUES,&is0));
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,col_bs,ncolblocks,cidx,PETSC_COPY_VALUES,&is1));
  CHKERRQ(PetscFree2(ridx,cidx));

  if (diag) {
    CHKERRQ(ISDestroy(&is1));
    CHKERRQ(PetscObjectReference((PetscObject)is0));
    is1        = is0;
    ncolblocks = nrowblocks;
  }

  CHKERRQ(GetLocalRef(J,is0,is1,&B));

  CHKERRQ(MatZeroEntries(J));

  CHKERRQ(PetscMalloc3(row_bs,&irow,col_bs,&icol,row_bs*col_bs,&vals));
  for (i=0; i<nrowblocks; i++) {
    for (j=0; j<ncolblocks; j++) {
      for (k=0; k<row_bs; k++) {
        for (l=0; l<col_bs; l++) {
          vals[k*col_bs+l] = i * 100000 + j * 1000 + k * 10 + l;
        }
        irow[k] = i*row_bs+k;
      }
      for (l=0; l<col_bs; l++) icol[l] = j*col_bs+l;
      if (blocked) {
        CHKERRQ(MatSetValuesBlockedLocal(B,1,&i,1,&j,vals,ADD_VALUES));
      } else {
        CHKERRQ(MatSetValuesLocal(B,row_bs,irow,col_bs,icol,vals,ADD_VALUES));
      }
    }
  }
  CHKERRQ(PetscFree3(irow,icol,vals));

  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatView(J,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISDestroy(&is0));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&J));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      filter: grep -v "type: mpi"
      args: -blocked {{0 1}} -mat_type {{aij baij}}

TEST*/

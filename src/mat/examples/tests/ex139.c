
const char help[] = "Test MatCreateLocalRef()\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "GetLocalRef"
static PetscErrorCode GetLocalRef(Mat A,IS isrow,IS iscol,Mat *B)
{
  PetscErrorCode ierr;
  IS istmp;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Extracting LocalRef with isrow:\n");CHKERRQ(ierr);
  ierr = ISOnComm(isrow,PETSC_COMM_WORLD,PETSC_COPY_VALUES,&istmp);CHKERRQ(ierr);
  ierr = ISView(istmp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&istmp);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Extracting LocalRef with iscol (only rank=0 shown):\n");CHKERRQ(ierr);
  ierr = ISOnComm(iscol,PETSC_COMM_WORLD,PETSC_COPY_VALUES,&istmp);CHKERRQ(ierr);
  ierr = ISView(istmp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&istmp);CHKERRQ(ierr);

  ierr = MatCreateLocalRef(A,isrow,iscol,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  PetscErrorCode         ierr;
  MPI_Comm               comm;
  Mat                    J,B;
  PetscInt               i,j,k,l,rstart,rend,m,n,top_bs,row_bs,col_bs,nlocblocks,*idx,nrowblocks,ncolblocks,*ridx,*cidx,*icol,*irow;
  PetscScalar            *vals;
  ISLocalToGlobalMapping brmap,rmap;
  IS                     is0,is1;
  PetscBool              diag,blocked;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = PetscOptionsBegin(comm,PETSC_NULL,"LocalRef Test Options",PETSC_NULL);CHKERRQ(ierr);
  {
    top_bs = 2; row_bs = 2; col_bs = 2; diag = PETSC_FALSE; blocked = PETSC_FALSE;
    ierr = PetscOptionsInt("-top_bs","Block size of top-level matrix",0,top_bs,&top_bs,0);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-row_bs","Block size of row map",0,row_bs,&row_bs,0);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-col_bs","Block size of col map",0,col_bs,&col_bs,0);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-diag","Extract a diagonal black",0,diag,&diag,0);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-blocked","Use block insertion",0,blocked,&blocked,0);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = MatCreate(comm,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,6,6,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSize(J,top_bs);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(J,top_bs,PETSC_DECIDE,0);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(J,top_bs,PETSC_DECIDE,0,PETSC_DECIDE,0);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(J,&rstart,&rend);CHKERRQ(ierr);

  nlocblocks = (rend-rstart)/top_bs + 2;
  ierr = PetscMalloc(nlocblocks*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  for (i=0; i<nlocblocks; i++) {
    idx[i] = (rstart/top_bs + i - 1 + m/top_bs) % (m/top_bs);
  }
  ierr = ISLocalToGlobalMappingCreate(comm,nlocblocks,idx,PETSC_OWN_POINTER,&brmap);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Block ISLocalToGlobalMapping:\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(brmap,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMappingBlock(J,brmap,brmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingUnBlock(brmap,top_bs,&rmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,rmap,rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&brmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rmap);CHKERRQ(ierr);

  /* Create index sets for local submatrix */
  nrowblocks = (rend-rstart)/row_bs;
  ncolblocks = (rend-rstart)/col_bs;
  ierr = PetscMalloc2(nrowblocks,PetscInt,&ridx,ncolblocks,PetscInt,&cidx);CHKERRQ(ierr);
  for (i=0; i<nrowblocks; i++) ridx[i] = i + ((i > nrowblocks/2) ^ !rstart);
  for (i=0; i<ncolblocks; i++) cidx[i] = i + ((i < ncolblocks/2) ^ !!rstart);
  ierr = ISCreateBlock(PETSC_COMM_SELF,row_bs,nrowblocks,ridx,PETSC_COPY_VALUES,&is0);CHKERRQ(ierr);
  ierr = ISCreateBlock(PETSC_COMM_SELF,col_bs,ncolblocks,cidx,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  ierr = PetscFree2(ridx,cidx);CHKERRQ(ierr);

  if (diag) {
    ierr = ISDestroy(&is1);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)is0);CHKERRQ(ierr);
    is1 = is0;
    ncolblocks = nrowblocks;
  }

  ierr = GetLocalRef(J,is0,is1,&B);CHKERRQ(ierr);

  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  ierr = PetscMalloc3(row_bs,PetscInt,&irow,col_bs,PetscInt,&icol,row_bs*col_bs,PetscScalar,&vals);CHKERRQ(ierr);
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
        ierr = MatSetValuesBlockedLocal(B,1,&i,1,&j,vals,ADD_VALUES);CHKERRQ(ierr);
      } else {
        ierr = MatSetValuesLocal(B,row_bs,irow,col_bs,icol,vals,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree3(irow,icol,vals);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDestroy(&is0);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

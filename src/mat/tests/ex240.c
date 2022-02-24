static char help[] ="Tests MatFDColoringSetValues()\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM                     da;
  PetscErrorCode         ierr;
  PetscInt               N, mx = 5,my = 4,i,j,nc,nrow,n,ncols,rstart,*colors,*map;
  const PetscInt         *cols;
  const PetscScalar      *vals;
  Mat                    A,B;
  PetscReal              norm;
  MatFDColoring          fdcoloring;
  ISColoring             iscoloring;
  PetscScalar            *cm;
  const ISColoringValue  *icolors;
  PetscMPIInt            rank;
  ISLocalToGlobalMapping ltog;
  PetscBool              single,two;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,mx,my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateMatrix(da,&A));

  /* as a test copy the matrices from the standard format to the compressed format; this is not scalable but is ok because just for testing */
  /*    first put the coloring in the global ordering */
  CHKERRQ(DMCreateColoring(da,IS_COLORING_LOCAL,&iscoloring));
  CHKERRQ(ISColoringGetColors(iscoloring,&n,&nc,&icolors));
  CHKERRQ(DMGetLocalToGlobalMapping(da,&ltog));
  CHKERRQ(PetscMalloc1(n,&map));
  for (i=0; i<n; i++) map[i] = i;
  CHKERRQ(ISLocalToGlobalMappingApply(ltog,n,map,map));
  CHKERRQ(MatGetSize(A,&N,NULL));
  CHKERRQ(PetscMalloc1(N,&colors));
  for (i=0; i<N; i++) colors[i] = -1;
  for (i=0; i<n; i++) colors[map[i]]= icolors[i];
  CHKERRQ(PetscFree(map));
  CHKERRQ(PetscSynchronizedPrintf(MPI_COMM_WORLD,"[%d]Global colors \n",rank));
  for (i=0; i<N; i++) CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " %" PetscInt_FMT "\n",i,colors[i]));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));

  /*   second, compress the A matrix */
  CHKERRQ(MatSetRandom(A,NULL));
  CHKERRQ(MatView(A,NULL));
  CHKERRQ(MatGetLocalSize(A,&nrow,NULL));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,NULL));
  CHKERRQ(PetscCalloc1(nrow*nc,&cm));
  for (i=0; i<nrow; i++) {
    CHKERRQ(MatGetRow(A,rstart+i,&ncols,&cols,&vals));
    for (j=0; j<ncols; j++) {
      PetscCheckFalse(colors[cols[j]] < 0,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Global column %" PetscInt_FMT " had no color",cols[j]);
      cm[i + nrow*colors[cols[j]]] = vals[j];
    }
    CHKERRQ(MatRestoreRow(A,rstart+i,&ncols,&cols,&vals));
  }

  /* print compressed matrix */
  CHKERRQ(PetscSynchronizedPrintf(MPI_COMM_WORLD,"[%d] Compressed matrix \n",rank));
  for (i=0; i<nrow; i++) {
    for (j=0; j<nc; j++) {
      CHKERRQ(PetscSynchronizedPrintf(MPI_COMM_WORLD,"%12.4e  ",(double)PetscAbsScalar(cm[i+nrow*j])));
    }
    CHKERRQ(PetscSynchronizedPrintf(MPI_COMM_WORLD,"\n"));
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));

  /* put the compressed matrix into the standard matrix */
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  CHKERRQ(MatZeroEntries(A));
  CHKERRQ(MatView(B,0));
  CHKERRQ(MatFDColoringCreate(A,iscoloring,&fdcoloring));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-single_block",&single));
  if (single) {
    CHKERRQ(MatFDColoringSetBlockSize(fdcoloring,PETSC_DEFAULT,nc));
  }
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-two_block",&two));
  if (two) {
    CHKERRQ(MatFDColoringSetBlockSize(fdcoloring,PETSC_DEFAULT,2));
  }
  CHKERRQ(MatFDColoringSetFromOptions(fdcoloring));
  CHKERRQ(MatFDColoringSetUp(A,iscoloring,fdcoloring));

  CHKERRQ(MatFDColoringSetValues(A,fdcoloring,cm));
  CHKERRQ(MatView(A,NULL));

  /* check the values were put in the correct locations */
  CHKERRQ(MatAXPY(A,-1.0,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatView(A,NULL));
  CHKERRQ(MatNorm(A,NORM_FROBENIUS,&norm));
  if (norm > PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix is not identical, problem with MatFDColoringSetValues()\n"));
  }
  CHKERRQ(PetscFree(colors));
  CHKERRQ(PetscFree(cm));
  CHKERRQ(ISColoringDestroy(&iscoloring));
  CHKERRQ(MatFDColoringDestroy(&fdcoloring));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      requires: !complex

   test:
      suffix: single
      requires: !complex
      nsize: 2
      args: -single_block
      output_file: output/ex240_1.out

   test:
      suffix: two
      requires: !complex
      nsize: 2
      args: -two_block
      output_file: output/ex240_1.out

   test:
      suffix: 2
      requires: !complex
      nsize: 5

TEST*/

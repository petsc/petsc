static char help[] ="Tests MatFDColoringSetValues()\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM                     da;
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,mx,my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateMatrix(da,&A));

  /* as a test copy the matrices from the standard format to the compressed format; this is not scalable but is ok because just for testing */
  /*    first put the coloring in the global ordering */
  PetscCall(DMCreateColoring(da,IS_COLORING_LOCAL,&iscoloring));
  PetscCall(ISColoringGetColors(iscoloring,&n,&nc,&icolors));
  PetscCall(DMGetLocalToGlobalMapping(da,&ltog));
  PetscCall(PetscMalloc1(n,&map));
  for (i=0; i<n; i++) map[i] = i;
  PetscCall(ISLocalToGlobalMappingApply(ltog,n,map,map));
  PetscCall(MatGetSize(A,&N,NULL));
  PetscCall(PetscMalloc1(N,&colors));
  for (i=0; i<N; i++) colors[i] = -1;
  for (i=0; i<n; i++) colors[map[i]]= icolors[i];
  PetscCall(PetscFree(map));
  PetscCall(PetscSynchronizedPrintf(MPI_COMM_WORLD,"[%d]Global colors \n",rank));
  for (i=0; i<N; i++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " %" PetscInt_FMT "\n",i,colors[i]));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));

  /*   second, compress the A matrix */
  PetscCall(MatSetRandom(A,NULL));
  PetscCall(MatView(A,NULL));
  PetscCall(MatGetLocalSize(A,&nrow,NULL));
  PetscCall(MatGetOwnershipRange(A,&rstart,NULL));
  PetscCall(PetscCalloc1(nrow*nc,&cm));
  for (i=0; i<nrow; i++) {
    PetscCall(MatGetRow(A,rstart+i,&ncols,&cols,&vals));
    for (j=0; j<ncols; j++) {
      PetscCheck(colors[cols[j]] >= 0,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Global column %" PetscInt_FMT " had no color",cols[j]);
      cm[i + nrow*colors[cols[j]]] = vals[j];
    }
    PetscCall(MatRestoreRow(A,rstart+i,&ncols,&cols,&vals));
  }

  /* print compressed matrix */
  PetscCall(PetscSynchronizedPrintf(MPI_COMM_WORLD,"[%d] Compressed matrix \n",rank));
  for (i=0; i<nrow; i++) {
    for (j=0; j<nc; j++) {
      PetscCall(PetscSynchronizedPrintf(MPI_COMM_WORLD,"%12.4e  ",(double)PetscAbsScalar(cm[i+nrow*j])));
    }
    PetscCall(PetscSynchronizedPrintf(MPI_COMM_WORLD,"\n"));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));

  /* put the compressed matrix into the standard matrix */
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  PetscCall(MatZeroEntries(A));
  PetscCall(MatView(B,0));
  PetscCall(MatFDColoringCreate(A,iscoloring,&fdcoloring));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-single_block",&single));
  if (single) PetscCall(MatFDColoringSetBlockSize(fdcoloring,PETSC_DEFAULT,nc));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-two_block",&two));
  if (two) PetscCall(MatFDColoringSetBlockSize(fdcoloring,PETSC_DEFAULT,2));
  PetscCall(MatFDColoringSetFromOptions(fdcoloring));
  PetscCall(MatFDColoringSetUp(A,iscoloring,fdcoloring));

  PetscCall(MatFDColoringSetValues(A,fdcoloring,cm));
  PetscCall(MatView(A,NULL));

  /* check the values were put in the correct locations */
  PetscCall(MatAXPY(A,-1.0,B,SAME_NONZERO_PATTERN));
  PetscCall(MatView(A,NULL));
  PetscCall(MatNorm(A,NORM_FROBENIUS,&norm));
  if (norm > PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Matrix is not identical, problem with MatFDColoringSetValues()\n"));
  }
  PetscCall(PetscFree(colors));
  PetscCall(PetscFree(cm));
  PetscCall(ISColoringDestroy(&iscoloring));
  PetscCall(MatFDColoringDestroy(&fdcoloring));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
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

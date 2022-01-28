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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,mx,my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);

  /* as a test copy the matrices from the standard format to the compressed format; this is not scalable but is ok because just for testing */
  /*    first put the coloring in the global ordering */
  ierr = DMCreateColoring(da,IS_COLORING_LOCAL,&iscoloring);CHKERRQ(ierr);
  ierr = ISColoringGetColors(iscoloring,&n,&nc,&icolors);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&map);CHKERRQ(ierr);
  for (i=0; i<n; i++) map[i] = i;
  ierr = ISLocalToGlobalMappingApply(ltog,n,map,map);CHKERRQ(ierr);
  ierr = MatGetSize(A,&N,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(N,&colors);CHKERRQ(ierr);
  for (i=0; i<N; i++) colors[i] = -1;
  for (i=0; i<n; i++) colors[map[i]]= icolors[i];
  ierr = PetscFree(map);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(MPI_COMM_WORLD,"[%d]Global colors \n",rank);CHKERRQ(ierr);
  for (i=0; i<N; i++) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " %" PetscInt_FMT "\n",i,colors[i]);CHKERRQ(ierr);}
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout);CHKERRQ(ierr);

  /*   second, compress the A matrix */
  ierr = MatSetRandom(A,NULL);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&nrow,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(nrow*nc,&cm);CHKERRQ(ierr);
  for (i=0; i<nrow; i++) {
    ierr = MatGetRow(A,rstart+i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++) {
      if (colors[cols[j]] < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Global column %" PetscInt_FMT " had no color",cols[j]);
      cm[i + nrow*colors[cols[j]]] = vals[j];
    }
    ierr = MatRestoreRow(A,rstart+i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }

  /* print compressed matrix */
  ierr = PetscSynchronizedPrintf(MPI_COMM_WORLD,"[%d] Compressed matrix \n",rank);CHKERRQ(ierr);
  for (i=0; i<nrow; i++) {
    for (j=0; j<nc; j++) {
      ierr = PetscSynchronizedPrintf(MPI_COMM_WORLD,"%12.4e  ",(double)PetscAbsScalar(cm[i+nrow*j]));CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(MPI_COMM_WORLD,"\n");CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout);CHKERRQ(ierr);

  /* put the compressed matrix into the standard matrix */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatView(B,0);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(A,iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-single_block",&single);CHKERRQ(ierr);
  if (single) {
    ierr = MatFDColoringSetBlockSize(fdcoloring,PETSC_DEFAULT,nc);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(NULL,NULL,"-two_block",&two);CHKERRQ(ierr);
  if (two) {
    ierr = MatFDColoringSetBlockSize(fdcoloring,PETSC_DEFAULT,2);CHKERRQ(ierr);
  }
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetUp(A,iscoloring,fdcoloring);CHKERRQ(ierr);

  ierr = MatFDColoringSetValues(A,fdcoloring,cm);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);

  /* check the values were put in the correct locations */
  ierr = MatAXPY(A,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  if (norm > PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix is not identical, problem with MatFDColoringSetValues()\n");CHKERRQ(ierr);
  }
  ierr = PetscFree(colors);CHKERRQ(ierr);
  ierr = PetscFree(cm);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&fdcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
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

static char help[] = "Tests MatPermute() in parallel.\n\n";
/* Results:
   Sequential:
   - seqaij:   correct permutation
   - seqbaij:  permutation not supported for this MATTYPE
   - seqsbaij: permutation not supported for this MATTYPE
   Parallel:
   - mpiaij:   correct permutation
   - mpibaij:  correct permutation
   - mpisbaij: permutation not supported for this MATTYPE
 */

#include <petscmat.h>

int main(int argc,char **argv)
{
  const struct {PetscInt i,j; PetscScalar v;} entries[] = {{0,3,1.},{1,2,2.},{2,1,3.},{2,5,4.},{3,0,5.},{3,6,6.},{4,1,7.},{4,4,8.}};
  const PetscInt ixrow[5]                               = {4,2,1,0,3},ixcol[7] = {5,3,6,1,2,0,4};
  Mat            A,B;
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend,cstart,cend;
  IS             isrow,iscol;
  PetscViewer    viewer;
  PetscBool      view_sparse;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* ------- Assemble matrix, --------- */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,5,7));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  CHKERRQ(MatGetOwnershipRangeColumn(A,&cstart,&cend));

  for (i=0; i<(PetscInt)(sizeof(entries)/sizeof(entries[0])); i++) {
    CHKERRQ(MatSetValue(A,entries[i].i,entries[i].j,entries[i].v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* ------ Prepare index sets ------ */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,rend-rstart,ixrow+rstart,PETSC_USE_POINTER,&isrow));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,cend-cstart,ixcol+cstart,PETSC_USE_POINTER,&iscol));
  CHKERRQ(ISSetPermutation(isrow));
  CHKERRQ(ISSetPermutation(iscol));

  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  view_sparse = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-view_sparse", &view_sparse, NULL));
  if (!view_sparse) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_DENSE));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Original matrix\n"));
  CHKERRQ(MatView(A,viewer));

  CHKERRQ(MatPermute(A,isrow,iscol,&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Permuted matrix\n"));
  CHKERRQ(MatView(B,viewer));

  if (!view_sparse) {
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Row permutation\n"));
  CHKERRQ(ISView(isrow,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Column permutation\n"));
  CHKERRQ(ISView(iscol,viewer));

  /* Free data structures */
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -view_sparse

   test:
      suffix: 2
      nsize: 2
      args: -view_sparse

   test:
      suffix: 2b
      nsize: 2
      args: -mat_type baij -view_sparse

   test:
      suffix: 3
      nsize: 3
      args: -view_sparse

   test:
      suffix: 3b
      nsize: 3
      args: -mat_type baij -view_sparse

   test:
      suffix: dense
      args: -mat_type dense

TEST*/

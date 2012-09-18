static char help[] = "Tests MatPermute() in parallel.\n\n";
/* Results:
   Sequential:
   - seqaij:   correct permutation
   - seqbaij:  permutation not supported for this MATTYPE
   - seqsbaij: permutation not supported for this MATTYPE
   Parallel:
   - mpiaij:   incorrect permutation (disable this op for now).
   - mpibaij:  weird error indicating that it can't handle rectangular matrices (disable this op for now).
   - mpisbaij: permutation not supported for this MATTYPE
 */
#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  const struct {PetscInt i,j; PetscScalar v;} entries[] = {{0,3,1.},{1,2,2.},{2,1,3.},{2,5,4.},{3,0,5.},{3,6,6.},{4,1,7.},{4,4,8.}};
  const PetscInt ixrow[5] = {4,2,1,3,0},ixcol[7] = {5,3,6,1,2,0,4};
  Mat            A,B;
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend,cstart,cend;
  IS             isrow,iscol;
  PetscViewer    viewer,sviewer;
  PetscBool      view_sparse;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  /* ------- Assemble matrix, --------- */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,5,7);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(A,&cstart,&cend);CHKERRQ(ierr);

  for (i=0; i<(PetscInt)(sizeof(entries)/sizeof(entries[0])); i++) {
    ierr = MatSetValue(A,entries[i].i,entries[i].j,entries[i].v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* ------ Prepare index sets ------ */
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,rend-rstart,ixrow+rstart,PETSC_USE_POINTER,&isrow);CHKERRQ(ierr);
  /* The column index set must currently be  */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,7,ixcol,PETSC_USE_POINTER,&iscol);CHKERRQ(ierr);
  ierr = ISSetPermutation(isrow);CHKERRQ(ierr);
  ierr = ISSetPermutation(iscol);CHKERRQ(ierr);

  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  view_sparse = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL, "-view_sparse", &view_sparse, PETSC_NULL); CHKERRQ(ierr);
  if (!view_sparse) {
    ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_DENSE);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"Original matrix\n");CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);

  ierr = MatPermute(A,isrow,iscol,&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Permuted matrix\n");CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"Row permutation\n");CHKERRQ(ierr);
  ierr = ISView(isrow,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Column permutation\n");CHKERRQ(ierr);
  ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
  ierr = ISView(iscol,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);

  /* Free data structures */
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}


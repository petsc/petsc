static char help[] = "Tests MatPermute() for a square matrix in parallel.\n\n";
#include <petscmat.h>
/* Results:
   Sequential:
   - seqaij:   correct permutation
   - seqbaij:  permutation not supported for this MATTYPE
   - seqsbaij: permutation not supported for this MATTYPE
   Parallel:
   - mpiaij:   incorrect permutation (disable this op for now)
   - seqbaij:  correct permutation, but all manner of memory leaks
               (disable this op for now, because it appears broken for nonsquare matrices; see ex151)
   - seqsbaij: permutation not supported for this MATTYPE

 */
int main(int argc,char **argv)
{
  const struct {PetscInt i,j; PetscScalar v;} entries[] = {{0,3,1.},{1,2,2.},{2,1,3.},{2,4,4.},{3,0,5.},{3,3,6.},{4,1,7.},{4,4,8.}};
  const PetscInt ixrow[5]                               = {4,2,1,3,0},ixcol[5] = {3,2,1,4,0};
  Mat            A,B;
  PetscInt       i,rstart,rend,cstart,cend;
  IS             isrow,iscol;
  PetscViewer    viewer,sviewer;
  PetscBool      view_sparse;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  /* ------- Assemble matrix, --------- */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,5,5));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  PetscCall(MatGetOwnershipRangeColumn(A,&cstart,&cend));

  for (i=0; i<(PetscInt)PETSC_STATIC_ARRAY_LENGTH(entries); i++) {
    PetscCall(MatSetValue(A,entries[i].i,entries[i].j,entries[i].v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* ------ Prepare index sets ------ */
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,rend-rstart,ixrow+rstart,PETSC_USE_POINTER,&isrow));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,5,ixcol,PETSC_USE_POINTER,&iscol));
  PetscCall(ISSetPermutation(isrow));
  PetscCall(ISSetPermutation(iscol));

  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  view_sparse = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-view_sparse", &view_sparse, NULL));
  if (!view_sparse) {
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_DENSE));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,"Original matrix\n"));
  PetscCall(MatView(A,viewer));

  PetscCall(MatPermute(A,isrow,iscol,&B));
  PetscCall(PetscViewerASCIIPrintf(viewer,"Permuted matrix\n"));
  PetscCall(MatView(B,viewer));
  if (!view_sparse) {
    PetscCall(PetscViewerPopFormat(viewer));
  }

  PetscCall(PetscViewerASCIIPrintf(viewer,"Row permutation\n"));
  PetscCall(ISView(isrow,viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"Column permutation\n"));
  PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  PetscCall(ISView(iscol,sviewer));
  PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));

  /* Free data structures */
  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));

  PetscCall(PetscFinalize());
  return 0;
}

static char help[] = "Test MatSeqAIJSetValuesBatch: setting batches of elements using the GPU.\n\
This works with SeqAIJCUSP matrices.\n\n";
#include <petscdmda.h>

/* We will use a structured mesh for this assembly test. Each square will be divided into two triangles:
  C       D
   _______
  |\      | The matrix for 0 and 1 is /   1  -0.5 -0.5 \
  | \   1 |                           | -0.5  0.5  0.0 |
  |  \    |                           \ -0.5  0.0  0.5 /
  |   \   |
  |    \  |
  |  0  \ |
  |      \|
  ---------
  A       B
 */

PetscErrorCode IntegrateCells(DM dm, PetscInt *Ne, PetscInt *Nl, PetscInt *N, PetscInt **elemRows, PetscScalar **elemMats) {
  DMDALocalInfo  info;
  PetscInt      *er;
  PetscScalar   *em;
  PetscInt       X, Y, dof;
  PetscInt       nl, ne;
  PetscInt       k  = 0, m  = 0;
  PetscInt       i, j;
  PetscLogEvent  integrationEvent;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventRegister("ElemIntegration", DM_CLASSID, &integrationEvent);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(integrationEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm, 0, &X, &Y,0,0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  nl   = dof*3;
  ne   = 2 * (X-1) * (Y-1);
  *N   = X * Y * dof;
  *Ne  = ne;
  *Nl  = nl;
  ierr = PetscMalloc2(ne*nl, PetscInt, elemRows, ne*nl*nl, PetscScalar, elemMats);CHKERRQ(ierr);
  er   = *elemRows;
  em   = *elemMats;
  for(j = info.ys; j < info.ys+info.ym-1; ++j) {
    for(i = info.xs; i < info.xs+info.xm-1; ++i) {
      PetscInt rowA = j*X     + i, rowB = j*X     + i+1;
      PetscInt rowC = (j+1)*X + i, rowD = (j+1)*X + i+1;

      /* Lower triangle */
      er[k+0] = rowA; em[m+0*nl+0] =  1.0; em[m+0*nl+1] = -0.5; em[m+0*nl+2] = -0.5;
      er[k+1] = rowB; em[m+1*nl+0] = -0.5; em[m+1*nl+1] =  0.5; em[m+1*nl+2] =  0.0;
      er[k+2] = rowC; em[m+2*nl+0] = -0.5; em[m+2*nl+1] =  0.0; em[m+2*nl+2] =  0.5;
      k += nl; m += nl*nl;
      /* Upper triangle */
      er[k+0] = rowD; em[m+0*nl+0] =  1.0; em[m+0*nl+1] = -0.5; em[m+0*nl+2] = -0.5;
      er[k+1] = rowC; em[m+1*nl+0] = -0.5; em[m+1*nl+1] =  0.5; em[m+1*nl+2] =  0.0;
      er[k+2] = rowB; em[m+2*nl+0] = -0.5; em[m+2*nl+1] =  0.0; em[m+2*nl+2] =  0.5;
      k += nl; m += nl*nl;
    }
  }
  ierr = PetscLogEventEnd(integrationEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AssembleMatrix(DM dm, PetscInt Ne, PetscInt Nl, PetscInt *elemRows, PetscScalar *elemMats, Mat *A) {
  PetscInt       e;
  PetscLogEvent  assemblyEvent;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventRegister("ElemAssembly", DM_CLASSID, &assemblyEvent);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetMatrix(dm, MATAIJ, A);CHKERRQ(ierr);
  ierr = MatZeroEntries(*A);CHKERRQ(ierr);
  for(e = 0; e < Ne; ++e) {
    ierr = MatSetValues(*A, Nl, &elemRows[Nl*e], Nl, &elemRows[Nl*e], &elemMats[Nl*Nl*e], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  Mat            A;
  PetscViewer    viewer;
  PetscInt       Nl, Ne, N;
  PetscInt      *elemRows;
  PetscScalar   *elemMats;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, 0, help);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX, -3, -3, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &dm);CHKERRQ(ierr);
  ierr = IntegrateCells(dm, &Ne, &Nl, &N, &elemRows, &elemMats);CHKERRQ(ierr);
  /* Construct matrix using GPU */
  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);CHKERRQ(ierr);
  ierr = MatSetType(A, MATSEQAIJCUSP);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, 0, PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetValuesBatch(A, Ne, Nl, N, elemRows, elemMats);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, PETSC_NULL, &viewer);CHKERRQ(ierr);
  if (N > 500) {ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);}
  ierr = MatView(A, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  /* Construct matrix using CPU */
  ierr = AssembleMatrix(dm, Ne, Nl, elemRows, elemMats, &A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, PETSC_NULL, &viewer);CHKERRQ(ierr);
  if (N > 500) {ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);}
  ierr = MatView(A, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  /* Cleanup */
  ierr = PetscFree2(elemRows, elemMats);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

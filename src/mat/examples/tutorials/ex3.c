static char help[] = "Test MatSeqAIJSetValuesBatch: setting batches of elements using the GPU.\n\
This works with SeqAIJCUSP matrices.\n\n";
#include <petscmat.h>

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

PetscErrorCode IntegrateCells(PetscInt *Nl, PetscInt *Ne, PetscInt *N, PetscInt **elemRows, PetscScalar **elemMats) {
  PetscInt      *er;
  PetscScalar   *em;
  PetscInt       nl = 3, ne = 2 /*triangles*/ * 2 /*x size-1*/ * 2 /*y size-1*/;
  PetscInt       k  = 0, m  = 0;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(ne*nl, PetscInt, elemRows, ne*nl*nl, PetscScalar, elemMats);CHKERRQ(ierr);
  *N  = 3 /*x size*/ * 3 /*y size*/;
  *Ne = ne;
  *Nl = nl;
  er  = *elemRows;
  em  = *elemMats;
  for(j = 0; j < 2; ++j) {
    for(i = 0; i < 2; ++i) {
      PetscInt rowA = j*nl     + i, rowB = j*nl     + i+1;
      PetscInt rowC = (j+1)*nl + i, rowD = (j+1)*nl + i+1;

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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  Mat            A;
  PetscInt       Nl, Ne, N;
  PetscInt      *elemRows;
  PetscScalar   *elemMats;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, 0, help);CHKERRQ(ierr);
  ierr = MatCreate(MPI_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = IntegrateCells(&Nl, &Ne, &N, &elemRows, &elemMats);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);CHKERRQ(ierr);
  ierr = MatSetType(A, MATSEQAIJCUSP);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, 0, PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetValuesBatch(A, Ne, Nl, N, elemRows, elemMats);CHKERRQ(ierr);
  ierr = PetscFree2(elemRows, elemMats);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

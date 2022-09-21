static char help[] = "Demonstrates the use of the COO interface to PETSc matrices for finite element computations\n\n";

/*
     The COO interface for PETSc matrices provides a convenient way to provide finite element element stiffness matrices to PETSc matrix that should work
   well on both CPUs and GPUs. It is an alternative to using MatSetValues()

     This example is intended for people who are NOT using DMPLEX or libCEED or any other higher-level infrastructure for finite elements;
   it is only to demonstrate the concepts in a simple way for those people who are interested and for those people who are using PETSc for
   linear algebra solvers but are managing their own finite element process.

     Please do NOT use this example as a starting point to writing your own finite element code from scratch!

     Each element in this example has three vertices; hence the the usage below needs to be adjusted for elements of a different number of vertices.
*/

#include <petscmat.h>
#include "ex18.h"

static PetscErrorCode CreateFEStruct(FEStruct *fe)
{
  PetscFunctionBeginUser;
  fe->Nv = 5;
  fe->Ne = 3;
  PetscCall(PetscMalloc1(3 * fe->Ne, &fe->vertices));
  /* the three vertices associated with each element in order of element */
  fe->vertices[0 + 0] = 0;
  fe->vertices[0 + 1] = 1;
  fe->vertices[0 + 2] = 2;
  fe->vertices[3 + 0] = 2;
  fe->vertices[3 + 1] = 1;
  fe->vertices[3 + 2] = 3;
  fe->vertices[6 + 0] = 2;
  fe->vertices[6 + 1] = 4;
  fe->vertices[6 + 2] = 3;
  fe->n               = 5;
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyFEStruct(FEStruct *fe)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(fe->vertices));
  PetscCall(PetscFree(fe->coo));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMatrix(FEStruct *fe, Mat *A)
{
  PetscInt *oor, *ooc, cnt = 0;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, fe->n, fe->n, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(*A));

  /* determine for each entry in each element stiffness matrix the global row and colum */
  /* since the element is triangular with piecewise linear basis functions there are three degrees of freedom per element, one for each vertex */
  PetscCall(PetscMalloc2(3 * 3 * fe->Ne, &oor, 3 * 3 * fe->Ne, &ooc));
  for (PetscInt e = 0; e < fe->Ne; e++) {
    for (PetscInt vi = 0; vi < 3; vi++) {
      for (PetscInt vj = 0; vj < 3; vj++) {
        oor[cnt]   = fe->vertices[3 * e + vi];
        ooc[cnt++] = fe->vertices[3 * e + vj];
      }
    }
  }
  PetscCall(MatSetPreallocationCOO(*A, 3 * 3 * fe->Ne, oor, ooc));
  PetscCall(PetscFree2(oor, ooc));

  /* determine the offset into the COO value array the offset of each element stiffness; there are 9 = 3*3 entries for each element stiffness */
  /* for lists of elements with different numbers of degrees of freedom assocated with each element the offsets will not be uniform */
  PetscCall(PetscMalloc1(fe->Ne, &fe->coo));
  fe->coo[0] = 0;
  for (PetscInt e = 1; e < fe->Ne; e++) fe->coo[e] = fe->coo[e - 1] + 3 * 3;
  PetscFunctionReturn(0);
}

static PetscErrorCode FillMatrixCPU(FEStruct *fe, Mat A)
{
  PetscScalar s[9];

  PetscFunctionBeginUser;
  /* simulation of traditional PETSc CPU based finite assembly process */
  for (PetscInt e = 0; e < fe->Ne; e++) {
    for (PetscInt vi = 0; vi < 3; vi++) {
      for (PetscInt vj = 0; vj < 3; vj++) s[3 * vi + vj] = vi + 2 * vj;
    }
    PetscCall(MatSetValues(A, 3, fe->vertices + 3 * e, 3, fe->vertices + 3 * e, s, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
   Shows an example of tracking element offsets explicitly, which allows for
   mixed-topology meshes and combining both volume and surface parts into the weak form.
*/
static PetscErrorCode FillMatrixCPUCOO(FEStruct *fe, Mat A)
{
  PetscScalar *v, *s;

  PetscFunctionBeginUser;
  /* simulation of CPU based finite assembly process with COO */
  PetscCall(PetscMalloc1(3 * 3 * fe->Ne, &v));
  for (PetscInt e = 0; e < fe->Ne; e++) {
    s = v + fe->coo[e]; /* point to location in COO of current element stiffness */
    for (PetscInt vi = 0; vi < 3; vi++) {
      for (PetscInt vj = 0; vj < 3; vj++) s[3 * vi + vj] = vi + 2 * vj;
    }
  }
  PetscCall(MatSetValuesCOO(A, v, ADD_VALUES));
  PetscCall(PetscFree(v));
  PetscFunctionReturn(0);
}

/*
  Uses a multi-dimensional indexing technique that works for homogeneous meshes
  such as single-topology with volume integral only.
*/
static PetscErrorCode FillMatrixCPUCOO3d(FEStruct *fe, Mat A)
{
  PetscScalar(*s)[3][3];

  PetscFunctionBeginUser;
  /* simulation of CPU based finite assembly process with COO */
  PetscCall(PetscMalloc1(fe->Ne, &s));
  for (PetscInt e = 0; e < fe->Ne; e++) {
    for (PetscInt vi = 0; vi < 3; vi++) {
      for (PetscInt vj = 0; vj < 3; vj++) s[e][vi][vj] = vi + 2 * vj;
    }
  }
  PetscCall(MatSetValuesCOO(A, (PetscScalar *)s, INSERT_VALUES));
  PetscCall(PetscFree(s));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  Mat         A;
  FEStruct    fe;
  PetscMPIInt size;
  PetscBool   is_kokkos, is_cuda;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size <= 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Demonstration is only for sequential runs");

  PetscCall(CreateFEStruct(&fe));
  PetscCall(CreateMatrix(&fe, &A));

  PetscCall(FillMatrixCPU(&fe, A));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatZeroEntries(A));
  PetscCall(FillMatrixCPUCOO(&fe, A));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatZeroEntries(A));
  PetscCall(FillMatrixCPUCOO3d(&fe, A));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatZeroEntries(A));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJKOKKOS, &is_kokkos));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &is_cuda));
#if defined(PETSC_HAVE_KOKKOS)
  if (is_kokkos) PetscCall(FillMatrixKokkosCOO(&fe, A));
#endif
#if defined(PETSC_HAVE_CUDA)
  if (is_cuda) PetscCall(FillMatrixCUDACOO(&fe, A));
#endif
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(DestroyFEStruct(&fe));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: cuda kokkos_kernels
    depends: ex18cu.cu ex18kok.kokkos.cxx

  testset:
    filter: grep -v "type"
    output_file: output/ex18_1.out

    test:
      suffix: kok
      requires: kokkos_kernels
      args: -mat_type aijkokkos

    test:
      suffix: cuda
      requires: cuda
      args: -mat_type aijcusparse

TEST*/

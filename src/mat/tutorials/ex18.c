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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  fe->Nv = 5;
  fe->Ne = 3;
  ierr = PetscMalloc1(3*fe->Ne,&fe->vertices);CHKERRQ(ierr);
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
  fe->n  = 5;
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyFEStruct(FEStruct *fe)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(fe->vertices);CHKERRQ(ierr);
  ierr = PetscFree(fe->coo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMatrix(FEStruct *fe,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       *oor,*ooc,cnt = 0;

  PetscFunctionBeginUser;
  ierr = MatCreate(PETSC_COMM_WORLD,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,fe->n,fe->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);

  /* determine for each entry in each element stiffness matrix the global row and colum */
  /* since the element is triangular with piecewise linear basis functions there are three degrees of freedom per element, one for each vertex */
  ierr = PetscMalloc2(3*3*fe->Ne,&oor,3*3*fe->Ne,&ooc);CHKERRQ(ierr);
  for (PetscInt e=0; e<fe->Ne; e++) {
    for (PetscInt vi=0; vi<3; vi++) {
      for (PetscInt vj=0; vj<3; vj++) {
        oor[cnt]   = fe->vertices[3*e+vi];
        ooc[cnt++] = fe->vertices[3*e+vj];
      }
    }
  }
  ierr = MatSetPreallocationCOO(*A,3*3*fe->Ne,oor,ooc);CHKERRQ(ierr);
  ierr = PetscFree2(oor,ooc);CHKERRQ(ierr);

  /* determine the offset into the COO value array the offset of each element stiffness; there are 9 = 3*3 entries for each element stiffness */
  /* for lists of elements with different numbers of degrees of freedom assocated with each element the offsets will not be uniform */
  ierr = PetscMalloc1(fe->Ne,&fe->coo);CHKERRQ(ierr);
  fe->coo[0] = 0;
  for (PetscInt e=1; e<fe->Ne; e++) {
    fe->coo[e] = fe->coo[e-1] + 3*3;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FillMatrixCPU(FEStruct *fe,Mat A)
{
  PetscErrorCode ierr;
  PetscScalar    s[9];

  PetscFunctionBeginUser;
  /* simulation of traditional PETSc CPU based finite assembly process */
  for (PetscInt e=0; e<fe->Ne; e++) {
    for (PetscInt vi=0; vi<3; vi++) {
      for (PetscInt vj=0; vj<3; vj++) {
        s[3*vi+vj] = vi+2*vj;
      }
    }
    ierr = MatSetValues(A,3,fe->vertices + 3*e,3, fe->vertices + 3*e,s,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Shows an example of tracking element offsets explicitly, which allows for
   mixed-topology meshes and combining both volume and surface parts into the weak form.
*/
static PetscErrorCode FillMatrixCPUCOO(FEStruct *fe,Mat A)
{
  PetscErrorCode ierr;
  PetscScalar    *v,*s;

  PetscFunctionBeginUser;
  /* simulation of CPU based finite assembly process with COO */
  ierr = PetscMalloc1(3*3*fe->Ne,&v);CHKERRQ(ierr);
  for (PetscInt e=0; e<fe->Ne; e++) {
    s = v + fe->coo[e]; /* point to location in COO of current element stiffness */
    for (PetscInt vi=0; vi<3; vi++) {
      for (PetscInt vj=0; vj<3; vj++) {
        s[3*vi+vj] = vi+2*vj;
      }
    }
  }
  ierr = MatSetValuesCOO(A,v,ADD_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Uses a multi-dimensional indexing technique that works for homogeneous meshes
  such as single-topology with volume integral only.
*/
static PetscErrorCode FillMatrixCPUCOO3d(FEStruct *fe,Mat A)
{
  PetscErrorCode ierr;
  PetscScalar    (*s)[3][3];

  PetscFunctionBeginUser;
  /* simulation of CPU based finite assembly process with COO */
  ierr = PetscMalloc1(fe->Ne,&s);CHKERRQ(ierr);
  for (PetscInt e=0; e<fe->Ne; e++) {
    for (PetscInt vi=0; vi<3; vi++) {
      for (PetscInt vj=0; vj<3; vj++) {
        s[e][vi][vj] = vi+2*vj;
      }
    }
  }
  ierr = MatSetValuesCOO(A,(PetscScalar*)s,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  Mat             A;
  PetscErrorCode  ierr;
  FEStruct        fe;
  PetscMPIInt     size;
  PetscBool       is_kokkos,is_cuda;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheck(size <= 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Demonstration is only for sequential runs");

  ierr = CreateFEStruct(&fe);CHKERRQ(ierr);
  ierr = CreateMatrix(&fe,&A);CHKERRQ(ierr);

  ierr = FillMatrixCPU(&fe,A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = FillMatrixCPUCOO(&fe,A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = FillMatrixCPUCOO3d(&fe,A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJKOKKOS,&is_kokkos);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&is_cuda);CHKERRQ(ierr);
 #if defined(PETSC_HAVE_KOKKOS)
  if (is_kokkos) {ierr = FillMatrixKokkosCOO(&fe,A);CHKERRQ(ierr);}
 #endif
 #if defined(PETSC_HAVE_CUDA)
  if (is_cuda) {ierr = FillMatrixCUDACOO(&fe,A);CHKERRQ(ierr);}
 #endif
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DestroyFEStruct(&fe);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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

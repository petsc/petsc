static char help[] = "Tests MATHTOOL with a derived htool::IMatrix<PetscScalar> class\n\n";

#include <petscmat.h>
#include <htool/misc/petsc.hpp>

static PetscErrorCode GenEntries(PetscInt sdim,PetscInt M,PetscInt N,const PetscInt *J,const PetscInt *K,PetscScalar *ptr,void *ctx)
{
  PetscInt  d,j,k;
  PetscReal diff = 0.0,*coords = (PetscReal*)(ctx);

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      diff = 0.0;
      for (d = 0; d < sdim; d++) diff += (coords[J[j]*sdim+d] - coords[K[k]*sdim+d]) * (coords[J[j]*sdim+d] - coords[K[k]*sdim+d]);
      ptr[j+M*k] = 1.0/(1.0e-2 + PetscSqrtReal(diff));
    }
  }
  PetscFunctionReturn(0);
}
class MyIMatrix : public htool::VirtualGenerator<PetscScalar> {
  private:
  PetscReal *coords;
  PetscInt  sdim;
  public:
  MyIMatrix(PetscInt M,PetscInt N,PetscInt spacedim,PetscReal* gcoords) : htool::VirtualGenerator<PetscScalar>(M,N),coords(gcoords),sdim(spacedim) { }

  void copy_submatrix(PetscInt M,PetscInt N,const PetscInt *J,const PetscInt *K,PetscScalar *ptr) const override
  {
    PetscReal diff = 0.0;

    PetscFunctionBeginUser;
    for (PetscInt j = 0; j < M; j++) /* could be optimized by the user how they see fit, e.g., vectorization */
      for (PetscInt k = 0; k < N; k++) {
        diff = 0.0;
        for (PetscInt d = 0; d < sdim; d++) diff += (coords[J[j]*sdim+d] - coords[K[k]*sdim+d]) * (coords[J[j]*sdim+d] - coords[K[k]*sdim+d]);
        ptr[j+M*k] = 1.0/(1.0e-2 + PetscSqrtReal(diff));
      }
    PetscFunctionReturnVoid();
  }
};

int main(int argc,char **argv)
{
  Mat            A,B,P,R;
  PetscInt       m = 100,dim = 3,M,begin = 0;
  PetscMPIInt    size;
  PetscReal      *coords,*gcoords,norm,epsilon,relative;
  PetscBool      sym = PETSC_FALSE;
  PetscRandom    rdm;
  MatHtoolKernel kernel = GenEntries;
  MyIMatrix      *imatrix;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mat_htool_epsilon",&epsilon,NULL));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  M = size*m;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscMalloc1(m*dim,&coords));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomGetValuesReal(rdm,m*dim,coords));
  CHKERRQ(PetscCalloc1(M*dim,&gcoords));
  CHKERRMPI(MPI_Exscan(&m,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  CHKERRQ(PetscArraycpy(gcoords+begin*dim,coords,m*dim));
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
  imatrix = new MyIMatrix(M,M,dim,gcoords);
  CHKERRQ(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,NULL,imatrix,&A)); /* block-wise assembly using htool::IMatrix<PetscScalar>::copy_submatrix() */
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,sym));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(A,NULL,"-A_view"));
  CHKERRQ(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&B)); /* entry-wise assembly using GenEntries() */
  CHKERRQ(MatSetOption(B,MAT_SYMMETRIC,sym));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(B,NULL,"-B_view"));
  CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&P));
  CHKERRQ(MatNorm(P,NORM_FROBENIUS,&relative));
  CHKERRQ(MatConvert(B,MATDENSE,MAT_INITIAL_MATRIX,&R));
  CHKERRQ(MatAXPY(R,-1.0,P,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(R,NORM_INFINITY,&norm));
  PetscCheckFalse(PetscAbsReal(norm/relative) > epsilon,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A(!symmetric)-A(symmetric)|| = %g (> %g)",(double)PetscAbsReal(norm/relative),(double)epsilon);
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&R));
  CHKERRQ(MatDestroy(&P));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFree(gcoords));
  CHKERRQ(PetscFree(coords));
  delete imatrix;
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: htool
   test:
      requires: htool
      suffix: 2
      nsize: 4
      args: -m_local 120 -mat_htool_epsilon 1.0e-2 -symmetric {{false true}shared output}
      output_file: output/ex101.out

TEST*/

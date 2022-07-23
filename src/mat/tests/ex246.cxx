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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)NULL,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mat_htool_epsilon",&epsilon,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  M = size*m;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscMalloc1(m*dim,&coords));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  PetscCall(PetscRandomGetValuesReal(rdm,m*dim,coords));
  PetscCall(PetscCalloc1(M*dim,&gcoords));
  PetscCallMPI(MPI_Exscan(&m,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscArraycpy(gcoords+begin*dim,coords,m*dim));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
  imatrix = new MyIMatrix(M,M,dim,gcoords);
  PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,NULL,imatrix,&A)); /* block-wise assembly using htool::IMatrix<PetscScalar>::copy_submatrix() */
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,sym));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A,NULL,"-A_view"));
  PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&B)); /* entry-wise assembly using GenEntries() */
  PetscCall(MatSetOption(B,MAT_SYMMETRIC,sym));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(B,NULL,"-B_view"));
  PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&P));
  PetscCall(MatNorm(P,NORM_FROBENIUS,&relative));
  PetscCall(MatConvert(B,MATDENSE,MAT_INITIAL_MATRIX,&R));
  PetscCall(MatAXPY(R,-1.0,P,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(R,NORM_INFINITY,&norm));
  PetscCheck(PetscAbsReal(norm/relative) <= epsilon,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A(!symmetric)-A(symmetric)|| = %g (> %g)",(double)PetscAbsReal(norm/relative),(double)epsilon);
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&P));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(gcoords));
  PetscCall(PetscFree(coords));
  delete imatrix;
  PetscCall(PetscFinalize());
  return 0;
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

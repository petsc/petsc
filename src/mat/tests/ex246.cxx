static char help[] = "Tests MATHTOOL with a derived htool::IMatrix<PetscScalar> class\n\n";

#include <petscmat.h>
#include <htool/misc/petsc.hpp>

static PetscScalar GenEntry(PetscInt sdim,PetscInt i,PetscInt j,void *ctx)
{
  PetscInt  d;
  PetscReal diff = 0.0,*coords = (PetscReal*)(ctx);

  PetscFunctionBeginUser;
  for (d = 0; d < sdim; d++) { diff += (coords[i*sdim+d] - coords[j*sdim+d]) * (coords[i*sdim+d] - coords[j*sdim+d]); }
  PetscFunctionReturn(1.0/(1.0e-2 + PetscSqrtReal(diff)));
}

class MyIMatrix : public htool::IMatrix<PetscScalar> {
  private:
  PetscReal *coords;
  PetscInt  sdim;
  public:
  MyIMatrix(PetscInt M,PetscInt N,PetscInt spacedim,PetscReal* gcoords) : htool::IMatrix<PetscScalar>(M,N),coords(gcoords),sdim(spacedim) { }

  PetscScalar get_coef(const PetscInt &i, const PetscInt &j) const override
  {
    PetscReal diff = 0.0;

    PetscFunctionBeginUser;
    for (PetscInt d = 0; d < sdim; d++) { diff += (coords[i*sdim+d] - coords[j*sdim+d]) * (coords[i*sdim+d] - coords[j*sdim+d]); }
    PetscFunctionReturn(1.0/(1.0e-2 + PetscSqrtReal(diff)));
  }

  void copy_submatrix(PetscInt M, PetscInt N, const PetscInt *const rows, const PetscInt *const cols, PetscScalar *ptr) const override
  {
    PetscFunctionBeginUser;
    for (PetscInt j = 0; j < M; j++) /* could be optimized by the user how they see fit, e.g., vectorization */
      for (PetscInt k = 0; k < N; k++) ptr[j+k*M] = this->get_coef(rows[j],cols[k]);
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
  MatHtoolKernel kernel = GenEntry;
  MyIMatrix      *imatrix;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mat_htool_epsilon",&epsilon,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  M = size*m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(m*dim,&coords);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomGetValuesReal(rdm,m*dim,coords);CHKERRQ(ierr);
  ierr = PetscCalloc1(M*dim,&gcoords);CHKERRQ(ierr);
  ierr = MPI_Exscan(&m,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = PetscArraycpy(gcoords+begin*dim,coords,m*dim);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  imatrix = new MyIMatrix(M,M,dim,gcoords);
  ierr = MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,NULL,imatrix,&A);CHKERRQ(ierr); /* block-wise assembly using htool::IMatrix<PetscScalar>::copy_submatrix() */
  ierr = MatSetOption(A,MAT_SYMMETRIC,sym);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);
  ierr = MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&B);CHKERRQ(ierr); /* entry-wise assembly using GenEntry() */
  ierr = MatSetOption(B,MAT_SYMMETRIC,sym);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(B,NULL,"-B_view");CHKERRQ(ierr);
  ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&P);CHKERRQ(ierr);
  ierr = MatNorm(P,NORM_FROBENIUS,&relative);CHKERRQ(ierr);
  ierr = MatConvert(B,MATDENSE,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr);
  ierr = MatAXPY(R,-1.0,P,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(R,NORM_INFINITY,&norm);CHKERRQ(ierr);
  if (PetscAbsReal(norm/relative) > epsilon) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A(!symmetric)-A(symmetric)|| = %g (> %g)",(double)PetscAbsReal(norm/relative),(double)epsilon);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree(gcoords);CHKERRQ(ierr);
  ierr = PetscFree(coords);CHKERRQ(ierr);
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

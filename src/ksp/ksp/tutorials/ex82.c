#include <petscksp.h>

static char help[] = "Solves a linear system using PCHPDDM and MATHTOOL.\n\n";

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

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
  Vec            b,x;
  Mat            A;
  PetscInt       m = 100,dim = 3,M,begin = 0,n = 0,overlap = 1;
  PetscMPIInt    size;
  PetscReal      *coords,*gcoords;
  MatHtoolKernel kernel = GenEntries;
  PetscBool      flg,sym = PETSC_FALSE;
  PetscRandom    rdm;

  PetscCall(PetscInitialize(&argc,&argv,(char*)NULL,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-overlap",&overlap,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  M = size*m;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscMalloc1(m*dim,&coords));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  PetscCall(PetscRandomGetValuesReal(rdm,m*dim,coords));
  PetscCall(PetscCalloc1(M*dim,&gcoords));
  PetscCallMPI(MPI_Exscan(&m,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscArraycpy(gcoords+begin*dim,coords,m*dim));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&A));
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,sym));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A,&b,&x));
  PetscCall(VecSetRandom(b,rdm));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCHPDDM,&flg));
  if (flg) {
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    Mat aux;
    IS  is;
    PetscCall(MatGetOwnershipRange(A,&begin,&n));
    n -= begin;
    PetscCall(ISCreateStride(PETSC_COMM_SELF,n,begin,1,&is));
    PetscCall(MatIncreaseOverlap(A,1,&is,overlap));
    PetscCall(ISGetLocalSize(is,&n));
    PetscCall(MatCreateDense(PETSC_COMM_SELF,n,n,n,n,NULL,&aux));
    PetscCall(MatSetOption(aux,MAT_SYMMETRIC,sym));
    PetscCall(MatAssemblyBegin(aux,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(aux,MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(aux,1.0)); /* just the local identity matrix, not very meaningful numerically, but just testing that the necessary plumbing is there */
    PetscCall(PCHPDDMSetAuxiliaryMat(pc,is,aux,NULL,NULL));
    PetscCall(ISDestroy(&is));
    PetscCall(MatDestroy(&aux));
#endif
  }
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(gcoords));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: htool hpddm

   test:
      requires: htool hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 4
      # different numbers of iterations depending on PetscScalar type
      filter: sed -e "s/symmetry: S/symmetry: N/g" -e "/number of dense/d" -e "s/Linear solve converged due to CONVERGED_RTOL iterations 13/Linear solve converged due to CONVERGED_RTOL iterations 18/g"
      args: -ksp_view -ksp_converged_reason -mat_htool_epsilon 1e-2 -m_local 200 -pc_type hpddm -pc_hpddm_define_subdomains -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 1 -pc_hpddm_coarse_pc_type lu -pc_hpddm_levels_1_eps_gen_non_hermitian -symmetric {{false true}shared output} -overlap 2
      output_file: output/ex82_1.out

TEST*/

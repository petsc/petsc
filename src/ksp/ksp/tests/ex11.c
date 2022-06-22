static const char help[] = "Solves a Q1-P0 Stokes problem from Underworld.\n\
\n\
You can obtain a sample matrix from http://ftp.mcs.anl.gov/pub/petsc/Datafiles/matrices/underworld32.gz\n\
and run with -f underworld32.gz\n\n";

#include <petscksp.h>
#include <petscdmda.h>

static PetscErrorCode replace_submats(Mat A, IS isu, IS isp)
{
  Mat            A11,A22,A12,A21;
  Mat            nA11,nA22,nA12,nA21;
  const char     *prefix;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSubMatrix(A,isu,isu,MAT_INITIAL_MATRIX,&A11));
  PetscCall(MatCreateSubMatrix(A,isu,isp,MAT_INITIAL_MATRIX,&A12));
  PetscCall(MatCreateSubMatrix(A,isp,isu,MAT_INITIAL_MATRIX,&A21));
  PetscCall(MatCreateSubMatrix(A,isp,isp,MAT_INITIAL_MATRIX,&A22));
  PetscCall(MatDuplicate(A11,MAT_COPY_VALUES,&nA11));
  PetscCall(MatDuplicate(A12,MAT_COPY_VALUES,&nA12));
  PetscCall(MatDuplicate(A21,MAT_COPY_VALUES,&nA21));
  PetscCall(MatDuplicate(A22,MAT_COPY_VALUES,&nA22));
  PetscCall(MatGetOptionsPrefix(A11,&prefix));
  PetscCall(MatSetOptionsPrefix(nA11,prefix));
  PetscCall(MatGetOptionsPrefix(A22,&prefix));
  PetscCall(MatSetOptionsPrefix(nA22,prefix));
  PetscCall(MatNestSetSubMat(A,0,0,nA11));
  PetscCall(MatNestSetSubMat(A,0,1,nA12));
  PetscCall(MatNestSetSubMat(A,1,0,nA21));
  PetscCall(MatNestSetSubMat(A,1,1,nA22));
  PetscCall(MatDestroy(&A11));
  PetscCall(MatDestroy(&A12));
  PetscCall(MatDestroy(&A21));
  PetscCall(MatDestroy(&A22));
  PetscCall(MatDestroy(&nA11));
  PetscCall(MatDestroy(&nA12));
  PetscCall(MatDestroy(&nA21));
  PetscCall(MatDestroy(&nA22));
  PetscFunctionReturn(0);
}

PetscErrorCode LSCLoadTestOperators(Mat *A11,Mat *A12,Mat *A21,Mat *A22,Vec *b1,Vec *b2)
{
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD,A11));
  PetscCall(MatCreate(PETSC_COMM_WORLD,A12));
  PetscCall(MatCreate(PETSC_COMM_WORLD,A21));
  PetscCall(MatCreate(PETSC_COMM_WORLD,A22));
  PetscCall(MatSetOptionsPrefix(*A11,"a11_"));
  PetscCall(MatSetOptionsPrefix(*A22,"a22_"));
  PetscCall(MatSetFromOptions(*A11));
  PetscCall(MatSetFromOptions(*A22));
  PetscCall(VecCreate(PETSC_COMM_WORLD,b1));
  PetscCall(VecCreate(PETSC_COMM_WORLD,b2));
  /* Load matrices from a Q1-P0 discretisation of variable viscosity Stokes. The matrix blocks are packed into one file. */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must provide a matrix file with -f");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  PetscCall(MatLoad(*A11,viewer));
  PetscCall(MatLoad(*A12,viewer));
  PetscCall(MatLoad(*A21,viewer));
  PetscCall(MatLoad(*A22,viewer));
  PetscCall(VecLoad(*b1,viewer));
  PetscCall(VecLoad(*b2,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode LoadTestMatrices(Mat *_A,Vec *_x,Vec *_b,IS *_isu,IS *_isp)
{
  Vec            f,h,x,b,bX[2];
  Mat            A,Auu,Aup,Apu,App,bA[2][2];
  IS             is_u,is_p,bis[2];
  PetscInt       lnu,lnp,nu,np,i,start_u,end_u,start_p,end_p;
  VecScatter     *vscat;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  /* fetch test matrices and vectors */
  PetscCall(LSCLoadTestOperators(&Auu,&Aup,&Apu,&App,&f,&h));

  /* build the mat-nest */
  PetscCall(VecGetSize(f,&nu));
  PetscCall(VecGetSize(h,&np));

  PetscCall(VecGetLocalSize(f,&lnu));
  PetscCall(VecGetLocalSize(h,&lnp));

  PetscCall(VecGetOwnershipRange(f,&start_u,&end_u));
  PetscCall(VecGetOwnershipRange(h,&start_p,&end_p));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] lnu = %" PetscInt_FMT " | lnp = %" PetscInt_FMT " \n", rank, lnu, lnp));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] s_u = %" PetscInt_FMT " | e_u = %" PetscInt_FMT " \n", rank, start_u, end_u));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] s_p = %" PetscInt_FMT " | e_p = %" PetscInt_FMT " \n", rank, start_p, end_p));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] is_u (offset) = %" PetscInt_FMT " \n", rank, start_u+start_p));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] is_p (offset) = %" PetscInt_FMT " \n", rank, start_u+start_p+lnu));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  PetscCall(ISCreateStride(PETSC_COMM_WORLD,lnu,start_u+start_p,1,&is_u));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,lnp,start_u+start_p+lnu,1,&is_p));

  bis[0]   = is_u; bis[1]   = is_p;
  bA[0][0] = Auu;  bA[0][1] = Aup;
  bA[1][0] = Apu;  bA[1][1] = App;
  PetscCall(MatCreateNest(PETSC_COMM_WORLD,2,bis,2,bis,&bA[0][0],&A));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Pull f,h into b */
  PetscCall(MatCreateVecs(A,&b,&x));
  bX[0] = f;  bX[1] = h;
  PetscCall(PetscMalloc1(2,&vscat));
  for (i=0; i<2; i++) {
    PetscCall(VecScatterCreate(b,bis[i],bX[i],NULL,&vscat[i]));
    PetscCall(VecScatterBegin(vscat[i],bX[i],b,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(vscat[i],bX[i],b,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterDestroy(&vscat[i]));
  }

  PetscCall(PetscFree(vscat));
  PetscCall(MatDestroy(&Auu));
  PetscCall(MatDestroy(&Aup));
  PetscCall(MatDestroy(&Apu));
  PetscCall(MatDestroy(&App));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&h));

  *_isu = is_u;
  *_isp = is_p;
  *_A   = A;
  *_x   = x;
  *_b   = b;
  PetscFunctionReturn(0);
}

PetscErrorCode port_lsd_bfbt(void)
{
  Mat            A,P;
  Vec            x,b;
  KSP            ksp_A;
  PC             pc_A;
  IS             isu,isp;
  PetscBool      test_fs = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(LoadTestMatrices(&A,&x,&b,&isu,&isp));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_fs",&test_fs,NULL));
  if (!test_fs) {
    PetscCall(PetscObjectReference((PetscObject)A));
    P = A;
  } else {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&P));
  }
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp_A));
  PetscCall(KSPSetOptionsPrefix(ksp_A,"fc_"));
  PetscCall(KSPSetOperators(ksp_A,A,P));

  PetscCall(KSPSetFromOptions(ksp_A));
  PetscCall(KSPGetPC(ksp_A,&pc_A));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc_A,PCLU,&test_fs));
  if (test_fs) {
    PetscCall(MatDestroy(&P));
    PetscCall(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&P));
    PetscCall(KSPSetOperators(ksp_A,A,P));
    PetscCall(KSPSetFromOptions(ksp_A));
    PetscCall(KSPSolve(ksp_A,b,x));
  } else {
    PetscCall(PCFieldSplitSetBlockSize(pc_A,2));
    PetscCall(PCFieldSplitSetIS(pc_A,"velocity",isu));
    PetscCall(PCFieldSplitSetIS(pc_A,"pressure",isp));
    PetscCall(KSPSolve(ksp_A,b,x));

    /* Pull u,p out of x */
    {
      PetscInt    loc;
      PetscReal   max,norm;
      PetscScalar sum;
      Vec         uvec,pvec;
      VecScatter  uscat,pscat;
      Mat         A11,A22;

      /* grab matrices and create the compatable u,p vectors */
      PetscCall(MatCreateSubMatrix(A,isu,isu,MAT_INITIAL_MATRIX,&A11));
      PetscCall(MatCreateSubMatrix(A,isp,isp,MAT_INITIAL_MATRIX,&A22));

      PetscCall(MatCreateVecs(A11,&uvec,NULL));
      PetscCall(MatCreateVecs(A22,&pvec,NULL));

      /* perform the scatter from x -> (u,p) */
      PetscCall(VecScatterCreate(x,isu,uvec,NULL,&uscat));
      PetscCall(VecScatterBegin(uscat,x,uvec,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(uscat,x,uvec,INSERT_VALUES,SCATTER_FORWARD));

      PetscCall(VecScatterCreate(x,isp,pvec,NULL,&pscat));
      PetscCall(VecScatterBegin(pscat,x,pvec,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pscat,x,pvec,INSERT_VALUES,SCATTER_FORWARD));

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-- vector vector values --\n"));
      PetscCall(VecMin(uvec,&loc,&max));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Min(u)  = %1.6f [loc=%" PetscInt_FMT "]\n",(double)max,loc));
      PetscCall(VecMax(uvec,&loc,&max));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Max(u)  = %1.6f [loc=%" PetscInt_FMT "]\n",(double)max,loc));
      PetscCall(VecNorm(uvec,NORM_2,&norm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Norm(u) = %1.6f \n",(double)norm));
      PetscCall(VecSum(uvec,&sum));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Sum(u)  = %1.6f \n",(double)PetscRealPart(sum)));

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-- pressure vector values --\n"));
      PetscCall(VecMin(pvec,&loc,&max));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Min(p)  = %1.6f [loc=%" PetscInt_FMT "]\n",(double)max,loc));
      PetscCall(VecMax(pvec,&loc,&max));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Max(p)  = %1.6f [loc=%" PetscInt_FMT "]\n",(double)max,loc));
      PetscCall(VecNorm(pvec,NORM_2,&norm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Norm(p) = %1.6f \n",(double)norm));
      PetscCall(VecSum(pvec,&sum));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Sum(p)  = %1.6f \n",(double)PetscRealPart(sum)));

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-- Full vector values --\n"));
      PetscCall(VecMin(x,&loc,&max));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Min(u,p)  = %1.6f [loc=%" PetscInt_FMT "]\n",(double)max,loc));
      PetscCall(VecMax(x,&loc,&max));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Max(u,p)  = %1.6f [loc=%" PetscInt_FMT "]\n",(double)max,loc));
      PetscCall(VecNorm(x,NORM_2,&norm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Norm(u,p) = %1.6f \n",(double)norm));
      PetscCall(VecSum(x,&sum));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Sum(u,p)  = %1.6f \n",(double)PetscRealPart(sum)));

      PetscCall(VecScatterDestroy(&uscat));
      PetscCall(VecScatterDestroy(&pscat));
      PetscCall(VecDestroy(&uvec));
      PetscCall(VecDestroy(&pvec));
      PetscCall(MatDestroy(&A11));
      PetscCall(MatDestroy(&A22));
    }

    /* test second solve by changing the mat associated to the MATNEST blocks */
    {
      PetscCall(replace_submats(A,isu,isp));
      PetscCall(replace_submats(P,isu,isp));
      PetscCall(KSPSolve(ksp_A,b,x));
    }
  }

  PetscCall(KSPDestroy(&ksp_A));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(ISDestroy(&isu));
  PetscCall(ISDestroy(&isp));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(port_lsd_bfbt());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/underworld32.gz -fc_ksp_view -fc_ksp_monitor_short -fc_ksp_type fgmres -fc_ksp_max_it 4000 -fc_ksp_diagonal_scale -fc_pc_type fieldsplit -fc_pc_fieldsplit_type SCHUR -fc_pc_fieldsplit_schur_fact_type UPPER -fc_fieldsplit_velocity_ksp_type cg -fc_fieldsplit_velocity_pc_type cholesky -fc_fieldsplit_velocity_pc_factor_mat_ordering_type nd -fc_fieldsplit_pressure_ksp_max_it 100 -fc_fieldsplit_pressure_ksp_constant_null_space -fc_fieldsplit_pressure_ksp_monitor_short -fc_fieldsplit_pressure_pc_type lsc -fc_fieldsplit_pressure_lsc_ksp_type cg -fc_fieldsplit_pressure_lsc_ksp_max_it 100 -fc_fieldsplit_pressure_lsc_ksp_constant_null_space -fc_fieldsplit_pressure_lsc_ksp_converged_reason -fc_fieldsplit_pressure_lsc_pc_type icc -test_fs {{0 1}separate output} -fc_pc_fieldsplit_off_diag_use_amat {{0 1}separate output} -fc_pc_fieldsplit_diag_use_amat {{0 1}separate output}
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: 2
      nsize: 4
      args: -f ${DATAFILESPATH}/matrices/underworld32.gz -fc_ksp_view -fc_ksp_monitor_short -fc_ksp_type fgmres -fc_ksp_max_it 4000 -fc_ksp_diagonal_scale -fc_pc_type fieldsplit -fc_pc_fieldsplit_type SCHUR -fc_pc_fieldsplit_schur_fact_type UPPER -fc_fieldsplit_velocity_ksp_type cg -fc_fieldsplit_velocity_ksp_rtol 1.0e-6 -fc_fieldsplit_velocity_pc_type bjacobi -fc_fieldsplit_velocity_sub_pc_type cholesky -fc_fieldsplit_velocity_sub_pc_factor_mat_ordering_type nd -fc_fieldsplit_pressure_ksp_type fgmres -fc_fieldsplit_pressure_ksp_constant_null_space -fc_fieldsplit_pressure_ksp_monitor_short -fc_fieldsplit_pressure_pc_type lsc -fc_fieldsplit_pressure_lsc_ksp_type cg -fc_fieldsplit_pressure_lsc_ksp_rtol 1.0e-2 -fc_fieldsplit_pressure_lsc_ksp_constant_null_space -fc_fieldsplit_pressure_lsc_ksp_converged_reason -fc_fieldsplit_pressure_lsc_pc_type bjacobi -fc_fieldsplit_pressure_lsc_sub_pc_type icc -test_fs {{0 1}separate output} -fc_pc_fieldsplit_off_diag_use_amat {{0 1}separate output} -fc_pc_fieldsplit_diag_use_amat {{0 1}separate output}
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: 3
      nsize: 2
      args: -f ${DATAFILESPATH}/matrices/underworld32.gz -fc_ksp_view_pre -fc_pc_type lu
      requires: datafilespath mumps double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/

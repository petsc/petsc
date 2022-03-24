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
  CHKERRQ(MatCreateSubMatrix(A,isu,isu,MAT_INITIAL_MATRIX,&A11));
  CHKERRQ(MatCreateSubMatrix(A,isu,isp,MAT_INITIAL_MATRIX,&A12));
  CHKERRQ(MatCreateSubMatrix(A,isp,isu,MAT_INITIAL_MATRIX,&A21));
  CHKERRQ(MatCreateSubMatrix(A,isp,isp,MAT_INITIAL_MATRIX,&A22));
  CHKERRQ(MatDuplicate(A11,MAT_COPY_VALUES,&nA11));
  CHKERRQ(MatDuplicate(A12,MAT_COPY_VALUES,&nA12));
  CHKERRQ(MatDuplicate(A21,MAT_COPY_VALUES,&nA21));
  CHKERRQ(MatDuplicate(A22,MAT_COPY_VALUES,&nA22));
  CHKERRQ(MatGetOptionsPrefix(A11,&prefix));
  CHKERRQ(MatSetOptionsPrefix(nA11,prefix));
  CHKERRQ(MatGetOptionsPrefix(A22,&prefix));
  CHKERRQ(MatSetOptionsPrefix(nA22,prefix));
  CHKERRQ(MatNestSetSubMat(A,0,0,nA11));
  CHKERRQ(MatNestSetSubMat(A,0,1,nA12));
  CHKERRQ(MatNestSetSubMat(A,1,0,nA21));
  CHKERRQ(MatNestSetSubMat(A,1,1,nA22));
  CHKERRQ(MatDestroy(&A11));
  CHKERRQ(MatDestroy(&A12));
  CHKERRQ(MatDestroy(&A21));
  CHKERRQ(MatDestroy(&A22));
  CHKERRQ(MatDestroy(&nA11));
  CHKERRQ(MatDestroy(&nA12));
  CHKERRQ(MatDestroy(&nA21));
  CHKERRQ(MatDestroy(&nA22));
  PetscFunctionReturn(0);
}

PetscErrorCode LSCLoadTestOperators(Mat *A11,Mat *A12,Mat *A21,Mat *A22,Vec *b1,Vec *b2)
{
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  PetscFunctionBeginUser;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,A11));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,A12));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,A21));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,A22));
  CHKERRQ(MatSetOptionsPrefix(*A11,"a11_"));
  CHKERRQ(MatSetOptionsPrefix(*A22,"a22_"));
  CHKERRQ(MatSetFromOptions(*A11));
  CHKERRQ(MatSetFromOptions(*A22));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,b1));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,b2));
  /* Load matrices from a Q1-P0 discretisation of variable viscosity Stokes. The matrix blocks are packed into one file. */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must provide a matrix file with -f");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(MatLoad(*A11,viewer));
  CHKERRQ(MatLoad(*A12,viewer));
  CHKERRQ(MatLoad(*A21,viewer));
  CHKERRQ(MatLoad(*A22,viewer));
  CHKERRQ(VecLoad(*b1,viewer));
  CHKERRQ(VecLoad(*b2,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
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
  CHKERRQ(LSCLoadTestOperators(&Auu,&Aup,&Apu,&App,&f,&h));

  /* build the mat-nest */
  CHKERRQ(VecGetSize(f,&nu));
  CHKERRQ(VecGetSize(h,&np));

  CHKERRQ(VecGetLocalSize(f,&lnu));
  CHKERRQ(VecGetLocalSize(h,&lnp));

  CHKERRQ(VecGetOwnershipRange(f,&start_u,&end_u));
  CHKERRQ(VecGetOwnershipRange(h,&start_p,&end_p));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] lnu = %D | lnp = %D \n", rank, lnu, lnp));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] s_u = %D | e_u = %D \n", rank, start_u, end_u));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] s_p = %D | e_p = %D \n", rank, start_p, end_p));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] is_u (offset) = %D \n", rank, start_u+start_p));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] is_p (offset) = %D \n", rank, start_u+start_p+lnu));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,lnu,start_u+start_p,1,&is_u));
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,lnp,start_u+start_p+lnu,1,&is_p));

  bis[0]   = is_u; bis[1]   = is_p;
  bA[0][0] = Auu;  bA[0][1] = Aup;
  bA[1][0] = Apu;  bA[1][1] = App;
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,2,bis,2,bis,&bA[0][0],&A));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Pull f,h into b */
  CHKERRQ(MatCreateVecs(A,&b,&x));
  bX[0] = f;  bX[1] = h;
  CHKERRQ(PetscMalloc1(2,&vscat));
  for (i=0; i<2; i++) {
    CHKERRQ(VecScatterCreate(b,bis[i],bX[i],NULL,&vscat[i]));
    CHKERRQ(VecScatterBegin(vscat[i],bX[i],b,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(vscat[i],bX[i],b,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterDestroy(&vscat[i]));
  }

  CHKERRQ(PetscFree(vscat));
  CHKERRQ(MatDestroy(&Auu));
  CHKERRQ(MatDestroy(&Aup));
  CHKERRQ(MatDestroy(&Apu));
  CHKERRQ(MatDestroy(&App));
  CHKERRQ(VecDestroy(&f));
  CHKERRQ(VecDestroy(&h));

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
  CHKERRQ(LoadTestMatrices(&A,&x,&b,&isu,&isp));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_fs",&test_fs,NULL));
  if (!test_fs) {
    CHKERRQ(PetscObjectReference((PetscObject)A));
    P = A;
  } else {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&P));
  }
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp_A));
  CHKERRQ(KSPSetOptionsPrefix(ksp_A,"fc_"));
  CHKERRQ(KSPSetOperators(ksp_A,A,P));

  CHKERRQ(KSPSetFromOptions(ksp_A));
  CHKERRQ(KSPGetPC(ksp_A,&pc_A));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_A,PCLU,&test_fs));
  if (test_fs) {
    CHKERRQ(MatDestroy(&P));
    CHKERRQ(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&P));
    CHKERRQ(KSPSetOperators(ksp_A,A,P));
    CHKERRQ(KSPSetFromOptions(ksp_A));
    CHKERRQ(KSPSolve(ksp_A,b,x));
  } else {
    CHKERRQ(PCFieldSplitSetBlockSize(pc_A,2));
    CHKERRQ(PCFieldSplitSetIS(pc_A,"velocity",isu));
    CHKERRQ(PCFieldSplitSetIS(pc_A,"pressure",isp));
    CHKERRQ(KSPSolve(ksp_A,b,x));

    /* Pull u,p out of x */
    {
      PetscInt    loc;
      PetscReal   max,norm;
      PetscScalar sum;
      Vec         uvec,pvec;
      VecScatter  uscat,pscat;
      Mat         A11,A22;

      /* grab matrices and create the compatable u,p vectors */
      CHKERRQ(MatCreateSubMatrix(A,isu,isu,MAT_INITIAL_MATRIX,&A11));
      CHKERRQ(MatCreateSubMatrix(A,isp,isp,MAT_INITIAL_MATRIX,&A22));

      CHKERRQ(MatCreateVecs(A11,&uvec,NULL));
      CHKERRQ(MatCreateVecs(A22,&pvec,NULL));

      /* perform the scatter from x -> (u,p) */
      CHKERRQ(VecScatterCreate(x,isu,uvec,NULL,&uscat));
      CHKERRQ(VecScatterBegin(uscat,x,uvec,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(uscat,x,uvec,INSERT_VALUES,SCATTER_FORWARD));

      CHKERRQ(VecScatterCreate(x,isp,pvec,NULL,&pscat));
      CHKERRQ(VecScatterBegin(pscat,x,pvec,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pscat,x,pvec,INSERT_VALUES,SCATTER_FORWARD));

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"-- vector vector values --\n"));
      CHKERRQ(VecMin(uvec,&loc,&max));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Min(u)  = %1.6f [loc=%D]\n",(double)max,loc));
      CHKERRQ(VecMax(uvec,&loc,&max));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Max(u)  = %1.6f [loc=%D]\n",(double)max,loc));
      CHKERRQ(VecNorm(uvec,NORM_2,&norm));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Norm(u) = %1.6f \n",(double)norm));
      CHKERRQ(VecSum(uvec,&sum));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Sum(u)  = %1.6f \n",(double)PetscRealPart(sum)));

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"-- pressure vector values --\n"));
      CHKERRQ(VecMin(pvec,&loc,&max));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Min(p)  = %1.6f [loc=%D]\n",(double)max,loc));
      CHKERRQ(VecMax(pvec,&loc,&max));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Max(p)  = %1.6f [loc=%D]\n",(double)max,loc));
      CHKERRQ(VecNorm(pvec,NORM_2,&norm));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Norm(p) = %1.6f \n",(double)norm));
      CHKERRQ(VecSum(pvec,&sum));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Sum(p)  = %1.6f \n",(double)PetscRealPart(sum)));

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"-- Full vector values --\n"));
      CHKERRQ(VecMin(x,&loc,&max));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Min(u,p)  = %1.6f [loc=%D]\n",(double)max,loc));
      CHKERRQ(VecMax(x,&loc,&max));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Max(u,p)  = %1.6f [loc=%D]\n",(double)max,loc));
      CHKERRQ(VecNorm(x,NORM_2,&norm));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Norm(u,p) = %1.6f \n",(double)norm));
      CHKERRQ(VecSum(x,&sum));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Sum(u,p)  = %1.6f \n",(double)PetscRealPart(sum)));

      CHKERRQ(VecScatterDestroy(&uscat));
      CHKERRQ(VecScatterDestroy(&pscat));
      CHKERRQ(VecDestroy(&uvec));
      CHKERRQ(VecDestroy(&pvec));
      CHKERRQ(MatDestroy(&A11));
      CHKERRQ(MatDestroy(&A22));
    }

    /* test second solve by changing the mat associated to the MATNEST blocks */
    {
      CHKERRQ(replace_submats(A,isu,isp));
      CHKERRQ(replace_submats(P,isu,isp));
      CHKERRQ(KSPSolve(ksp_A,b,x));
    }
  }

  CHKERRQ(KSPDestroy(&ksp_A));
  CHKERRQ(MatDestroy(&P));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(ISDestroy(&isu));
  CHKERRQ(ISDestroy(&isp));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{

  CHKERRQ(PetscInitialize(&argc,&argv,0,help));
  CHKERRQ(port_lsd_bfbt());
  CHKERRQ(PetscFinalize());
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

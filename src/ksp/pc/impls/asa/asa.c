
/*  --------------------------------------------------------------------

      Contributed by Arvid Bessen, Columbia University, June 2007

     This file implements a ASA preconditioner in PETSc as part of PC.

     The adaptive smoothed aggregation algorithm is described in the paper
     "Adaptive Smoothed Aggregation (ASA)", M. Brezina, R. Falgout, S. MacLachlan,
     T. Manteuffel, S. McCormick, and J. Ruge, SIAM Journal on Scientific Computing,
     SISC Volume 25 Issue 6, Pages 1896-1920.

     For an example usage of this preconditioner, see, e.g.
     $PETSC_DIR/src/ksp/ksp/examples/tutorials/ex38.c ex39.c
     and other files in that directory.

     This code is still somewhat experimental. A number of improvements would be
     - keep vectors allocated on each level, instead of destroying them
       (see mainly PCApplyVcycleOnLevel_ASA)
     - in PCCreateTransferOp_ASA we get all of the submatrices at once, this could
       be optimized by differentiating between local and global matrices
     - the code does not handle it gracefully if there is just one level
     - if relaxation is sufficient, exit of PCInitializationStage_ASA is not
       completely clean
     - default values could be more reasonable, especially for parallel solves,
       where we need a parallel LU or similar
     - the richardson scaling parameter is somewhat special, should be treated in a
       good default way
     - a number of parameters for smoother (sor_omega, etc.) that we store explicitly
       could be kept in the respective smoothers themselves
     - some parameters have to be set via command line options, there are no direct
       function calls available
     - numerous other stuff

     Example runs in parallel would be with parameters like
     mpiexec ./program -pc_asa_coarse_pc_factor_mat_solver_package mumps -pc_asa_direct_solver 200
     -pc_asa_max_cand_vecs 4 -pc_asa_mu_initial 50 -pc_asa_richardson_scale 1.0
     -pc_asa_rq_improve 0.9 -asa_smoother_pc_type asm -asa_smoother_sub_pc_type sor

    -------------------------------------------------------------------- */

/*
  This defines the data structures for the smoothed aggregation procedure
*/
#include <../src/ksp/pc/impls/asa/asa.h>
#include <petscblaslapack.h>

/* -------------------------------------------------------------------------- */

/* Event logging */

PetscLogEvent PC_InitializationStage_ASA, PC_GeneralSetupStage_ASA;
PetscLogEvent PC_CreateTransferOp_ASA, PC_CreateVcycle_ASA;
PetscBool  asa_events_registered = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PCASASetTolerances"
/*@C
    PCASASetTolerances - Sets the convergence thresholds for ASA algorithm

    Collective on PC

    Input Parameter:
+   pc - the context
.   rtol - the relative convergence tolerance
    (relative decrease in the residual norm)
.   abstol - the absolute convergence tolerance
    (absolute size of the residual norm)
.   dtol - the divergence tolerance
    (amount residual can increase before KSPDefaultConverged()
    concludes that the method is diverging)
-   maxits - maximum number of iterations to use

    Notes:
    Use PETSC_DEFAULT to retain the default value of any of the tolerances.

    Level: advanced
@*/
PetscErrorCode  PCASASetTolerances(PC pc, PetscReal rtol, PetscReal abstol,PetscReal dtol, PetscInt maxits)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCASASetTolerances_C",(PC,PetscReal,PetscReal,PetscReal,PetscInt),(pc,rtol,abstol,dtol,maxits));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCASASetTolerances_ASA"
PetscErrorCode  PCASASetTolerances_ASA(PC pc, PetscReal rtol, PetscReal abstol,PetscReal dtol, PetscInt maxits)
{
  PC_ASA         *asa = (PC_ASA *) pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (rtol != PETSC_DEFAULT)   asa->rtol   = rtol;
  if (abstol != PETSC_DEFAULT)   asa->abstol   = abstol;
  if (dtol != PETSC_DEFAULT)   asa->divtol = dtol;
  if (maxits != PETSC_DEFAULT) asa->max_it = maxits;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PCCreateLevel_ASA"
/*
   PCCreateLevel_ASA - Creates one level for the ASA algorithm

   Input Parameters:
+  level - current level
.  comm - MPI communicator object
.  next - pointer to next level
.  prev - pointer to previous level
.  ksptype - the KSP type for the smoothers on this level
-  pctype - the PC type for the smoothers on this level

   Output Parameters:
.  new_asa_lev - the newly created level

.keywords: ASA, create, levels, multigrid
*/
PetscErrorCode  PCCreateLevel_ASA(PC_ASA_level **new_asa_lev, int level,MPI_Comm comm, PC_ASA_level *prev,
                                                    PC_ASA_level *next,KSPType ksptype, PCType pctype)
{
  PetscErrorCode ierr;
  PC_ASA_level   *asa_lev;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(PC_ASA_level), &asa_lev);CHKERRQ(ierr);

  asa_lev->level = level;
  asa_lev->size  = 0;

  asa_lev->A = 0;
  asa_lev->B = 0;
  asa_lev->x = 0;
  asa_lev->b = 0;
  asa_lev->r = 0;

  asa_lev->dm           = 0;
  asa_lev->aggnum       = 0;
  asa_lev->agg          = 0;
  asa_lev->loc_agg_dofs = 0;
  asa_lev->agg_corr     = 0;
  asa_lev->bridge_corr  = 0;

  asa_lev->P = 0;
  asa_lev->Pt = 0;
  asa_lev->smP = 0;
  asa_lev->smPt = 0;

  asa_lev->comm = comm;

  asa_lev->smoothd = 0;
  asa_lev->smoothu = 0;

  asa_lev->prev = prev;
  asa_lev->next = next;

  *new_asa_lev = asa_lev;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PrintResNorm"
PetscErrorCode PrintResNorm(Mat A, Vec x, Vec b, Vec r)
{
  PetscErrorCode ierr;
  PetscBool      destroyr = PETSC_FALSE;
  PetscReal      resnorm;
  MPI_Comm       Acomm;

  PetscFunctionBegin;
  if (!r) {
    ierr = MatGetVecs(A, PETSC_NULL, &r);CHKERRQ(ierr);
    destroyr = PETSC_TRUE;
  }
  ierr = MatMult(A, x, r);CHKERRQ(ierr);
  ierr = VecAYPX(r, -1.0, b);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &resnorm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) A, &Acomm);CHKERRQ(ierr);
  ierr = PetscPrintf(Acomm, "Residual norm is %f.\n", resnorm);CHKERRQ(ierr);

  if (destroyr) {
    ierr = VecDestroy(&r);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PrintEnergyNorm"
PetscErrorCode PrintEnergyNormOfDiff(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;
  Vec            vecdiff, Avecdiff;
  PetscScalar    dotprod;
  PetscReal      dotabs;
  MPI_Comm       Acomm;

  PetscFunctionBegin;
  ierr = VecDuplicate(x, &vecdiff);CHKERRQ(ierr);
  ierr = VecWAXPY(vecdiff, -1.0, x, y);CHKERRQ(ierr);
  ierr = MatGetVecs(A, PETSC_NULL, &Avecdiff);CHKERRQ(ierr);
  ierr = MatMult(A, vecdiff, Avecdiff);CHKERRQ(ierr);
  ierr = VecDot(vecdiff, Avecdiff, &dotprod);CHKERRQ(ierr);
  dotabs = PetscAbsScalar(dotprod);
  ierr = PetscObjectGetComm((PetscObject) A, &Acomm);CHKERRQ(ierr);
  ierr = PetscPrintf(Acomm, "Energy norm %f.\n", dotabs);CHKERRQ(ierr);
  ierr = VecDestroy(&vecdiff);CHKERRQ(ierr);
  ierr = VecDestroy(&Avecdiff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroyLevel_ASA - Destroys one level of the ASA preconditioner

   Input Parameter:
.  asa_lev - pointer to level that should be destroyed

*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroyLevel_ASA"
PetscErrorCode PCDestroyLevel_ASA(PC_ASA_level *asa_lev)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&(asa_lev->A));CHKERRQ(ierr);
  ierr = MatDestroy(&(asa_lev->B));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa_lev->x));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa_lev->b));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa_lev->r));CHKERRQ(ierr);

  if (asa_lev->dm) {ierr = DMDestroy(&asa_lev->dm);CHKERRQ(ierr);}

  ierr = MatDestroy(&(asa_lev->agg));CHKERRQ(ierr);
  ierr = PetscFree(asa_lev->loc_agg_dofs);CHKERRQ(ierr);
  ierr = MatDestroy(&(asa_lev->agg_corr));CHKERRQ(ierr);
  ierr = MatDestroy(&(asa_lev->bridge_corr));CHKERRQ(ierr);

  ierr = MatDestroy(&(asa_lev->P));CHKERRQ(ierr);
  ierr = MatDestroy(&(asa_lev->Pt));CHKERRQ(ierr);
  ierr = MatDestroy(&(asa_lev->smP));CHKERRQ(ierr);
  ierr = MatDestroy(&(asa_lev->smPt));CHKERRQ(ierr);

  if (asa_lev->smoothd != asa_lev->smoothu) {
    if (asa_lev->smoothd) {ierr = KSPDestroy(&asa_lev->smoothd);CHKERRQ(ierr);}
  }
  if (asa_lev->smoothu) {ierr = KSPDestroy(&asa_lev->smoothu);CHKERRQ(ierr);}

  ierr = PetscFree(asa_lev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCComputeSpectralRadius_ASA - Computes the spectral radius of asa_lev->A
   and stores it it asa_lev->spec_rad

   Input Parameters:
.  asa_lev - the level we are treating

   Compute spectral radius with  sqrt(||A||_1 ||A||_inf) >= ||A||_2 >= rho(A)

*/
#undef __FUNCT__
#define __FUNCT__ "PCComputeSpectralRadius_ASA"
PetscErrorCode PCComputeSpectralRadius_ASA(PC_ASA_level *asa_lev)
{
  PetscErrorCode ierr;
  PetscReal      norm_1, norm_inf;

  PetscFunctionBegin;
  ierr = MatNorm(asa_lev->A, NORM_1, &norm_1);CHKERRQ(ierr);
  ierr = MatNorm(asa_lev->A, NORM_INFINITY, &norm_inf);CHKERRQ(ierr);
  asa_lev->spec_rad = PetscSqrtReal(norm_1*norm_inf);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetRichardsonScale_ASA"
PetscErrorCode PCSetRichardsonScale_ASA(KSP ksp, PetscReal spec_rad, PetscReal richardson_scale) {
  PetscErrorCode ierr;
  PC             pc;
  PetscBool      flg;
  PetscReal      spec_rad_inv;

  PetscFunctionBegin;
  ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);CHKERRQ(ierr);
  if (richardson_scale != PETSC_DECIDE) {
    ierr = KSPRichardsonSetScale(ksp, richardson_scale);CHKERRQ(ierr);
  } else {
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)(pc), PCNONE, &flg);CHKERRQ(ierr);
    if (flg) {
      /* WORK: this is just an educated guess. Any number between 0 and 2/rho(A)
	 should do. asa_lev->spec_rad has to be an upper bound on rho(A). */
      spec_rad_inv = 1.0/spec_rad;
      ierr = KSPRichardsonSetScale(ksp, spec_rad_inv);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP, "Unknown PC type for smoother. Please specify scaling factor with -pc_asa_richardson_scale\n");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetSORomega_ASA"
PetscErrorCode PCSetSORomega_ASA(PC pc, PetscReal sor_omega)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sor_omega != PETSC_DECIDE) {
    ierr = PCSORSetOmega(pc, sor_omega);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCSetupSmoothersOnLevel_ASA - Creates the smoothers of the level.
   We assume that asa_lev->A and asa_lev->spec_rad are correctly computed

   Input Parameters:
+  asa - the data structure for the ASA preconditioner
.  asa_lev - the level we are treating
-  maxits - maximum number of iterations to use
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetupSmoothersOnLevel_ASA"
PetscErrorCode PCSetupSmoothersOnLevel_ASA(PC_ASA *asa, PC_ASA_level *asa_lev, PetscInt maxits)
{
  PetscErrorCode    ierr;
  PetscBool         flg;
  PC                pc;

  PetscFunctionBegin;
  /* destroy old smoothers */
  if (asa_lev->smoothu && asa_lev->smoothu != asa_lev->smoothd) {
    ierr = KSPDestroy(&asa_lev->smoothu);CHKERRQ(ierr);
  }
  asa_lev->smoothu = 0;
  if (asa_lev->smoothd) {
    ierr = KSPDestroy(&asa_lev->smoothd);CHKERRQ(ierr);
  }
  asa_lev->smoothd = 0;
  /* create smoothers */
  ierr = KSPCreate(asa_lev->comm,&asa_lev->smoothd);CHKERRQ(ierr);
  ierr = KSPSetType(asa_lev->smoothd, asa->ksptype_smooth);CHKERRQ(ierr);
  ierr = KSPGetPC(asa_lev->smoothd,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,asa->pctype_smooth);CHKERRQ(ierr);

  /* set up problems for smoothers */
  ierr = KSPSetOperators(asa_lev->smoothd, asa_lev->A, asa_lev->A, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetTolerances(asa_lev->smoothd, asa->smoother_rtol, asa->smoother_abstol, asa->smoother_dtol, maxits);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)(asa_lev->smoothd), KSPRICHARDSON, &flg);CHKERRQ(ierr);
  if (flg) {
    /* special parameters for certain smoothers */
    ierr = KSPSetInitialGuessNonzero(asa_lev->smoothd, PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPGetPC(asa_lev->smoothd, &pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc, PCSOR, &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCSetSORomega_ASA(pc, asa->sor_omega);CHKERRQ(ierr);
    } else {
      /* just set asa->richardson_scale to get some very basic smoother */
      ierr = PCSetRichardsonScale_ASA(asa_lev->smoothd, asa_lev->spec_rad, asa->richardson_scale);CHKERRQ(ierr);
    }
    /* this would be the place to add support for other preconditioners */
  }
  ierr = KSPSetOptionsPrefix(asa_lev->smoothd, "asa_smoother_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(asa_lev->smoothd);CHKERRQ(ierr);
  /* set smoothu equal to smoothd, this could change later */
  asa_lev->smoothu = asa_lev->smoothd;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetupDirectSolversOnLevel_ASA - Creates the direct solvers on the coarsest level.
   We assume that asa_lev->A and asa_lev->spec_rad are correctly computed

   Input Parameters:
+  asa - the data structure for the ASA preconditioner
.  asa_lev - the level we are treating
-  maxits - maximum number of iterations to use
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetupDirectSolversOnLevel_ASA"
PetscErrorCode PCSetupDirectSolversOnLevel_ASA(PC_ASA *asa, PC_ASA_level *asa_lev, PetscInt maxits)
{
  PetscErrorCode    ierr;
  PetscBool         flg;
  PetscMPIInt       comm_size;
  PC                pc;

  PetscFunctionBegin;
  if (asa_lev->smoothu && asa_lev->smoothu != asa_lev->smoothd) {
    ierr = KSPDestroy(&asa_lev->smoothu);CHKERRQ(ierr);
  }
  asa_lev->smoothu = 0;
  if (asa_lev->smoothd) {
    ierr = KSPDestroy(&asa_lev->smoothd);CHKERRQ(ierr);
    asa_lev->smoothd = 0;
  }
  ierr = PetscStrcmp(asa->ksptype_direct, KSPPREONLY, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrcmp(asa->pctype_direct, PCLU, &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MPI_Comm_size(asa_lev->comm, &comm_size);CHKERRQ(ierr);
      if (comm_size > 1) {
	/* the LU PC will call MatSolve, we may have to set the correct type for the matrix
	   to have support for this in parallel */
	ierr = MatConvert(asa_lev->A, asa->coarse_mat_type, MAT_REUSE_MATRIX, &(asa_lev->A));CHKERRQ(ierr);
      }
    }
  }
  /* create new solvers */
  ierr = KSPCreate(asa_lev->comm,&asa_lev->smoothd);CHKERRQ(ierr);
  ierr = KSPSetType(asa_lev->smoothd, asa->ksptype_direct);CHKERRQ(ierr);
  ierr = KSPGetPC(asa_lev->smoothd,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,asa->pctype_direct);CHKERRQ(ierr);
  /* set up problems for direct solvers */
  ierr = KSPSetOperators(asa_lev->smoothd, asa_lev->A, asa_lev->A, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetTolerances(asa_lev->smoothd, asa->direct_rtol, asa->direct_abstol, asa->direct_dtol, maxits);CHKERRQ(ierr);
  /* user can set any option by using -pc_asa_direct_xxx */
  ierr = KSPSetOptionsPrefix(asa_lev->smoothd, "asa_coarse_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(asa_lev->smoothd);CHKERRQ(ierr);
  /* set smoothu equal to 0, not used */
  asa_lev->smoothu = 0;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreateAggregates_ASA - Creates the aggregates

   Input Parameters:
.  asa_lev - the level for which we should create the projection matrix

*/
#undef __FUNCT__
#define __FUNCT__ "PCCreateAggregates_ASA"
PetscErrorCode PCCreateAggregates_ASA(PC_ASA_level *asa_lev)
{
  PetscInt          m,n, m_loc,n_loc;
  PetscInt          m_loc_s, m_loc_e;
  const PetscScalar one = 1.0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Create nodal aggregates A_i^l */
  /* we use the DM grid information for that */
  if (asa_lev->dm) {
    /* coarsen DM and get the restriction matrix */
    ierr = DMCoarsen(asa_lev->dm, MPI_COMM_NULL, &(asa_lev->next->dm));CHKERRQ(ierr);
    ierr = DMCreateAggregates(asa_lev->next->dm, asa_lev->dm, &(asa_lev->agg));CHKERRQ(ierr);
    ierr = MatGetSize(asa_lev->agg, &m, &n);CHKERRQ(ierr);
    ierr = MatGetLocalSize(asa_lev->agg, &m_loc, &n_loc);CHKERRQ(ierr);
    if (n!=asa_lev->size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"DM interpolation matrix has incorrect size!\n");
    asa_lev->next->size = m;
    asa_lev->aggnum     = m;
    /* create the correlators, right now just identity matrices */
    ierr = MatCreateAIJ(asa_lev->comm, n_loc, n_loc, n, n, 1, PETSC_NULL, 1, PETSC_NULL,&(asa_lev->agg_corr));CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(asa_lev->agg_corr, &m_loc_s, &m_loc_e);CHKERRQ(ierr);
    for (m=m_loc_s; m<m_loc_e; m++) {
      ierr = MatSetValues(asa_lev->agg_corr, 1, &m, 1, &m, &one, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(asa_lev->agg_corr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(asa_lev->agg_corr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
/*     ierr = MatShift(asa_lev->agg_corr, 1.0);CHKERRQ(ierr); */
  } else {
    /* somehow define the aggregates without knowing the geometry */
    /* future WORK */
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Currently pure algebraic coarsening is not supported!");
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreateTransferOp_ASA - Creates the transfer operator P_{l+1}^l for current level

   Input Parameters:
+  asa_lev - the level for which should create the transfer operator
-  construct_bridge - true, if we should construct a bridge operator, false for normal prolongator

   If we add a second, third, ... candidate vector (i.e. more than one column in B), we
   have to relate the additional dimensions to the original aggregates. This is done through
   the "aggregate correlators" agg_corr and bridge_corr.
   The aggregate that is used in the construction is then given by
   asa_lev->agg * asa_lev->agg_corr
   for the regular prolongator construction and
   asa_lev->agg * asa_lev->bridge_corr
   for the bridging prolongator constructions.
*/
#undef __FUNCT__
#define __FUNCT__ "PCCreateTransferOp_ASA"
PetscErrorCode PCCreateTransferOp_ASA(PC_ASA_level *asa_lev, PetscBool  construct_bridge)
{
  PetscErrorCode ierr;

  const PetscReal Ca = 1e-3;
  PetscReal      cutoff;
  PetscInt       nodes_on_lev;

  Mat            logical_agg;
  PetscInt       mat_agg_loc_start, mat_agg_loc_end, mat_agg_loc_size;
  PetscInt       a;
  const PetscInt *agg = 0;
  PetscInt       **agg_arr = 0;

  IS             *idxm_is_B_arr = 0;
  PetscInt       *idxn_B = 0;
  IS             idxn_is_B, *idxn_is_B_arr = 0;

  Mat            *b_submat_arr = 0;

  PetscScalar    *b_submat = 0, *b_submat_tp = 0;
  PetscInt       *idxm = 0, *idxn = 0;
  PetscInt       cand_vecs_num;
  PetscInt       *cand_vec_length = 0;
  PetscInt       max_cand_vec_length = 0;
  PetscScalar    **b_orth_arr = 0;

  PetscInt       i,j;

  PetscScalar    *tau = 0, *work = 0;
  PetscBLASInt   info,b1,b2;

  PetscInt       max_cand_vecs_to_add;
  PetscInt       *new_loc_agg_dofs = 0;

  PetscInt       total_loc_cols = 0;
  PetscReal      norm;

  PetscInt       a_loc_m, a_loc_n;
  PetscInt       mat_loc_col_start, mat_loc_col_end, mat_loc_col_size;
  PetscInt       loc_agg_dofs_sum;
  PetscInt       row, col;
  PetscScalar    val;
  PetscMPIInt    comm_size, comm_rank;
  PetscInt       *loc_cols = 0;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_CreateTransferOp_ASA,0,0,0,0);CHKERRQ(ierr);

  ierr = MatGetSize(asa_lev->B, &nodes_on_lev, PETSC_NULL);CHKERRQ(ierr);

  /* If we add another candidate vector, we want to be able to judge, how much the new candidate
     improves our current projection operators and whether it is worth adding it.
     This is the precomputation necessary for implementing Notes (4.1) to (4.7).
     We require that all candidate vectors x stored in B are normalized such that
     <A x, x> = 1 and we thus do not have to compute this.
     For each aggregate A we can now test condition (4.5) and (4.6) by computing
     || quantity to check ||_{A}^2 <= cutoff * card(A)/N_l */
  cutoff = Ca/(asa_lev->spec_rad);

  /* compute logical aggregates by using the correlators */
  if (construct_bridge) {
    /* construct bridging operator */
    ierr = MatMatMult(asa_lev->agg, asa_lev->bridge_corr, MAT_INITIAL_MATRIX, 1.0, &logical_agg);CHKERRQ(ierr);
  } else {
    /* construct "regular" prolongator */
    ierr = MatMatMult(asa_lev->agg, asa_lev->agg_corr, MAT_INITIAL_MATRIX, 1.0, &logical_agg);CHKERRQ(ierr);
  }

  /* destroy correlator matrices for next level, these will be rebuilt in this routine */
  if (asa_lev->next) {
    ierr = MatDestroy(&(asa_lev->next->agg_corr));CHKERRQ(ierr);
    ierr = MatDestroy(&(asa_lev->next->bridge_corr));CHKERRQ(ierr);
  }

  /* find out the correct local row indices */
  ierr = MatGetOwnershipRange(logical_agg, &mat_agg_loc_start, &mat_agg_loc_end);CHKERRQ(ierr);
  mat_agg_loc_size = mat_agg_loc_end-mat_agg_loc_start;

  cand_vecs_num = asa_lev->cand_vecs;

  /* construct column indices idxn_B for reading from B */
  ierr = PetscMalloc(sizeof(PetscInt)*(cand_vecs_num), &idxn_B);CHKERRQ(ierr);
  for (i=0; i<cand_vecs_num; i++) {
    idxn_B[i] = i;
  }
  ierr = ISCreateGeneral(asa_lev->comm, asa_lev->cand_vecs, idxn_B,PETSC_COPY_VALUES, &idxn_is_B);CHKERRQ(ierr);
  ierr = PetscFree(idxn_B);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(IS)*mat_agg_loc_size, &idxn_is_B_arr);CHKERRQ(ierr);
  for (a=0; a<mat_agg_loc_size; a++) {
    idxn_is_B_arr[a] = idxn_is_B;
  }
  /* allocate storage for row indices idxm_B */
  ierr = PetscMalloc(sizeof(IS)*mat_agg_loc_size, &idxm_is_B_arr);CHKERRQ(ierr);

  /* Storage for the orthogonalized  submatrices of B and their sizes */
  ierr = PetscMalloc(sizeof(PetscInt)*mat_agg_loc_size, &cand_vec_length);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar*)*mat_agg_loc_size, &b_orth_arr);CHKERRQ(ierr);
  /* Storage for the information about each aggregate */
  ierr = PetscMalloc(sizeof(PetscInt*)*mat_agg_loc_size, &agg_arr);CHKERRQ(ierr);
  /* Storage for the number of candidate vectors that are orthonormal and used in each submatrix */
  ierr = PetscMalloc(sizeof(PetscInt)*mat_agg_loc_size, &new_loc_agg_dofs);CHKERRQ(ierr);

  /* loop over local aggregates */
  for (a=0; a<mat_agg_loc_size; a++) {
       /* get info about current aggregate, this gives the rows we have to get from B */
       ierr = MatGetRow(logical_agg, a+mat_agg_loc_start, &cand_vec_length[a], &agg, 0);CHKERRQ(ierr);
       /* copy aggregate information */
       ierr = PetscMalloc(sizeof(PetscInt)*cand_vec_length[a], &(agg_arr[a]));CHKERRQ(ierr);
       ierr = PetscMemcpy(agg_arr[a], agg, sizeof(PetscInt)*cand_vec_length[a]);CHKERRQ(ierr);
       /* restore row */
       ierr = MatRestoreRow(logical_agg, a+mat_agg_loc_start, &cand_vec_length[a], &agg, 0);CHKERRQ(ierr);

       /* create index sets */
       ierr = ISCreateGeneral(PETSC_COMM_SELF, cand_vec_length[a], agg_arr[a],PETSC_COPY_VALUES, &(idxm_is_B_arr[a]));CHKERRQ(ierr);
       /* maximum candidate vector length */
       if (cand_vec_length[a] > max_cand_vec_length) { max_cand_vec_length = cand_vec_length[a]; }
  }
  /* destroy logical_agg, no longer needed */
  ierr = MatDestroy(&logical_agg);CHKERRQ(ierr);

  /* get the entries for aggregate from B */
  ierr = MatGetSubMatrices(asa_lev->B, mat_agg_loc_size, idxm_is_B_arr, idxn_is_B_arr, MAT_INITIAL_MATRIX, &b_submat_arr);CHKERRQ(ierr);

  /* clean up all the index sets */
  for (a=0; a<mat_agg_loc_size; a++) { ISDestroy(&idxm_is_B_arr[a]);CHKERRQ(ierr); }
  ierr = PetscFree(idxm_is_B_arr);CHKERRQ(ierr);
  ierr = ISDestroy(&idxn_is_B);CHKERRQ(ierr);
  ierr = PetscFree(idxn_is_B_arr);CHKERRQ(ierr);

  /* storage for the values from each submatrix */
  ierr = PetscMalloc(sizeof(PetscScalar)*max_cand_vec_length*cand_vecs_num, &b_submat);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*max_cand_vec_length*cand_vecs_num, &b_submat_tp);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*max_cand_vec_length, &idxm);CHKERRQ(ierr);
  for (i=0; i<max_cand_vec_length; i++) { idxm[i] = i; }
  ierr = PetscMalloc(sizeof(PetscInt)*cand_vecs_num, &idxn);CHKERRQ(ierr);
  for (i=0; i<cand_vecs_num; i++) { idxn[i] = i; }
  /* work storage for QR algorithm */
  ierr = PetscMalloc(sizeof(PetscScalar)*max_cand_vec_length, &tau);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*cand_vecs_num, &work);CHKERRQ(ierr);

  /* orthogonalize all submatrices and store them in b_orth_arr */
  for (a=0; a<mat_agg_loc_size; a++) {
       /* Get the entries for aggregate from B. This is row ordered (although internally
	  it is column ordered and we will waste some energy transposing it).
	  WORK: use something like MatGetArray(b_submat_arr[a], &b_submat) but be really
	  careful about all the different matrix types */
       ierr = MatGetValues(b_submat_arr[a], cand_vec_length[a], idxm, cand_vecs_num, idxn, b_submat);CHKERRQ(ierr);

       if (construct_bridge) {
	 /* if we are constructing a bridging restriction/interpolation operator, we have
	    to use the same number of dofs as in our previous construction */
	 max_cand_vecs_to_add = asa_lev->loc_agg_dofs[a];
       } else {
	 /* for a normal restriction/interpolation operator, we should make sure that we
	    do not create linear dependence by accident */
	 max_cand_vecs_to_add = PetscMin(cand_vec_length[a], cand_vecs_num);
       }

       /* We use LAPACK to compute the QR decomposition of b_submat. For LAPACK we have to
	  transpose the matrix. We might throw out some column vectors during this process.
	  We are keeping count of the number of column vectors that we use (and therefore the
	  number of dofs on the lower level) in new_loc_agg_dofs[a]. */
       new_loc_agg_dofs[a] = 0;
       for (j=0; j<max_cand_vecs_to_add; j++) {
	 /* check for condition (4.5) */
	 norm = 0.0;
	 for (i=0; i<cand_vec_length[a]; i++) {
	   norm += PetscRealPart(b_submat[i*cand_vecs_num+j])*PetscRealPart(b_submat[i*cand_vecs_num+j])
	     + PetscImaginaryPart(b_submat[i*cand_vecs_num+j])*PetscImaginaryPart(b_submat[i*cand_vecs_num+j]);
	 }
	 /* only add candidate vector if bigger than cutoff or first candidate */
	 if ((!j) || (norm > cutoff*((PetscReal) cand_vec_length[a])/((PetscReal) nodes_on_lev))) {
	   /* passed criterion (4.5), we have not implemented criterion (4.6) yet */
	   for (i=0; i<cand_vec_length[a]; i++) {
	     b_submat_tp[new_loc_agg_dofs[a]*cand_vec_length[a]+i] = b_submat[i*cand_vecs_num+j];
	   }
	   new_loc_agg_dofs[a]++;
	 }
	 /* #ifdef PCASA_VERBOSE */
	 else {
	   ierr = PetscPrintf(asa_lev->comm, "Cutoff criteria invoked\n");CHKERRQ(ierr);
	 }
	 /* #endif */
       }

       CHKMEMQ;
       /* orthogonalize b_submat_tp using the QR algorithm from LAPACK */
       b1 = PetscBLASIntCast(*(cand_vec_length+a));
       b2 = PetscBLASIntCast(*(new_loc_agg_dofs+a));

       ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_MISSING_LAPACK_GEQRF)
       LAPACKgeqrf_(&b1, &b2, b_submat_tp, &b1, tau, work, &b2, &info);
       if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, "LAPACKgeqrf_ LAPACK routine failed");
#else
       SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"geqrf() - Lapack routine is unavailable\n");
#endif
#if !defined(PETSC_MISSING_LAPACK_ORGQR)
       LAPACKungqr_(&b1, &b2, &b2, b_submat_tp, &b1, tau, work, &b2, &info);
#else
       SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"ORGQR - Lapack routine is unavailable\nIf linking with ESSL you MUST also link with full LAPACK, for example\nuse ./configure with --with-blas-lib=libessl.a --with-lapack-lib=/usr/local/lib/liblapack.a'");
#endif
       if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, "LAPACKungqr_ LAPACK routine failed");
       ierr = PetscFPTrapPop();CHKERRQ(ierr);

       /* Transpose b_submat_tp and store it in b_orth_arr[a]. If we are constructing a
	  bridging restriction/interpolation operator, we could end up with less dofs than
	  we previously had. We fill those up with zeros. */
       if (!construct_bridge) {
	 ierr = PetscMalloc(sizeof(PetscScalar)*cand_vec_length[a]*new_loc_agg_dofs[a], b_orth_arr+a);CHKERRQ(ierr);
	 for (j=0; j<new_loc_agg_dofs[a]; j++) {
	   for (i=0; i<cand_vec_length[a]; i++) {
	     b_orth_arr[a][i*new_loc_agg_dofs[a]+j] = b_submat_tp[j*cand_vec_length[a]+i];
	   }
	 }
       } else {
	 /* bridge, might have to fill up */
	 ierr = PetscMalloc(sizeof(PetscScalar)*cand_vec_length[a]*max_cand_vecs_to_add, b_orth_arr+a);CHKERRQ(ierr);
	 for (j=0; j<new_loc_agg_dofs[a]; j++) {
	   for (i=0; i<cand_vec_length[a]; i++) {
	     b_orth_arr[a][i*max_cand_vecs_to_add+j] = b_submat_tp[j*cand_vec_length[a]+i];
	   }
	 }
	 for (j=new_loc_agg_dofs[a]; j<max_cand_vecs_to_add; j++) {
	   for (i=0; i<cand_vec_length[a]; i++) {
	     b_orth_arr[a][i*max_cand_vecs_to_add+j] = 0.0;
	   }
	 }
	 new_loc_agg_dofs[a] = max_cand_vecs_to_add;
       }
       /* the number of columns in asa_lev->P that are local to this process */
       total_loc_cols += new_loc_agg_dofs[a];
  } /* end of loop over local aggregates */

  /* destroy the submatrices, also frees all allocated space */
  ierr = MatDestroyMatrices(mat_agg_loc_size, &b_submat_arr);CHKERRQ(ierr);
  /* destroy all other workspace */
  ierr = PetscFree(b_submat);CHKERRQ(ierr);
  ierr = PetscFree(b_submat_tp);CHKERRQ(ierr);
  ierr = PetscFree(idxm);CHKERRQ(ierr);
  ierr = PetscFree(idxn);CHKERRQ(ierr);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);

  /* destroy old matrix P, Pt */
  ierr = MatDestroy(&(asa_lev->P));CHKERRQ(ierr);
  ierr = MatDestroy(&(asa_lev->Pt));CHKERRQ(ierr);

  ierr = MatGetLocalSize(asa_lev->A, &a_loc_m, &a_loc_n);CHKERRQ(ierr);

  /* determine local range */
  ierr = MPI_Comm_size(asa_lev->comm, &comm_size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(asa_lev->comm, &comm_rank);CHKERRQ(ierr);
  ierr = PetscMalloc(comm_size*sizeof(PetscInt), &loc_cols);CHKERRQ(ierr);
  ierr = MPI_Allgather(&total_loc_cols, 1, MPIU_INT, loc_cols, 1, MPIU_INT, asa_lev->comm);CHKERRQ(ierr);
  mat_loc_col_start = 0;
  for (i=0;i<comm_rank;i++) {
    mat_loc_col_start += loc_cols[i];
  }
  mat_loc_col_end = mat_loc_col_start + loc_cols[i];
  mat_loc_col_size = mat_loc_col_end-mat_loc_col_start;
  if (mat_loc_col_size != total_loc_cols) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR, "Local size does not match matrix size");
  ierr = PetscFree(loc_cols);CHKERRQ(ierr);

  /* we now have enough information to create asa_lev->P */
  ierr = MatCreateAIJ(asa_lev->comm, a_loc_n,  total_loc_cols, asa_lev->size, PETSC_DETERMINE,
			 cand_vecs_num, PETSC_NULL, cand_vecs_num, PETSC_NULL, &(asa_lev->P));CHKERRQ(ierr);
  /* create asa_lev->Pt */
  ierr = MatCreateAIJ(asa_lev->comm, total_loc_cols, a_loc_n, PETSC_DETERMINE, asa_lev->size,
			 max_cand_vec_length, PETSC_NULL, max_cand_vec_length, PETSC_NULL, &(asa_lev->Pt));CHKERRQ(ierr);
  if (asa_lev->next) {
    /* create correlator for aggregates of next level */
    ierr = MatCreateAIJ(asa_lev->comm, mat_agg_loc_size, total_loc_cols, PETSC_DETERMINE, PETSC_DETERMINE,
			   cand_vecs_num, PETSC_NULL, cand_vecs_num, PETSC_NULL, &(asa_lev->next->agg_corr));CHKERRQ(ierr);
    /* create asa_lev->next->bridge_corr matrix */
    ierr = MatCreateAIJ(asa_lev->comm, mat_agg_loc_size, total_loc_cols, PETSC_DETERMINE, PETSC_DETERMINE,
			   cand_vecs_num, PETSC_NULL, cand_vecs_num, PETSC_NULL, &(asa_lev->next->bridge_corr));CHKERRQ(ierr);
  }

  /* this is my own hack, but it should give the columns that we should write to */
  ierr = MatGetOwnershipRangeColumn(asa_lev->P, &mat_loc_col_start, &mat_loc_col_end);CHKERRQ(ierr);
  mat_loc_col_size = mat_loc_col_end-mat_loc_col_start;
  if (mat_loc_col_size != total_loc_cols) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "The number of local columns in asa_lev->P assigned to this processor does not match the local vector size");

  loc_agg_dofs_sum = 0;
  /* construct P, Pt, agg_corr, bridge_corr */
  for (a=0; a<mat_agg_loc_size; a++) {
    /* store b_orth_arr[a] in P */
    for (i=0; i<cand_vec_length[a]; i++) {
      row = agg_arr[a][i];
      for (j=0; j<new_loc_agg_dofs[a]; j++) {
	col = mat_loc_col_start + loc_agg_dofs_sum + j;
	val = b_orth_arr[a][i*new_loc_agg_dofs[a] + j];
	ierr = MatSetValues(asa_lev->P, 1, &row, 1, &col, &val, INSERT_VALUES);CHKERRQ(ierr);
	val = PetscConj(val);
	ierr = MatSetValues(asa_lev->Pt, 1, &col, 1, &row, &val, INSERT_VALUES);CHKERRQ(ierr);
      }
    }

    /* compute aggregate correlation matrices */
    if (asa_lev->next) {
      row = a+mat_agg_loc_start;
      for (i=0; i<new_loc_agg_dofs[a]; i++) {
	col = mat_loc_col_start + loc_agg_dofs_sum + i;
	val = 1.0;
	ierr = MatSetValues(asa_lev->next->agg_corr, 1, &row, 1, &col, &val, INSERT_VALUES);CHKERRQ(ierr);
	/* for the bridge operator we leave out the newest candidates, i.e.
	   we set bridge_corr to 1.0 for all columns up to asa_lev->loc_agg_dofs[a] and to
	   0.0 between asa_lev->loc_agg_dofs[a] and new_loc_agg_dofs[a] */
	if (!(asa_lev->loc_agg_dofs && (i >= asa_lev->loc_agg_dofs[a]))) {
	  ierr = MatSetValues(asa_lev->next->bridge_corr, 1, &row, 1, &col, &val, INSERT_VALUES);CHKERRQ(ierr);
	}
      }
    }

    /* move to next entry point col */
    loc_agg_dofs_sum += new_loc_agg_dofs[a];
  } /* end of loop over local aggregates */

  ierr = MatAssemblyBegin(asa_lev->P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(asa_lev->P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(asa_lev->Pt,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(asa_lev->Pt,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (asa_lev->next) {
    ierr = MatAssemblyBegin(asa_lev->next->agg_corr,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(asa_lev->next->agg_corr,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(asa_lev->next->bridge_corr,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(asa_lev->next->bridge_corr,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* if we are not constructing a bridging operator, switch asa_lev->loc_agg_dofs
     and new_loc_agg_dofs */
  if (construct_bridge) {
    ierr = PetscFree(new_loc_agg_dofs);CHKERRQ(ierr);
  } else {
    if (asa_lev->loc_agg_dofs) {
      ierr = PetscFree(asa_lev->loc_agg_dofs);CHKERRQ(ierr);
    }
    asa_lev->loc_agg_dofs = new_loc_agg_dofs;
  }

  /* clean up */
  for (a=0; a<mat_agg_loc_size; a++) {
    ierr = PetscFree(b_orth_arr[a]);CHKERRQ(ierr);
    ierr = PetscFree(agg_arr[a]);CHKERRQ(ierr);
  }
  ierr = PetscFree(cand_vec_length);CHKERRQ(ierr);
  ierr = PetscFree(b_orth_arr);CHKERRQ(ierr);
  ierr = PetscFree(agg_arr);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(PC_CreateTransferOp_ASA, 0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSmoothProlongator_ASA - Computes the smoothed prolongators I and It on the level

   Input Parameters:
.  asa_lev - the level for which the smoothed prolongator is constructed
*/
#undef __FUNCT__
#define __FUNCT__ "PCSmoothProlongator_ASA"
PetscErrorCode PCSmoothProlongator_ASA(PC_ASA_level *asa_lev)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&(asa_lev->smP));CHKERRQ(ierr);
  ierr = MatDestroy(&(asa_lev->smPt));CHKERRQ(ierr);
  /* compute prolongator I_{l+1}^l = S_l P_{l+1}^l */
  /* step 1: compute I_{l+1}^l = A_l P_{l+1}^l */
  ierr = MatMatMult(asa_lev->A, asa_lev->P, MAT_INITIAL_MATRIX, 1, &(asa_lev->smP));CHKERRQ(ierr);
  ierr = MatMatMult(asa_lev->Pt, asa_lev->A, MAT_INITIAL_MATRIX, 1, &(asa_lev->smPt));CHKERRQ(ierr);
  /* step 2: shift and scale to get I_{l+1}^l = P_{l+1}^l - 4/(3/rho) A_l P_{l+1}^l */
  ierr = MatAYPX(asa_lev->smP, -4./(3.*(asa_lev->spec_rad)), asa_lev->P, SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAYPX(asa_lev->smPt, -4./(3.*(asa_lev->spec_rad)), asa_lev->Pt, SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCCreateVcycle_ASA - Creates the V-cycle, when aggregates are already defined

   Input Parameters:
.  asa - the preconditioner context
*/
#undef __FUNCT__
#define __FUNCT__ "PCCreateVcycle_ASA"
PetscErrorCode PCCreateVcycle_ASA(PC_ASA *asa)
{
  PetscErrorCode ierr;
  PC_ASA_level   *asa_lev, *asa_next_lev;
  Mat            AI;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_CreateVcycle_ASA, 0,0,0,0);CHKERRQ(ierr);

  if (!asa) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL, "asa pointer is NULL");
  if (!(asa->levellist)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL, "no levels found");
  asa_lev = asa->levellist;
  ierr = PCComputeSpectralRadius_ASA(asa_lev);CHKERRQ(ierr);
  ierr = PCSetupSmoothersOnLevel_ASA(asa, asa_lev, asa->nu);CHKERRQ(ierr);

  while(asa_lev->next) {
    asa_next_lev = asa_lev->next;
    /* (a) aggregates are already constructed */

    /* (b) construct B_{l+1} and P_{l+1}^l using (2.11) */
    /* construct P_{l+1}^l */
    ierr = PCCreateTransferOp_ASA(asa_lev, PETSC_FALSE);CHKERRQ(ierr);

    /* construct B_{l+1} */
    ierr = MatDestroy(&(asa_next_lev->B));CHKERRQ(ierr);
    ierr = MatMatMult(asa_lev->Pt, asa_lev->B, MAT_INITIAL_MATRIX, 1, &(asa_next_lev->B));CHKERRQ(ierr);
    asa_next_lev->cand_vecs = asa_lev->cand_vecs;

    /* (c) construct smoothed prolongator */
    ierr = PCSmoothProlongator_ASA(asa_lev);CHKERRQ(ierr);

    /* (d) construct coarse matrix */
    /* Define coarse matrix A_{l+1} = (I_{l+1}^l)^T A_l I_{l+1}^l */
    ierr = MatDestroy(&(asa_next_lev->A));CHKERRQ(ierr);
       ierr = MatMatMult(asa_lev->A, asa_lev->smP, MAT_INITIAL_MATRIX, 1.0, &AI);CHKERRQ(ierr);
     ierr = MatMatMult(asa_lev->smPt, AI, MAT_INITIAL_MATRIX, 1.0, &(asa_next_lev->A));CHKERRQ(ierr);
     ierr = MatDestroy(&AI);CHKERRQ(ierr);
    /*     ierr = MatPtAP(asa_lev->A, asa_lev->smP, MAT_INITIAL_MATRIX, 1, &(asa_next_lev->A));CHKERRQ(ierr); */
    ierr = MatGetSize(asa_next_lev->A, PETSC_NULL, &(asa_next_lev->size));CHKERRQ(ierr);
    ierr = PCComputeSpectralRadius_ASA(asa_next_lev);CHKERRQ(ierr);
    ierr = PCSetupSmoothersOnLevel_ASA(asa, asa_next_lev, asa->nu);CHKERRQ(ierr);
    /* create corresponding vectors x_{l+1}, b_{l+1}, r_{l+1} */
    ierr = VecDestroy(&(asa_next_lev->x));CHKERRQ(ierr);
    ierr = VecDestroy(&(asa_next_lev->b));CHKERRQ(ierr);
    ierr = VecDestroy(&(asa_next_lev->r));CHKERRQ(ierr);
    ierr = MatGetVecs(asa_next_lev->A, &(asa_next_lev->x), &(asa_next_lev->b));CHKERRQ(ierr);
    ierr = MatGetVecs(asa_next_lev->A, PETSC_NULL, &(asa_next_lev->r));CHKERRQ(ierr);

    /* go to next level */
    asa_lev = asa_lev->next;
  } /* end of while loop over the levels */
  /* asa_lev now points to the coarsest level, set up direct solver there */
  ierr = PCComputeSpectralRadius_ASA(asa_lev);CHKERRQ(ierr);
  ierr = PCSetupDirectSolversOnLevel_ASA(asa, asa_lev, asa->nu);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(PC_CreateVcycle_ASA, 0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCAddCandidateToB_ASA - Inserts a candidate vector in B

   Input Parameters:
+  B - the matrix to insert into
.  col_idx - the column we should insert to
.  x - the vector to insert
-  A - system matrix

   Function will insert normalized x into B, such that <A x, x> = 1
   (x itself is not changed). If B is projected down then this property
   is kept. If <A_l x_l, x_l> = 1 and the next level is defined by
   x_{l+1} = Pt x_l  and  A_{l+1} = Pt A_l P then
   <A_{l+1} x_{l+1}, x_l> = <Pt A_l P Pt x_l, Pt x_l>
   = <A_l P Pt x_l, P Pt x_l> = <A_l x_l, x_l> = 1
   because of the definition of P in (2.11).
*/
#undef __FUNCT__
#define __FUNCT__ "PCAddCandidateToB_ASA"
PetscErrorCode PCAddCandidateToB_ASA(Mat B, PetscInt col_idx, Vec x, Mat A)
{
  PetscErrorCode ierr;
  Vec            Ax;
  PetscScalar    dotprod;
  PetscReal      norm;
  PetscInt       i, loc_start, loc_end;
  PetscScalar    val, *vecarray;

  PetscFunctionBegin;
  ierr = MatGetVecs(A, PETSC_NULL, &Ax);CHKERRQ(ierr);
  ierr = MatMult(A, x, Ax);CHKERRQ(ierr);
  ierr = VecDot(Ax, x, &dotprod);CHKERRQ(ierr);
  norm = PetscSqrtReal(PetscAbsScalar(dotprod));
  ierr = VecGetOwnershipRange(x, &loc_start, &loc_end);CHKERRQ(ierr);
  ierr = VecGetArray(x, &vecarray);CHKERRQ(ierr);
  for (i=loc_start; i<loc_end; i++) {
    val = vecarray[i-loc_start]/norm;
    ierr = MatSetValues(B, 1, &i, 1, &col_idx, &val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &vecarray);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
-  x - a starting guess for a hard to approximate vector, if PETSC_NULL, will be generated
*/
#undef __FUNCT__
#define __FUNCT__ "PCInitializationStage_ASA"
PetscErrorCode PCInitializationStage_ASA(PC pc, Vec x)
{
  PetscErrorCode ierr;
  PetscInt       l;
  PC_ASA         *asa = (PC_ASA*)pc->data;
  PC_ASA_level   *asa_lev, *asa_next_lev;
  PetscRandom    rctx;     /* random number generator context */

  Vec            ax;
  PetscScalar    tmp;
  PetscReal      prevnorm, norm;

  PetscBool      skip_steps_f_i = PETSC_FALSE;
  PetscBool      sufficiently_coarsened = PETSC_FALSE;

  PetscInt       vec_size, vec_loc_size;
  PetscInt       loc_vec_low, loc_vec_high;
  PetscInt       i,j;

/*   Vec            xhat = 0; */

  Mat            AI;

  Vec            cand_vec, cand_vec_new;
  PetscBool      isrichardson;
  PC             coarse_pc;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_InitializationStage_ASA,0,0,0,0);CHKERRQ(ierr);
  l=1;
  /* create first level */
  ierr = PCCreateLevel_ASA(&(asa->levellist), l, asa->comm, 0, 0, asa->ksptype_smooth, asa->pctype_smooth);CHKERRQ(ierr);
  asa_lev = asa->levellist;

  /* Set matrix */
  asa_lev->A = asa->A;
  ierr = MatGetSize(asa_lev->A, &i, &j);CHKERRQ(ierr);
  asa_lev->size = i;
  ierr = PCComputeSpectralRadius_ASA(asa_lev);CHKERRQ(ierr);
  ierr = PCSetupSmoothersOnLevel_ASA(asa, asa_lev, asa->mu_initial);CHKERRQ(ierr);

  /* Set DM */
  asa_lev->dm = pc->dm;
  ierr = PetscObjectReference((PetscObject)pc->dm);CHKERRQ(ierr);

  ierr = PetscPrintf(asa_lev->comm, "Initialization stage\n");CHKERRQ(ierr);

  if (x) {
    /* use starting guess */
    ierr = VecDestroy(&(asa_lev->x));CHKERRQ(ierr);
    ierr = VecDuplicate(x, &(asa_lev->x));CHKERRQ(ierr);
    ierr = VecCopy(x, asa_lev->x);CHKERRQ(ierr);
  } else {
    /* select random starting vector */
    ierr = VecDestroy(&(asa_lev->x));CHKERRQ(ierr);
    ierr = MatGetVecs(asa_lev->A, &(asa_lev->x), 0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(asa_lev->comm,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(asa_lev->x, rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  }

  /* create right hand side */
  ierr = VecDestroy(&(asa_lev->b));CHKERRQ(ierr);
  ierr = MatGetVecs(asa_lev->A, &(asa_lev->b), 0);
  ierr = VecSet(asa_lev->b, 0.0);

  /* relax and check whether that's enough already */
  /* compute old norm */
  ierr = MatGetVecs(asa_lev->A, 0, &ax);CHKERRQ(ierr);
  ierr = MatMult(asa_lev->A, asa_lev->x, ax);CHKERRQ(ierr);
  ierr = VecDot(asa_lev->x, ax, &tmp);CHKERRQ(ierr);
  prevnorm = PetscAbsScalar(tmp);
  ierr = PetscPrintf(asa_lev->comm, "Residual norm of starting guess: %f\n", prevnorm);CHKERRQ(ierr);

  /* apply mu_initial relaxations */
  ierr = KSPSolve(asa_lev->smoothd, asa_lev->b, asa_lev->x);CHKERRQ(ierr);
  /* compute new norm */
  ierr = MatMult(asa_lev->A, asa_lev->x, ax);CHKERRQ(ierr);
  ierr = VecDot(asa_lev->x, ax, &tmp);CHKERRQ(ierr);
  norm = PetscAbsScalar(tmp);
  ierr = VecDestroy(&(ax));CHKERRQ(ierr);
  ierr = PetscPrintf(asa_lev->comm, "Residual norm of relaxation after %g %D relaxations: %g %g\n", asa->epsilon,asa->mu_initial, norm,prevnorm);CHKERRQ(ierr);

  /* Check if it already converges by itself */
  if (norm/prevnorm <= pow(asa->epsilon, (PetscReal) asa->mu_initial)) {
    /* converges by relaxation alone */
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Relaxation should be sufficient to treat this problem. "
	    "Use relaxation or decrease epsilon with -pc_asa_epsilon");
  } else {
    /* set the number of relaxations to asa->mu from asa->mu_initial */
    ierr = PCSetupSmoothersOnLevel_ASA(asa, asa_lev, asa->mu);CHKERRQ(ierr);

    /* Let's do some multigrid ! */
    sufficiently_coarsened = PETSC_FALSE;

    /* do the whole initialization stage loop */
    while (!sufficiently_coarsened) {
      ierr = PetscPrintf(asa_lev->comm, "Initialization stage: creating level %D\n", asa_lev->level+1);CHKERRQ(ierr);

      /* (a) Set candidate matrix B_l = x_l */
      /* get the correct vector sizes and data */
      ierr = VecGetSize(asa_lev->x, &vec_size);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(asa_lev->x, &loc_vec_low, &loc_vec_high);CHKERRQ(ierr);
      vec_loc_size = loc_vec_high - loc_vec_low;

      /* create matrix for candidates */
      ierr = MatCreateDense(asa_lev->comm, vec_loc_size, PETSC_DECIDE, vec_size, asa->max_cand_vecs, PETSC_NULL, &(asa_lev->B));CHKERRQ(ierr);
      /* set the first column */
      ierr = PCAddCandidateToB_ASA(asa_lev->B, 0, asa_lev->x, asa_lev->A);CHKERRQ(ierr);
      asa_lev->cand_vecs = 1;

      /* create next level */
      ierr = PCCreateLevel_ASA(&(asa_lev->next), asa_lev->level+1,  asa_lev->comm, asa_lev, PETSC_NULL, asa->ksptype_smooth, asa->pctype_smooth);CHKERRQ(ierr);
      asa_next_lev = asa_lev->next;

      /* (b) Create nodal aggregates A_i^l */
      ierr = PCCreateAggregates_ASA(asa_lev);CHKERRQ(ierr);

      /* (c) Define tentatative prolongator P_{l+1}^l and candidate matrix B_{l+1}
	     using P_{l+1}^l B_{l+1} = B_l and (P_{l+1}^l)^T P_{l+1}^l = I */
      ierr = PCCreateTransferOp_ASA(asa_lev, PETSC_FALSE);CHKERRQ(ierr);

      /* future WORK: set correct fill ratios for all the operations below */
      ierr = MatMatMult(asa_lev->Pt, asa_lev->B, MAT_INITIAL_MATRIX, 1, &(asa_next_lev->B));CHKERRQ(ierr);
      asa_next_lev->cand_vecs = asa_lev->cand_vecs;

      /* (d) Define prolongator I_{l+1}^l = S_l P_{l+1}^l */
      ierr = PCSmoothProlongator_ASA(asa_lev);CHKERRQ(ierr);

      /* (e) Define coarse matrix A_{l+1} = (I_{l+1}^l)^T A_l I_{l+1}^l */
            ierr = MatMatMult(asa_lev->A, asa_lev->smP, MAT_INITIAL_MATRIX, 1.0, &AI);CHKERRQ(ierr);
      ierr = MatMatMult(asa_lev->smPt, AI, MAT_INITIAL_MATRIX, 1.0, &(asa_next_lev->A));CHKERRQ(ierr);
      ierr = MatDestroy(&AI);CHKERRQ(ierr);
      /*      ierr = MatPtAP(asa_lev->A, asa_lev->smP, MAT_INITIAL_MATRIX, 1, &(asa_next_lev->A));CHKERRQ(ierr); */
      ierr = MatGetSize(asa_next_lev->A, PETSC_NULL, &(asa_next_lev->size));CHKERRQ(ierr);
      ierr = PCComputeSpectralRadius_ASA(asa_next_lev);CHKERRQ(ierr);
      ierr = PCSetupSmoothersOnLevel_ASA(asa, asa_next_lev, asa->mu);CHKERRQ(ierr);

      /* coarse enough for direct solver? */
      ierr = MatGetSize(asa_next_lev->A, &i, &j);CHKERRQ(ierr);
      if (PetscMax(i,j) <= asa->direct_solver) {
	ierr = PetscPrintf(asa_lev->comm, "Level %D can be treated directly.\n"
			   "Algorithm will use %D levels.\n", asa_next_lev->level,
			   asa_next_lev->level);CHKERRQ(ierr);
	break; /* go to step 5 */
      }

      if (!skip_steps_f_i) {
	/* (f) Set x_{l+1} = B_{l+1}, we just compute it again */
        ierr = VecDestroy(&(asa_next_lev->x));CHKERRQ(ierr);
	ierr = MatGetVecs(asa_lev->P, &(asa_next_lev->x), 0);CHKERRQ(ierr);
	ierr = MatMult(asa_lev->Pt, asa_lev->x, asa_next_lev->x);CHKERRQ(ierr);

/* 	/\* (g) Make copy \hat{x}_{l+1} = x_{l+1} *\/ */
/* 	ierr = VecDuplicate(asa_next_lev->x, &xhat);CHKERRQ(ierr); */
/* 	ierr = VecCopy(asa_next_lev->x, xhat);CHKERRQ(ierr); */
	
	/* Create b_{l+1} */
        ierr = VecDestroy(&(asa_next_lev->b));CHKERRQ(ierr);
	ierr = MatGetVecs(asa_next_lev->A, &(asa_next_lev->b), 0);
	ierr = VecSet(asa_next_lev->b, 0.0);

	/* (h) Relax mu times on A_{l+1} x = 0 */
	/* compute old norm */
	ierr = MatGetVecs(asa_next_lev->A, 0, &ax);CHKERRQ(ierr);
	ierr = MatMult(asa_next_lev->A, asa_next_lev->x, ax);CHKERRQ(ierr);
	ierr = VecDot(asa_next_lev->x, ax, &tmp);CHKERRQ(ierr);
	prevnorm = PetscAbsScalar(tmp);
	ierr = PetscPrintf(asa_next_lev->comm, "Residual norm of starting guess on level %D: %f\n", asa_next_lev->level, prevnorm);CHKERRQ(ierr);
	/* apply mu relaxations: WORK, make sure that mu is set correctly */
	ierr = KSPSolve(asa_next_lev->smoothd, asa_next_lev->b, asa_next_lev->x);CHKERRQ(ierr);
	/* compute new norm */
	ierr = MatMult(asa_next_lev->A, asa_next_lev->x, ax);CHKERRQ(ierr);
	ierr = VecDot(asa_next_lev->x, ax, &tmp);CHKERRQ(ierr);
	norm = PetscAbsScalar(tmp);
	ierr = VecDestroy(&(ax));CHKERRQ(ierr);
	ierr = PetscPrintf(asa_next_lev->comm, "Residual norm after Richardson iteration  on level %D: %f\n", asa_next_lev->level, norm);CHKERRQ(ierr);
	/* (i) Check if it already converges by itself */
	if (norm/prevnorm <= pow(asa->epsilon, (PetscReal) asa->mu)) {
	  /* relaxation reduces error sufficiently */
	  skip_steps_f_i = PETSC_TRUE;
	}
      }
      /* (j) go to next coarser level */
      l++;
      asa_lev = asa_next_lev;
    }
    /* Step 5. */
    asa->levels = asa_next_lev->level; /* WORK: correct? */

    /* Set up direct solvers on coarsest level */
    if (asa_next_lev->smoothd != asa_next_lev->smoothu) {
      if (asa_next_lev->smoothu) { KSPDestroy(&asa_next_lev->smoothu);CHKERRQ(ierr); }
    }
    ierr = KSPSetType(asa_next_lev->smoothd, asa->ksptype_direct);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)(asa_next_lev->smoothd), KSPRICHARDSON, &isrichardson);CHKERRQ(ierr);
    if (isrichardson) {
      ierr = KSPSetInitialGuessNonzero(asa_next_lev->smoothd, PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = KSPSetInitialGuessNonzero(asa_next_lev->smoothd, PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = KSPGetPC(asa_next_lev->smoothd, &coarse_pc);CHKERRQ(ierr);
    ierr = PCSetType(coarse_pc, asa->pctype_direct);CHKERRQ(ierr);
    asa_next_lev->smoothu = asa_next_lev->smoothd;
    ierr = PCSetupDirectSolversOnLevel_ASA(asa, asa_next_lev, asa->nu);CHKERRQ(ierr);

    /* update finest-level candidate matrix B_1 = I_2^1 I_3^2 ... I_{L-1}^{L-2} x_{L-1} */
    if (!asa_lev->prev) {
      /* just one relaxation level */
      ierr = VecDuplicate(asa_lev->x, &cand_vec);CHKERRQ(ierr);
      ierr = VecCopy(asa_lev->x, cand_vec);CHKERRQ(ierr);
    } else {
      /* interpolate up the chain */
      cand_vec = asa_lev->x;
      asa_lev->x = 0;
      while(asa_lev->prev) {
	/* interpolate to higher level */
	ierr = MatGetVecs(asa_lev->prev->smP, 0, &cand_vec_new);CHKERRQ(ierr);
	ierr = MatMult(asa_lev->prev->smP, cand_vec, cand_vec_new);CHKERRQ(ierr);
	ierr = VecDestroy(&(cand_vec));CHKERRQ(ierr);
	cand_vec = cand_vec_new;
	
	/* destroy all working vectors on the way */
	ierr = VecDestroy(&(asa_lev->x));CHKERRQ(ierr);
	ierr = VecDestroy(&(asa_lev->b));CHKERRQ(ierr);

	/* move to next higher level */
	asa_lev = asa_lev->prev;
      }
    }
    /* set the first column of B1 */
    ierr = PCAddCandidateToB_ASA(asa_lev->B, 0, cand_vec, asa_lev->A);CHKERRQ(ierr);
    ierr = VecDestroy(&(cand_vec));CHKERRQ(ierr);

    /* Step 6. Create V-cycle */
    ierr = PCCreateVcycle_ASA(asa);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PC_InitializationStage_ASA,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApplyVcycleOnLevel_ASA - Applies current V-cycle

   Input Parameters:
+  asa_lev - the current level we should recurse on
-  gamma - the number of recursive cycles we should run

*/
#undef __FUNCT__
#define __FUNCT__ "PCApplyVcycleOnLevel_ASA"
PetscErrorCode PCApplyVcycleOnLevel_ASA(PC_ASA_level *asa_lev, PetscInt gamma)
{
  PetscErrorCode ierr;
  PC_ASA_level   *asa_next_lev;
  PetscInt       g;

  PetscFunctionBegin;
  if (!asa_lev) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL, "Level is empty in PCApplyVcycleOnLevel_ASA");
  asa_next_lev = asa_lev->next;

  if (asa_next_lev) {
    /* 1. Presmoothing */
    ierr = KSPSolve(asa_lev->smoothd, asa_lev->b, asa_lev->x);CHKERRQ(ierr);
    /* 2. Coarse grid corrections */
/*     ierr = MatGetVecs(asa_lev->A, 0, &tmp);CHKERRQ(ierr); */
/*     ierr = MatGetVecs(asa_lev->smP, &(asa_next_lev->b), 0);CHKERRQ(ierr); */
/*     ierr = MatGetVecs(asa_next_lev->A, &(asa_next_lev->x), 0);CHKERRQ(ierr); */
    for (g=0; g<gamma; g++) {
      /* (a) get coarsened b_{l+1} = (I_{l+1}^l)^T (b_l - A_l x_l) */
      ierr = MatMult(asa_lev->A, asa_lev->x, asa_lev->r);CHKERRQ(ierr);
      ierr = VecAYPX(asa_lev->r, -1.0, asa_lev->b);CHKERRQ(ierr);
      ierr = MatMult(asa_lev->smPt, asa_lev->r, asa_next_lev->b);CHKERRQ(ierr);

      /* (b) Set x_{l+1} = 0 and recurse */
      ierr = VecSet(asa_next_lev->x, 0.0);CHKERRQ(ierr);
      ierr = PCApplyVcycleOnLevel_ASA(asa_next_lev, gamma);CHKERRQ(ierr);

      /* (c) correct solution x_l = x_l + I_{l+1}^l x_{l+1} */
      ierr = MatMultAdd(asa_lev->smP, asa_next_lev->x, asa_lev->x, asa_lev->x);CHKERRQ(ierr);
    }
/*     ierr = VecDestroy(&(asa_lev->r));CHKERRQ(ierr); */
/*     /\* discard x_{l+1}, b_{l+1} *\/ */
/*     ierr = VecDestroy(&(asa_next_lev->x));CHKERRQ(ierr); */
/*     ierr = VecDestroy(&(asa_next_lev->b));CHKERRQ(ierr); */

    /* 3. Postsmoothing */
    ierr = KSPSolve(asa_lev->smoothu, asa_lev->b, asa_lev->x);CHKERRQ(ierr);
  } else {
    /* Base case: solve directly */
    ierr = KSPSolve(asa_lev->smoothd, asa_lev->b, asa_lev->x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCGeneralSetupStage_ASA - Applies the ASA preconditioner to a vector. Algorithm
                             4 from the ASA paper

   Input Parameters:
+  asa - the data structure for the ASA algorithm
-  cand - a possible candidate vector, if PETSC_NULL, will be constructed randomly

   Output Parameters:
.  cand_added - PETSC_TRUE, if new candidate vector added, PETSC_FALSE otherwise
*/
#undef __FUNCT__
#define __FUNCT__ "PCGeneralSetupStage_ASA"
PetscErrorCode PCGeneralSetupStage_ASA(PC_ASA *asa, Vec cand, PetscBool  *cand_added)
{
  PetscErrorCode ierr;
  PC_ASA_level   *asa_lev, *asa_next_lev;

  PetscRandom    rctx;     /* random number generator context */
  PetscReal      r;
  PetscScalar    rs;
  PetscBool      nd_fast;

  Vec            ax;
  PetscScalar    tmp;
  PetscReal      norm, prevnorm = 0.0;
  PetscInt       c;

  PetscInt       loc_vec_low, loc_vec_high;
  PetscInt       i;

  PetscBool      skip_steps_d_j = PETSC_FALSE;

  PetscInt       *idxm, *idxn;
  PetscScalar    *v;

  Mat            AI;

  Vec            cand_vec, cand_vec_new;

  PetscFunctionBegin;
  *cand_added = PETSC_FALSE;

  asa_lev = asa->levellist;
  if (asa_lev == 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL, "No levels found in PCGeneralSetupStage_ASA");
  asa_next_lev = asa_lev->next;
  if (asa_next_lev == 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL, "Just one level, not implemented yet");

  ierr = PetscPrintf(asa_lev->comm, "General setup stage\n");CHKERRQ(ierr);

  ierr = PetscLogEventBegin(PC_GeneralSetupStage_ASA,0,0,0,0);CHKERRQ(ierr);

  /* 1. If max. dof per node on level 2 equals K, stop */
  if (asa_next_lev->cand_vecs >= asa->max_dof_lev_2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
		       "Maximum dof on level 2 reached: %D\n"
		       "Consider increasing this limit by setting it with -pc_asa_max_dof_lev_2\n",
		       asa->max_dof_lev_2);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* 2. Create copy of B_1 (skipped, we just replace the last column in step 8.) */

  if (!cand) {
    /* 3. Select a random x_1 */
    ierr = VecDestroy(&(asa_lev->x));CHKERRQ(ierr);
    ierr = MatGetVecs(asa_lev->A, &(asa_lev->x), 0);
    ierr = PetscRandomCreate(asa_lev->comm,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(asa_lev->x, &loc_vec_low, &loc_vec_high);CHKERRQ(ierr);
    for (i=loc_vec_low; i<loc_vec_high; i++) {
      ierr = PetscRandomGetValueReal(rctx, &r);CHKERRQ(ierr);
      rs = r;
      ierr = VecSetValues(asa_lev->x, 1, &i, &rs, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(asa_lev->x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(asa_lev->x);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  } else {
    ierr = VecDestroy(&(asa_lev->x));CHKERRQ(ierr);
    ierr = VecDuplicate(cand, &(asa_lev->x));CHKERRQ(ierr);
    ierr = VecCopy(cand, asa_lev->x);CHKERRQ(ierr);
  }

  /* create right hand side */
  ierr = VecDestroy(&(asa_lev->b));CHKERRQ(ierr);
  ierr = MatGetVecs(asa_lev->A, &(asa_lev->b), 0);
  ierr = VecSet(asa_lev->b, 0.0);

  /* Apply mu iterations of current V-cycle */
  nd_fast = PETSC_FALSE;
  ierr = MatGetVecs(asa_lev->A, 0, &ax);CHKERRQ(ierr);
  for (c=0; c<asa->mu; c++) {
    ierr = PCApplyVcycleOnLevel_ASA(asa_lev, asa->gamma);CHKERRQ(ierr);

    ierr = MatMult(asa_lev->A, asa_lev->x, ax);CHKERRQ(ierr);
    ierr = VecDot(asa_lev->x, ax, &tmp);CHKERRQ(ierr);
    norm = PetscAbsScalar(tmp);
    if (c>0) {
      if (norm/prevnorm < asa->epsilon) {
	nd_fast = PETSC_TRUE;
	break;
      }
    }
    prevnorm = norm;
  }
  ierr = VecDestroy(&(ax));CHKERRQ(ierr);

  /* 4. If energy norm decreases sufficiently fast, then stop */
  if (nd_fast) {
    ierr = PetscPrintf(asa_lev->comm, "nd_fast is true\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* 5. Update B_1, by adding new column x_1 */
  if (asa_lev->cand_vecs >= asa->max_cand_vecs) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM, "Number of candidate vectors will exceed allocated storage space");
  } else {
    ierr = PetscPrintf(asa_lev->comm, "Adding candidate vector %D\n", asa_lev->cand_vecs+1);CHKERRQ(ierr);
  }
  ierr = PCAddCandidateToB_ASA(asa_lev->B, asa_lev->cand_vecs, asa_lev->x, asa_lev->A);CHKERRQ(ierr);
  *cand_added = PETSC_TRUE;
  asa_lev->cand_vecs++;

  /* 6. loop over levels */
  while(asa_next_lev && asa_next_lev->next) {
    ierr = PetscPrintf(asa_lev->comm, "General setup stage: processing level %D\n", asa_next_lev->level);CHKERRQ(ierr);
    /* (a) define B_{l+1} and P_{l+1}^L */
    /* construct P_{l+1}^l */
    ierr = PCCreateTransferOp_ASA(asa_lev, PETSC_FALSE);CHKERRQ(ierr);

    /* construct B_{l+1} */
    ierr = MatDestroy(&(asa_next_lev->B));CHKERRQ(ierr);
    ierr = MatMatMult(asa_lev->Pt, asa_lev->B, MAT_INITIAL_MATRIX, 1.0, &(asa_next_lev->B));CHKERRQ(ierr);
    /* do not increase asa_next_lev->cand_vecs until step (j) */

    /* (b) construct prolongator I_{l+1}^l = S_l P_{l+1}^l */
    ierr = PCSmoothProlongator_ASA(asa_lev);CHKERRQ(ierr);
							
    /* (c) construct coarse matrix A_{l+1} = (I_{l+1}^l)^T A_l I_{l+1}^l */
    ierr = MatDestroy(&(asa_next_lev->A));CHKERRQ(ierr);
       ierr = MatMatMult(asa_lev->A, asa_lev->smP, MAT_INITIAL_MATRIX, 1.0, &AI);CHKERRQ(ierr);
    ierr = MatMatMult(asa_lev->smPt, AI, MAT_INITIAL_MATRIX, 1.0, &(asa_next_lev->A));CHKERRQ(ierr);
    ierr = MatDestroy(&AI);CHKERRQ(ierr);
				 /* ierr = MatPtAP(asa_lev->A, asa_lev->smP, MAT_INITIAL_MATRIX, 1, &(asa_next_lev->A));CHKERRQ(ierr); */
    ierr = MatGetSize(asa_next_lev->A, PETSC_NULL, &(asa_next_lev->size));CHKERRQ(ierr);
    ierr = PCComputeSpectralRadius_ASA(asa_next_lev);CHKERRQ(ierr);
    ierr = PCSetupSmoothersOnLevel_ASA(asa, asa_next_lev, asa->mu);CHKERRQ(ierr);

    if (! skip_steps_d_j) {
      /* (d) get vector x_{l+1} from last column in B_{l+1} */
      ierr = VecDestroy(&(asa_next_lev->x));CHKERRQ(ierr);
      ierr = MatGetVecs(asa_next_lev->B, 0, &(asa_next_lev->x));CHKERRQ(ierr);

      ierr = VecGetOwnershipRange(asa_next_lev->x, &loc_vec_low, &loc_vec_high);CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(PetscInt)*(loc_vec_high-loc_vec_low), &idxm);CHKERRQ(ierr);
      for (i=loc_vec_low; i<loc_vec_high; i++)
	idxm[i-loc_vec_low] = i;
      ierr = PetscMalloc(sizeof(PetscInt)*1, &idxn);CHKERRQ(ierr);
      idxn[0] = asa_next_lev->cand_vecs;

      ierr = PetscMalloc(sizeof(PetscScalar)*(loc_vec_high-loc_vec_low), &v);CHKERRQ(ierr);
      ierr = MatGetValues(asa_next_lev->B, loc_vec_high-loc_vec_low, idxm, 1, idxn, v);CHKERRQ(ierr);

      ierr = VecSetValues(asa_next_lev->x, loc_vec_high-loc_vec_low, idxm, v, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(asa_next_lev->x);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(asa_next_lev->x);CHKERRQ(ierr);

      ierr = PetscFree(v);CHKERRQ(ierr);
      ierr = PetscFree(idxm);CHKERRQ(ierr);
      ierr = PetscFree(idxn);CHKERRQ(ierr);

      /* (e) create bridge transfer operator P_{l+2}^{l+1}, by using the previously
	 computed candidates */
      ierr = PCCreateTransferOp_ASA(asa_next_lev, PETSC_TRUE);CHKERRQ(ierr);

      /* (f) construct bridging prolongator I_{l+2}^{l+1} = S_{l+1} P_{l+2}^{l+1} */
      ierr = PCSmoothProlongator_ASA(asa_next_lev);CHKERRQ(ierr);

      /* (g) compute <A_{l+1} x_{l+1}, x_{l+1}> and save it */
      ierr = MatGetVecs(asa_next_lev->A, 0, &ax);CHKERRQ(ierr);
      ierr = MatMult(asa_next_lev->A, asa_next_lev->x, ax);CHKERRQ(ierr);
      ierr = VecDot(asa_next_lev->x, ax, &tmp);CHKERRQ(ierr);
      prevnorm = PetscAbsScalar(tmp);
      ierr = VecDestroy(&(ax));CHKERRQ(ierr);

      /* (h) apply mu iterations of current V-cycle */
      /* set asa_next_lev->b */
      ierr = VecDestroy(&(asa_next_lev->b));CHKERRQ(ierr);
      ierr = VecDestroy(&(asa_next_lev->r));CHKERRQ(ierr);
      ierr = MatGetVecs(asa_next_lev->A, &(asa_next_lev->b), &(asa_next_lev->r));
      ierr = VecSet(asa_next_lev->b, 0.0);
      /* apply V-cycle */
      for (c=0; c<asa->mu; c++) {
	ierr = PCApplyVcycleOnLevel_ASA(asa_next_lev, asa->gamma);CHKERRQ(ierr);
      }

      /* (i) check convergence */
      /* compute <A_{l+1} x_{l+1}, x_{l+1}> and save it */
      ierr = MatGetVecs(asa_next_lev->A, 0, &ax);CHKERRQ(ierr);
      ierr = MatMult(asa_next_lev->A, asa_next_lev->x, ax);CHKERRQ(ierr);
      ierr = VecDot(asa_next_lev->x, ax, &tmp);CHKERRQ(ierr);
      norm = PetscAbsScalar(tmp);
      ierr = VecDestroy(&(ax));CHKERRQ(ierr);

      if (norm/prevnorm <= pow(asa->epsilon, (PetscReal) asa->mu)) skip_steps_d_j = PETSC_TRUE;

      /* (j) update candidate B_{l+1} */
      ierr = PCAddCandidateToB_ASA(asa_next_lev->B, asa_next_lev->cand_vecs, asa_next_lev->x, asa_next_lev->A);CHKERRQ(ierr);
      asa_next_lev->cand_vecs++;
    }
    /* go to next level */
    asa_lev = asa_lev->next;
    asa_next_lev = asa_next_lev->next;
  }

  /* 7. update the fine-level candidate */
  if (! asa_lev->prev) {
    /* just one coarsening level */
    ierr = VecDuplicate(asa_lev->x, &cand_vec);CHKERRQ(ierr);
    ierr = VecCopy(asa_lev->x, cand_vec);CHKERRQ(ierr);
  } else {
    cand_vec = asa_lev->x;
    asa_lev->x = 0;
    while(asa_lev->prev) {
      /* interpolate to higher level */
      ierr = MatGetVecs(asa_lev->prev->smP, 0, &cand_vec_new);CHKERRQ(ierr);
      ierr = MatMult(asa_lev->prev->smP, cand_vec, cand_vec_new);CHKERRQ(ierr);
      ierr = VecDestroy(&(cand_vec));CHKERRQ(ierr);
      cand_vec = cand_vec_new;

      /* destroy all working vectors on the way */
      ierr = VecDestroy(&(asa_lev->x));CHKERRQ(ierr);
      ierr = VecDestroy(&(asa_lev->b));CHKERRQ(ierr);

      /* move to next higher level */
      asa_lev = asa_lev->prev;
    }
  }
  /* 8. update B_1 by setting the last column of B_1 */
  ierr = PCAddCandidateToB_ASA(asa_lev->B, asa_lev->cand_vecs-1, cand_vec, asa_lev->A);CHKERRQ(ierr);
  ierr = VecDestroy(&(cand_vec));CHKERRQ(ierr);

  /* 9. create V-cycle */
  ierr = PCCreateVcycle_ASA(asa);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(PC_GeneralSetupStage_ASA,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCConstructMultigrid_ASA - creates the multigrid preconditionier, this is a fairly
   involved process, which runs extensive testing to compute good candidate vectors

   Input Parameters:
.  pc - the preconditioner context

 */
#undef __FUNCT__
#define __FUNCT__ "PCConstructMultigrid_ASA"
PetscErrorCode PCConstructMultigrid_ASA(PC pc)
{
  PetscErrorCode ierr;
  PC_ASA         *asa = (PC_ASA*)pc->data;
  PC_ASA_level   *asa_lev;
  PetscInt       i, ls, le;
  PetscScalar    *d;
  PetscBool      zeroflag = PETSC_FALSE;
  PetscReal      rnorm, rnorm_start;
  PetscReal      rq, rq_prev;
  PetscScalar    rq_nom, rq_denom;
  PetscBool      cand_added;
  PetscRandom    rctx;

  PetscFunctionBegin;

  /* check if we should scale with diagonal */
  if (asa->scale_diag) {
    /* Get diagonal scaling factors */
    ierr = MatGetVecs(pc->pmat,&(asa->invsqrtdiag),0);CHKERRQ(ierr);
    ierr = MatGetDiagonal(pc->pmat,asa->invsqrtdiag);CHKERRQ(ierr);
    /* compute (inverse) sqrt of diagonal */
    ierr = VecGetOwnershipRange(asa->invsqrtdiag, &ls, &le);CHKERRQ(ierr);
    ierr = VecGetArray(asa->invsqrtdiag, &d);CHKERRQ(ierr);
    for (i=0; i<le-ls; i++) {
      if (d[i] == 0.0) {
	d[i]     = 1.0;
	zeroflag = PETSC_TRUE;
      } else {
	d[i] = 1./PetscSqrtReal(PetscAbsScalar(d[i]));
      }
    }
    ierr = VecRestoreArray(asa->invsqrtdiag,&d);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(asa->invsqrtdiag);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(asa->invsqrtdiag);CHKERRQ(ierr);
    if (zeroflag) {
      ierr = PetscInfo(pc,"Zero detected in diagonal of matrix, using 1 at those locations\n");CHKERRQ(ierr);
    }

    /* scale the matrix and store it: D^{-1/2} A D^{-1/2} */
    ierr = MatDuplicate(pc->pmat, MAT_COPY_VALUES, &(asa->A)); /* probably inefficient */
    ierr = MatDiagonalScale(asa->A, asa->invsqrtdiag, asa->invsqrtdiag);CHKERRQ(ierr);
  } else {
    /* don't scale */
    asa->A = pc->pmat;
  }
  /* Initialization stage */
  ierr = PCInitializationStage_ASA(pc, PETSC_NULL);CHKERRQ(ierr);

  /* get first level */
  asa_lev = asa->levellist;

  ierr = PetscRandomCreate(asa->comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(asa_lev->x,rctx);CHKERRQ(ierr);

  /* compute starting residual */
  ierr = VecDestroy(&(asa_lev->r));CHKERRQ(ierr);
  ierr = MatGetVecs(asa_lev->A, PETSC_NULL, &(asa_lev->r));CHKERRQ(ierr);
  ierr = MatMult(asa_lev->A, asa_lev->x, asa_lev->r);CHKERRQ(ierr);
  /* starting residual norm */
  ierr = VecNorm(asa_lev->r, NORM_2, &rnorm_start);CHKERRQ(ierr);
  /* compute Rayleigh quotients */
  ierr = VecDot(asa_lev->x, asa_lev->r, &rq_nom);CHKERRQ(ierr);
  ierr = VecDot(asa_lev->x, asa_lev->x, &rq_denom);CHKERRQ(ierr);
  rq_prev = PetscAbsScalar(rq_nom / rq_denom);

  /* check if we have to add more candidates */
  for (i=0; i<asa->max_it; i++) {
    if (asa_lev->cand_vecs >= asa->max_cand_vecs) {
      /* reached limit for candidate vectors */
      break;
    }
    /* apply V-cycle */
    ierr = PCApplyVcycleOnLevel_ASA(asa_lev, asa->gamma);CHKERRQ(ierr);
    /* check convergence */
    ierr = MatMult(asa_lev->A, asa_lev->x, asa_lev->r);CHKERRQ(ierr);
    ierr = VecNorm(asa_lev->r, NORM_2, &rnorm);CHKERRQ(ierr);
    ierr = PetscPrintf(asa->comm, "After %D iterations residual norm is %f\n", i+1, rnorm);CHKERRQ(ierr);
    if (rnorm < rnorm_start*(asa->rtol) || rnorm < asa->abstol) {
      /* convergence */
      break;
    }
    /* compute new Rayleigh quotient */
    ierr = VecDot(asa_lev->x, asa_lev->r, &rq_nom);CHKERRQ(ierr);
    ierr = VecDot(asa_lev->x, asa_lev->x, &rq_denom);CHKERRQ(ierr);
    rq = PetscAbsScalar(rq_nom / rq_denom);
    ierr = PetscPrintf(asa->comm, "After %D iterations Rayleigh quotient of residual is %f\n", i+1, rq);CHKERRQ(ierr);
    /* test Rayleigh quotient decrease and add more candidate vectors if necessary */
    if (i && (rq > asa->rq_improve*rq_prev)) {
      /* improve interpolation by adding another candidate vector */
      ierr = PCGeneralSetupStage_ASA(asa, asa_lev->r, &cand_added);CHKERRQ(ierr);
      if (!cand_added) {
	/* either too many candidates for storage or cycle is already effective */
	ierr = PetscPrintf(asa->comm, "either too many candidates for storage or cycle is already effective\n");CHKERRQ(ierr);
	break;
      }
      ierr = VecSetRandom(asa_lev->x, rctx);CHKERRQ(ierr);
      rq_prev = rq*10000.; /* give the new V-cycle some grace period */
    } else {
      rq_prev = rq;
    }
  }

  ierr = VecDestroy(&(asa_lev->x));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa_lev->b));CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  asa->multigrid_constructed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_ASA - Applies the ASA preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__
#define __FUNCT__ "PCApply_ASA"
PetscErrorCode PCApply_ASA(PC pc,Vec x,Vec y)
{
  PC_ASA         *asa = (PC_ASA*)pc->data;
  PC_ASA_level   *asa_lev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!asa->multigrid_constructed) {
    ierr = PCConstructMultigrid_ASA(pc);CHKERRQ(ierr);
  }

  /* get first level */
  asa_lev = asa->levellist;

  /* set the right hand side */
  ierr = VecDuplicate(x, &(asa->b));CHKERRQ(ierr);
  ierr = VecCopy(x, asa->b);CHKERRQ(ierr);
  /* set starting vector */
  ierr = VecDestroy(&(asa->x));CHKERRQ(ierr);
  ierr = MatGetVecs(asa->A, &(asa->x), PETSC_NULL);CHKERRQ(ierr);
  ierr = VecSet(asa->x, 0.0);CHKERRQ(ierr);

  /* set vectors */
  asa_lev->x = asa->x;
  asa_lev->b = asa->b;

  ierr = PCApplyVcycleOnLevel_ASA(asa_lev, asa->gamma);CHKERRQ(ierr);

  /* Return solution */
  ierr = VecCopy(asa->x, y);CHKERRQ(ierr);

  /* delete working vectors */
  ierr = VecDestroy(&(asa->x));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa->b));CHKERRQ(ierr);
  asa_lev->x = PETSC_NULL;
  asa_lev->b = PETSC_NULL;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApplyRichardson_ASA - Applies the ASA iteration to solve a linear system

   Input Parameters:
.  pc - the preconditioner context
.  b - the right hand side

   Output Parameter:
.  x - output vector

  DOES NOT WORK!!!!!

 */
#undef __FUNCT__
#define __FUNCT__ "PCApplyRichardson_ASA"
PetscErrorCode PCApplyRichardson_ASA(PC pc,Vec b,Vec x,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool  guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_ASA         *asa = (PC_ASA*)pc->data;
  PC_ASA_level   *asa_lev;
  PetscInt       i;
  PetscReal      rnorm, rnorm_start;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (! asa->multigrid_constructed) {
    ierr = PCConstructMultigrid_ASA(pc);CHKERRQ(ierr);
  }

  /* get first level */
  asa_lev = asa->levellist;

  /* set the right hand side */
  ierr = VecDuplicate(b, &(asa->b));CHKERRQ(ierr);
  if (asa->scale_diag) {
    ierr = VecPointwiseMult(asa->b, asa->invsqrtdiag, b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b, asa->b);CHKERRQ(ierr);
  }
  /* set starting vector */
  ierr = VecDuplicate(x, &(asa->x));CHKERRQ(ierr);
  ierr = VecCopy(x, asa->x);CHKERRQ(ierr);

  /* compute starting residual */
  ierr = VecDestroy(&(asa->r));CHKERRQ(ierr);
  ierr = MatGetVecs(asa->A, &(asa->r), PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMult(asa->A, asa->x, asa->r);CHKERRQ(ierr);
  ierr = VecAYPX(asa->r, -1.0, asa->b);CHKERRQ(ierr);
  /* starting residual norm */
  ierr = VecNorm(asa->r, NORM_2, &rnorm_start);CHKERRQ(ierr);

  /* set vectors */
  asa_lev->x = asa->x;
  asa_lev->b = asa->b;

  *reason = PCRICHARDSON_CONVERGED_ITS;
  /* **************** Full algorithm loop *********************************** */
  for (i=0; i<its; i++) {
    /* apply V-cycle */
    ierr = PCApplyVcycleOnLevel_ASA(asa_lev, asa->gamma);CHKERRQ(ierr);
    /* check convergence */
    ierr = MatMult(asa->A, asa->x, asa->r);CHKERRQ(ierr);
    ierr = VecAYPX(asa->r, -1.0, asa->b);CHKERRQ(ierr);
    ierr = VecNorm(asa->r, NORM_2, &rnorm);CHKERRQ(ierr);
    ierr = PetscPrintf(asa->comm, "After %D iterations residual norm is %f\n", i+1, rnorm);CHKERRQ(ierr);
    if (rnorm < rnorm_start*(rtol)) {
      *reason = PCRICHARDSON_CONVERGED_RTOL;
      break;
    } else if (rnorm < asa->abstol) {
      *reason = PCRICHARDSON_CONVERGED_ATOL;
      break;
    } else if (rnorm > rnorm_start*(dtol)) {
      *reason = PCRICHARDSON_DIVERGED_DTOL;
      break;
    }
  }
  *outits = i;

  /* Return solution */
  if (asa->scale_diag) {
    ierr = VecPointwiseMult(x, asa->x, asa->invsqrtdiag);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x, asa->x);CHKERRQ(ierr);
  }

  /* delete working vectors */
  ierr = VecDestroy(&(asa->x));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa->b));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa->r));CHKERRQ(ierr);
  asa_lev->x = PETSC_NULL;
  asa_lev->b = PETSC_NULL;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_ASA - Destroys the private context for the ASA preconditioner
   that was created with PCCreate_ASA().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_ASA"
static PetscErrorCode PCDestroy_ASA(PC pc)
{
  PC_ASA         *asa;
  PC_ASA_level   *asa_lev;
  PC_ASA_level   *asa_next_level;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  asa = (PC_ASA*)pc->data;
  asa_lev = asa->levellist;

  /* Delete top level data */
  ierr = PetscFree(asa->ksptype_smooth);CHKERRQ(ierr);
  ierr = PetscFree(asa->pctype_smooth);CHKERRQ(ierr);
  ierr = PetscFree(asa->ksptype_direct);CHKERRQ(ierr);
  ierr = PetscFree(asa->pctype_direct);CHKERRQ(ierr);
  ierr = PetscFree(asa->coarse_mat_type);CHKERRQ(ierr);

  /* this is destroyed by the levels below  */
/*   ierr = MatDestroy(&(asa->A));CHKERRQ(ierr); */
  ierr = VecDestroy(&(asa->invsqrtdiag));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa->b));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa->x));CHKERRQ(ierr);
  ierr = VecDestroy(&(asa->r));CHKERRQ(ierr);

  /* Destroy each of the levels */
  while(asa_lev) {
    asa_next_level = asa_lev->next;
    ierr = PCDestroyLevel_ASA(asa_lev);CHKERRQ(ierr);
    asa_lev = asa_next_level;
  }

  ierr = PetscFree(asa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_ASA"
static PetscErrorCode PCSetFromOptions_ASA(PC pc)
{
  PC_ASA         *asa = (PC_ASA*)pc->data;
  PetscBool      flg;
  PetscErrorCode ierr;
  char           type[20];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);

  ierr = PetscOptionsHead("ASA options");CHKERRQ(ierr);
  /* convergence parameters */
  ierr = PetscOptionsInt("-pc_asa_nu","Number of cycles to run smoother","No manual page yet",asa->nu,&(asa->nu),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_asa_gamma","Number of cycles to run coarse grid correction","No manual page yet",asa->gamma,&(asa->gamma),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_asa_epsilon","Tolerance for the relaxation method","No manual page yet",asa->epsilon,&(asa->epsilon),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_asa_mu","Number of cycles to relax in setup stages","No manual page yet",asa->mu,&(asa->mu),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_asa_mu_initial","Number of cycles to relax for generating first candidate vector","No manual page yet",asa->mu_initial,&(asa->mu_initial),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_asa_direct_solver","For which matrix size should we use the direct solver?","No manual page yet",asa->direct_solver,&(asa->direct_solver),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_asa_scale_diag","Should we scale the matrix with the inverse of its diagonal?","No manual page yet",asa->scale_diag,&(asa->scale_diag),&flg);CHKERRQ(ierr);
  /* type of smoother used */
  ierr = PetscOptionsList("-pc_asa_smoother_ksp_type","The type of KSP to be used in the smoothers","No manual page yet",KSPList,asa->ksptype_smooth,type,20,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(asa->ksptype_smooth);CHKERRQ(ierr);
    ierr = PetscStrallocpy(type,&(asa->ksptype_smooth));CHKERRQ(ierr);
  }
  ierr = PetscOptionsList("-pc_asa_smoother_pc_type","The type of PC to be used in the smoothers","No manual page yet",PCList,asa->pctype_smooth,type,20,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(asa->pctype_smooth);CHKERRQ(ierr);
    ierr = PetscStrallocpy(type,&(asa->pctype_smooth));CHKERRQ(ierr);
  }
  ierr = PetscOptionsList("-pc_asa_direct_ksp_type","The type of KSP to be used in the direct solver","No manual page yet",KSPList,asa->ksptype_direct,type,20,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(asa->ksptype_direct);CHKERRQ(ierr);
    ierr = PetscStrallocpy(type,&(asa->ksptype_direct));CHKERRQ(ierr);
  }
  ierr = PetscOptionsList("-pc_asa_direct_pc_type","The type of PC to be used in the direct solver","No manual page yet",PCList,asa->pctype_direct,type,20,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(asa->pctype_direct);CHKERRQ(ierr);
    ierr = PetscStrallocpy(type,&(asa->pctype_direct));CHKERRQ(ierr);
  }
  /* options specific for certain smoothers */
  ierr = PetscOptionsReal("-pc_asa_richardson_scale","Scaling parameter for preconditioning in relaxation, if smoothing KSP is Richardson","No manual page yet",asa->richardson_scale,&(asa->richardson_scale),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_asa_sor_omega","Scaling parameter for preconditioning in relaxation, if smoothing KSP is Richardson","No manual page yet",asa->sor_omega,&(asa->sor_omega),&flg);CHKERRQ(ierr);
  /* options for direct solver */
  ierr = PetscOptionsString("-pc_asa_coarse_mat_type","The coarse level matrix type (e.g. SuperLU, MUMPS, ...)","No manual page yet",asa->coarse_mat_type, type,20,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(asa->coarse_mat_type);CHKERRQ(ierr);
    ierr = PetscStrallocpy(type,&(asa->coarse_mat_type));CHKERRQ(ierr);
  }
  /* storage allocation parameters */
  ierr = PetscOptionsInt("-pc_asa_max_cand_vecs","Maximum number of candidate vectors","No manual page yet",asa->max_cand_vecs,&(asa->max_cand_vecs),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_asa_max_dof_lev_2","The maximum number of degrees of freedom per node on level 2 (K in paper)","No manual page yet",asa->max_dof_lev_2,&(asa->max_dof_lev_2),&flg);CHKERRQ(ierr);
  /* construction parameters */
  ierr = PetscOptionsReal("-pc_asa_rq_improve","Threshold in RQ improvement for adding another candidate","No manual page yet",asa->rq_improve,&(asa->rq_improve),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_ASA"
static PetscErrorCode PCView_ASA(PC pc,PetscViewer viewer)
{
  PC_ASA          *asa = (PC_ASA*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;
  PC_ASA_level   *asa_lev = asa->levellist;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  ASA:\n");CHKERRQ(ierr);
    asa_lev = asa->levellist;
    while (asa_lev) {
      if (!asa_lev->next) {
        ierr = PetscViewerASCIIPrintf(viewer,"Coarse grid solver -- level %D -------------------------------\n",0);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Down solver (pre-smoother) on level ? -------------------------------\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(asa_lev->smoothd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      if (asa_lev->next && asa_lev->smoothd == asa_lev->smoothu) {
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) same as down solver (pre-smoother)\n");CHKERRQ(ierr);
      } else if (asa_lev->next){
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) on level ? -------------------------------\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(asa_lev->smoothu,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
      asa_lev = asa_lev->next;
    }
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCASA",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_ASA - Creates a ASA preconditioner context, PC_ASA,
   and sets this as the private data within the generic preconditioning
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_ASA"
PetscErrorCode  PCCreate_ASA(PC pc)
{
  PetscErrorCode ierr;
  PC_ASA         *asa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_ASA;
  /*  pc->ops->applytranspose      = PCApply_ASA;*/
  pc->ops->applyrichardson     = PCApplyRichardson_ASA;
  pc->ops->setup               = 0;
  pc->ops->destroy             = PCDestroy_ASA;
  pc->ops->setfromoptions      = PCSetFromOptions_ASA;
  pc->ops->view                = PCView_ASA;

  /* Set the data to pointer to 0 */
  pc->data                = (void*)0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASASetTolerances_C","PCASASetTolerances_ASA",PCASASetTolerances_ASA);CHKERRQ(ierr);

  /* register events */
  if (! asa_events_registered) {
    ierr = PetscLogEventRegister("PCInitializationStage_ASA", PC_CLASSID,&PC_InitializationStage_ASA);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("PCGeneralSetupStage_ASA",   PC_CLASSID,&PC_GeneralSetupStage_ASA);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("PCCreateTransferOp_ASA",    PC_CLASSID,&PC_CreateTransferOp_ASA);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("PCCreateVcycle_ASA",        PC_CLASSID,&PC_CreateVcycle_ASA);CHKERRQ(ierr);
    asa_events_registered = PETSC_TRUE;
  }

  /* Create new PC_ASA object */
  ierr = PetscNewLog(pc,PC_ASA,&asa);CHKERRQ(ierr);
  pc->data = (void*)asa;

  /* WORK: find some better initial values  */
  asa->nu             = 3;
  asa->gamma          = 1;
  asa->epsilon        = 1e-4;
  asa->mu             = 3;
  asa->mu_initial     = 20;
  asa->direct_solver  = 100;
  asa->scale_diag     = PETSC_TRUE;
  ierr = PetscStrallocpy(KSPRICHARDSON, (char **) &(asa->ksptype_smooth));CHKERRQ(ierr);
  ierr = PetscStrallocpy(PCSOR, (char **) &(asa->pctype_smooth));CHKERRQ(ierr);
  asa->smoother_rtol    = 1e-10;
  asa->smoother_abstol  = 1e-20;
  asa->smoother_dtol    = PETSC_DEFAULT;
  ierr = PetscStrallocpy(KSPPREONLY, (char **) &(asa->ksptype_direct));CHKERRQ(ierr);
  ierr = PetscStrallocpy(PCREDUNDANT, (char **) &(asa->pctype_direct));CHKERRQ(ierr);
  asa->direct_rtol      = 1e-10;
  asa->direct_abstol    = 1e-20;
  asa->direct_dtol      = PETSC_DEFAULT;
  asa->richardson_scale = PETSC_DECIDE;
  asa->sor_omega        = PETSC_DECIDE;
  ierr = PetscStrallocpy(MATSAME, (char **) &(asa->coarse_mat_type));CHKERRQ(ierr);

  asa->max_cand_vecs    = 4;
  asa->max_dof_lev_2    = 640; /* I don't think this parameter really matters, 640 should be enough for everyone! */

  asa->multigrid_constructed = PETSC_FALSE;

  asa->rtol       = 1e-10;
  asa->abstol     = 1e-15;
  asa->divtol     = 1e5;
  asa->max_it     = 10000;
  asa->rq_improve = 0.9;

  asa->A           = 0;
  asa->invsqrtdiag = 0;
  asa->b           = 0;
  asa->x           = 0;
  asa->r           = 0;

  asa->levels    = 0;
  asa->levellist = 0;

  asa->comm = ((PetscObject)pc)->comm;
  PetscFunctionReturn(0);
}
EXTERN_C_END

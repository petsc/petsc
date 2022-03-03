#include <petsc/private/kspimpl.h> /*I <petscksp.h> I*/
#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <petscdm.h>

static PetscBool  cited  = PETSC_FALSE;
static PetscBool  cited2 = PETSC_FALSE;
static const char citation[] =
"@article{ZampiniPCBDDC,\n"
"author = {Stefano Zampini},\n"
"title = {{PCBDDC}: A Class of Robust Dual-Primal Methods in {PETS}c},\n"
"journal = {SIAM Journal on Scientific Computing},\n"
"volume = {38},\n"
"number = {5},\n"
"pages = {S282-S306},\n"
"year = {2016},\n"
"doi = {10.1137/15M1025785},\n"
"URL = {http://dx.doi.org/10.1137/15M1025785},\n"
"eprint = {http://dx.doi.org/10.1137/15M1025785}\n"
"}\n"
"@article{ZampiniDualPrimal,\n"
"author = {Stefano Zampini},\n"
"title = {{D}ual-{P}rimal methods for the cardiac {B}idomain model},\n"
"volume = {24},\n"
"number = {04},\n"
"pages = {667-696},\n"
"year = {2014},\n"
"doi = {10.1142/S0218202513500632},\n"
"URL = {https://www.worldscientific.com/doi/abs/10.1142/S0218202513500632},\n"
"eprint = {https://www.worldscientific.com/doi/pdf/10.1142/S0218202513500632}\n"
"}\n";
static const char citation2[] =
"@article{li2013nonoverlapping,\n"
"title={A nonoverlapping domain decomposition method for incompressible Stokes equations with continuous pressures},\n"
"author={Li, Jing and Tu, Xuemin},\n"
"journal={SIAM Journal on Numerical Analysis},\n"
"volume={51},\n"
"number={2},\n"
"pages={1235--1253},\n"
"year={2013},\n"
"publisher={Society for Industrial and Applied Mathematics}\n"
"}\n";

/*
    This file implements the FETI-DP method in PETSc as part of KSP.
*/
typedef struct {
  KSP parentksp;
} KSP_FETIDPMon;

typedef struct {
  KSP              innerksp;         /* the KSP for the Lagrange multipliers */
  PC               innerbddc;        /* the inner BDDC object */
  PetscBool        fully_redundant;  /* true for using a fully redundant set of multipliers */
  PetscBool        userbddc;         /* true if the user provided the PCBDDC object */
  PetscBool        saddlepoint;      /* support for saddle point problems */
  IS               pP;               /* index set for pressure variables */
  Vec              rhs_flip;         /* see KSPFETIDPSetUpOperators */
  KSP_FETIDPMon    *monctx;          /* monitor context, used to pass user defined monitors
                                        in the physical space */
  PetscObjectState matstate;         /* these are needed just in the saddle point case */
  PetscObjectState matnnzstate;      /* where we are going to use MatZeroRows on pmat */
  PetscBool        statechanged;
  PetscBool        check;
} KSP_FETIDP;

static PetscErrorCode KSPFETIDPSetPressureOperator_FETIDP(KSP ksp, Mat P)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  if (P) fetidp->saddlepoint = PETSC_TRUE;
  CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_PPmat",(PetscObject)P));
  PetscFunctionReturn(0);
}

/*@
 KSPFETIDPSetPressureOperator - Sets the operator used to setup the pressure preconditioner for saddle point FETI-DP.

   Collective on ksp

   Input Parameters:
+  ksp - the FETI-DP Krylov solver
-  P - the linear operator to be preconditioned, usually the mass matrix.

   Level: advanced

   Notes:
    The operator can be either passed in a) monolithic global ordering, b) pressure-only global ordering
          or c) interface pressure ordering (if -ksp_fetidp_pressure_all false).
          In cases b) and c), the pressure ordering of dofs needs to satisfy
             pid_1 < pid_2  iff  gid_1 < gid_2
          where pid_1 and pid_2 are two different pressure dof numbers and gid_1 and gid_2 the corresponding
          id in the monolithic global ordering.

.seealso: MATIS, PCBDDC, KSPFETIDPGetInnerBDDC, KSPFETIDPGetInnerKSP, KSPSetOperators
@*/
PetscErrorCode KSPFETIDPSetPressureOperator(KSP ksp, Mat P)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (P) PetscValidHeaderSpecific(P,MAT_CLASSID,2);
  CHKERRQ(PetscTryMethod(ksp,"KSPFETIDPSetPressureOperator_C",(KSP,Mat),(ksp,P)));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPGetInnerKSP_FETIDP(KSP ksp, KSP* innerksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  *innerksp = fetidp->innerksp;
  PetscFunctionReturn(0);
}

/*@
 KSPFETIDPGetInnerKSP - Gets the KSP object for the Lagrange multipliers

   Input Parameters:
+  ksp - the FETI-DP KSP
-  innerksp - the KSP for the multipliers

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC, KSPFETIDPGetInnerBDDC
@*/
PetscErrorCode KSPFETIDPGetInnerKSP(KSP ksp, KSP* innerksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(innerksp,2);
  CHKERRQ(PetscUseMethod(ksp,"KSPFETIDPGetInnerKSP_C",(KSP,KSP*),(ksp,innerksp)));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPGetInnerBDDC_FETIDP(KSP ksp, PC* pc)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  *pc = fetidp->innerbddc;
  PetscFunctionReturn(0);
}

/*@
 KSPFETIDPGetInnerBDDC - Gets the BDDC preconditioner used to setup the FETI-DP matrix for the Lagrange multipliers

   Input Parameters:
+  ksp - the FETI-DP Krylov solver
-  pc - the BDDC preconditioner

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC, KSPFETIDPGetInnerKSP
@*/
PetscErrorCode KSPFETIDPGetInnerBDDC(KSP ksp, PC* pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(pc,2);
  CHKERRQ(PetscUseMethod(ksp,"KSPFETIDPGetInnerBDDC_C",(KSP,PC*),(ksp,pc)));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPSetInnerBDDC_FETIDP(KSP ksp, PC pc)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)pc));
  CHKERRQ(PCDestroy(&fetidp->innerbddc));
  fetidp->innerbddc = pc;
  fetidp->userbddc  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
 KSPFETIDPSetInnerBDDC - Sets the BDDC preconditioner used to setup the FETI-DP matrix for the Lagrange multipliers

   Collective on ksp

   Input Parameters:
+  ksp - the FETI-DP Krylov solver
-  pc - the BDDC preconditioner

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPGetInnerBDDC, KSPFETIDPGetInnerKSP
@*/
PetscErrorCode KSPFETIDPSetInnerBDDC(KSP ksp, PC pc)
{
  PetscBool      isbddc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCBDDC,&isbddc));
  PetscCheck(isbddc,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"KSPFETIDPSetInnerBDDC need a PCBDDC preconditioner");
  CHKERRQ(PetscTryMethod(ksp,"KSPFETIDPSetInnerBDDC_C",(KSP,PC),(ksp,pc)));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPBuildSolution_FETIDP(KSP ksp,Vec v,Vec *V)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  Mat            F;
  Vec            Xl;

  PetscFunctionBegin;
  CHKERRQ(KSPGetOperators(fetidp->innerksp,&F,NULL));
  CHKERRQ(KSPBuildSolution(fetidp->innerksp,NULL,&Xl));
  if (v) {
    CHKERRQ(PCBDDCMatFETIDPGetSolution(F,Xl,v));
    *V   = v;
  } else {
    CHKERRQ(PCBDDCMatFETIDPGetSolution(F,Xl,*V));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMonitor_FETIDP(KSP ksp,PetscInt it,PetscReal rnorm,void* ctx)
{
  KSP_FETIDPMon  *monctx = (KSP_FETIDPMon*)ctx;

  PetscFunctionBegin;
  CHKERRQ(KSPMonitor(monctx->parentksp,it,rnorm));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPComputeEigenvalues_FETIDP(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  CHKERRQ(KSPComputeEigenvalues(fetidp->innerksp,nmax,r,c,neig));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPComputeExtremeSingularValues_FETIDP(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  CHKERRQ(KSPComputeExtremeSingularValues(fetidp->innerksp,emax,emin));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPCheckOperators(KSP ksp, PetscViewer viewer)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  PC_IS          *pcis = (PC_IS*)fetidp->innerbddc->data;
  Mat_IS         *matis = (Mat_IS*)fetidp->innerbddc->pmat->data;
  Mat            F;
  FETIDPMat_ctx  fetidpmat_ctx;
  Vec            test_vec,test_vec_p = NULL,fetidp_global;
  IS             dirdofs,isvert;
  MPI_Comm       comm = PetscObjectComm((PetscObject)ksp);
  PetscScalar    sval,*array;
  PetscReal      val,rval;
  const PetscInt *vertex_indices;
  PetscInt       i,n_vertices;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCheckSameComm(ksp,1,viewer,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  PetscCheck(isascii,comm,PETSC_ERR_SUP,"Unsupported viewer");
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"----------FETI-DP MAT  --------------\n"));
  CHKERRQ(PetscViewerASCIIAddTab(viewer,2));
  CHKERRQ(KSPGetOperators(fetidp->innerksp,&F,NULL));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
  CHKERRQ(MatView(F,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerASCIISubtractTab(viewer,2));
  CHKERRQ(MatShellGetContext(F,&fetidpmat_ctx));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"----------FETI-DP TESTS--------------\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"All tests should return zero!\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"FETIDP MAT context in the "));
  if (fetidp->fully_redundant) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"fully redundant case for lagrange multipliers.\n"));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Non-fully redundant case for lagrange multiplier.\n"));
  }
  CHKERRQ(PetscViewerFlush(viewer));

  /* Get Vertices used to define the BDDC */
  CHKERRQ(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert));
  CHKERRQ(ISGetLocalSize(isvert,&n_vertices));
  CHKERRQ(ISGetIndices(isvert,&vertex_indices));

  /******************************************************************/
  /* TEST A/B: Test numbering of global fetidp dofs                 */
  /******************************************************************/
  CHKERRQ(MatCreateVecs(F,&fetidp_global,NULL));
  CHKERRQ(VecDuplicate(fetidpmat_ctx->lambda_local,&test_vec));
  CHKERRQ(VecSet(fetidp_global,1.0));
  CHKERRQ(VecSet(test_vec,1.));
  CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  if (fetidpmat_ctx->l2g_p) {
    CHKERRQ(VecDuplicate(fetidpmat_ctx->vP,&test_vec_p));
    CHKERRQ(VecSet(test_vec_p,1.));
    CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_p,fetidp_global,fetidpmat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_p,fetidp_global,fetidpmat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
  }
  CHKERRQ(VecAXPY(test_vec,-1.0,fetidpmat_ctx->lambda_local));
  CHKERRQ(VecNorm(test_vec,NORM_INFINITY,&val));
  CHKERRQ(VecDestroy(&test_vec));
  CHKERRMPI(MPI_Reduce(&val,&rval,1,MPIU_REAL,MPIU_MAX,0,comm));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"A: CHECK glob to loc: % 1.14e\n",rval));

  if (fetidpmat_ctx->l2g_p) {
    CHKERRQ(VecAXPY(test_vec_p,-1.0,fetidpmat_ctx->vP));
    CHKERRQ(VecNorm(test_vec_p,NORM_INFINITY,&val));
    CHKERRMPI(MPI_Reduce(&val,&rval,1,MPIU_REAL,MPIU_MAX,0,comm));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"A: CHECK glob to loc (p): % 1.14e\n",rval));
  }

  if (fetidp->fully_redundant) {
    CHKERRQ(VecSet(fetidp_global,0.0));
    CHKERRQ(VecSet(fetidpmat_ctx->lambda_local,0.5));
    CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecSum(fetidp_global,&sval));
    val  = PetscRealPart(sval)-fetidpmat_ctx->n_lambda;
    CHKERRMPI(MPI_Reduce(&val,&rval,1,MPIU_REAL,MPIU_MAX,0,comm));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"B: CHECK loc to glob: % 1.14e\n",rval));
  }

  if (fetidpmat_ctx->l2g_p) {
    CHKERRQ(VecSet(pcis->vec1_N,1.0));
    CHKERRQ(VecSet(pcis->vec1_global,0.0));
    CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));

    CHKERRQ(VecSet(fetidp_global,0.0));
    CHKERRQ(VecSet(fetidpmat_ctx->vP,-1.0));
    CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterBegin(fetidpmat_ctx->g2g_p,fetidp_global,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(fetidpmat_ctx->g2g_p,fetidp_global,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterBegin(fetidpmat_ctx->g2g_p,pcis->vec1_global,fetidp_global,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(fetidpmat_ctx->g2g_p,pcis->vec1_global,fetidp_global,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecSum(fetidp_global,&sval));
    val  = PetscRealPart(sval);
    CHKERRMPI(MPI_Reduce(&val,&rval,1,MPIU_REAL,MPIU_MAX,0,comm));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"B: CHECK loc to glob (p): % 1.14e\n",rval));
  }

  /******************************************************************/
  /* TEST C: It should hold B_delta*w=0, w\in\widehat{W}            */
  /* This is the meaning of the B matrix                            */
  /******************************************************************/

  CHKERRQ(VecSetRandom(pcis->vec1_N,NULL));
  CHKERRQ(VecSet(pcis->vec1_global,0.0));
  CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  /* Action of B_delta */
  CHKERRQ(MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local));
  CHKERRQ(VecSet(fetidp_global,0.0));
  CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecNorm(fetidp_global,NORM_INFINITY,&val));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"C: CHECK infty norm of B_delta*w (w continuous): % 1.14e\n",val));

  /******************************************************************/
  /* TEST D: It should hold E_Dw = w - P_Dw w\in\widetilde{W}       */
  /* E_D = R_D^TR                                                   */
  /* P_D = B_{D,delta}^T B_{delta}                                  */
  /* eq.44 Mandel Tezaur and Dohrmann 2005                          */
  /******************************************************************/

  /* compute a random vector in \widetilde{W} */
  CHKERRQ(VecSetRandom(pcis->vec1_N,NULL));
  /* set zero at vertices and essential dofs */
  CHKERRQ(VecGetArray(pcis->vec1_N,&array));
  for (i=0;i<n_vertices;i++) array[vertex_indices[i]] = 0.0;
  CHKERRQ(PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph,&dirdofs));
  if (dirdofs) {
    const PetscInt *idxs;
    PetscInt       ndir;

    CHKERRQ(ISGetLocalSize(dirdofs,&ndir));
    CHKERRQ(ISGetIndices(dirdofs,&idxs));
    for (i=0;i<ndir;i++) array[idxs[i]] = 0.0;
    CHKERRQ(ISRestoreIndices(dirdofs,&idxs));
  }
  CHKERRQ(VecRestoreArray(pcis->vec1_N,&array));
  /* store w for final comparison */
  CHKERRQ(VecDuplicate(pcis->vec1_B,&test_vec));
  CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,test_vec,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,test_vec,INSERT_VALUES,SCATTER_FORWARD));

  /* Jump operator P_D : results stored in pcis->vec1_B */
  /* Action of B_delta */
  CHKERRQ(MatMult(fetidpmat_ctx->B_delta,test_vec,fetidpmat_ctx->lambda_local));
  CHKERRQ(VecSet(fetidp_global,0.0));
  CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
  /* Action of B_Ddelta^T */
  CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B));

  /* Average operator E_D : results stored in pcis->vec2_B */
  CHKERRQ(PCBDDCScalingExtension(fetidpmat_ctx->pc,test_vec,pcis->vec1_global));
  CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD));

  /* test E_D=I-P_D */
  CHKERRQ(VecAXPY(pcis->vec1_B,1.0,pcis->vec2_B));
  CHKERRQ(VecAXPY(pcis->vec1_B,-1.0,test_vec));
  CHKERRQ(VecNorm(pcis->vec1_B,NORM_INFINITY,&val));
  CHKERRQ(VecDestroy(&test_vec));
  CHKERRMPI(MPI_Reduce(&val,&rval,1,MPIU_REAL,MPIU_MAX,0,comm));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"D: CHECK infty norm of E_D + P_D - I: % 1.14e\n",PetscGlobalRank,val));

  /******************************************************************/
  /* TEST E: It should hold R_D^TP_Dw=0 w\in\widetilde{W}           */
  /* eq.48 Mandel Tezaur and Dohrmann 2005                          */
  /******************************************************************/

  CHKERRQ(VecSetRandom(pcis->vec1_N,NULL));
  /* set zero at vertices and essential dofs */
  CHKERRQ(VecGetArray(pcis->vec1_N,&array));
  for (i=0;i<n_vertices;i++) array[vertex_indices[i]] = 0.0;
  if (dirdofs) {
    const PetscInt *idxs;
    PetscInt       ndir;

    CHKERRQ(ISGetLocalSize(dirdofs,&ndir));
    CHKERRQ(ISGetIndices(dirdofs,&idxs));
    for (i=0;i<ndir;i++) array[idxs[i]] = 0.0;
    CHKERRQ(ISRestoreIndices(dirdofs,&idxs));
  }
  CHKERRQ(VecRestoreArray(pcis->vec1_N,&array));

  /* Jump operator P_D : results stored in pcis->vec1_B */

  CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  /* Action of B_delta */
  CHKERRQ(MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local));
  CHKERRQ(VecSet(fetidp_global,0.0));
  CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD));
  /* Action of B_Ddelta^T */
  CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B));
  /* scaling */
  CHKERRQ(PCBDDCScalingExtension(fetidpmat_ctx->pc,pcis->vec1_B,pcis->vec1_global));
  CHKERRQ(VecNorm(pcis->vec1_global,NORM_INFINITY,&val));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"E: CHECK infty norm of R^T_D P_D: % 1.14e\n",val));

  if (!fetidp->fully_redundant) {
    /******************************************************************/
    /* TEST F: It should holds B_{delta}B^T_{D,delta}=I               */
    /* Corollary thm 14 Mandel Tezaur and Dohrmann 2005               */
    /******************************************************************/
    CHKERRQ(VecDuplicate(fetidp_global,&test_vec));
    CHKERRQ(VecSetRandom(fetidp_global,NULL));
    if (fetidpmat_ctx->l2g_p) {
      CHKERRQ(VecSet(fetidpmat_ctx->vP,0.));
      CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,INSERT_VALUES,SCATTER_FORWARD));
    }
    /* Action of B_Ddelta^T */
    CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B));
    /* Action of B_delta */
    CHKERRQ(MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local));
    CHKERRQ(VecSet(test_vec,0.0));
    CHKERRQ(VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,test_vec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,test_vec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecAXPY(fetidp_global,-1.,test_vec));
    CHKERRQ(VecNorm(fetidp_global,NORM_INFINITY,&val));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"E: CHECK infty norm of P^T_D - I: % 1.14e\n",val));
    CHKERRQ(VecDestroy(&test_vec));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"-------------------------------------\n"));
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(VecDestroy(&test_vec_p));
  CHKERRQ(ISDestroy(&dirdofs));
  CHKERRQ(VecDestroy(&fetidp_global));
  CHKERRQ(ISRestoreIndices(isvert,&vertex_indices));
  CHKERRQ(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPSetUpOperators(KSP ksp)
{
  KSP_FETIDP       *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC          *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  Mat              A,Ap;
  PetscInt         fid = -1;
  PetscMPIInt      size;
  PetscBool        ismatis,pisz,allp,schp;
  PetscBool        flip; /* Usually, Stokes is written (B = -\int_\Omega \nabla \cdot u q)
                           | A B'| | v | = | f |
                           | B 0 | | p | = | g |
                            If -ksp_fetidp_saddlepoint_flip is true, the code assumes it is written as
                           | A B'| | v | = | f |
                           |-B 0 | | p | = |-g |
                         */
  PetscObjectState matstate, matnnzstate;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  pisz = PETSC_FALSE;
  flip = PETSC_FALSE;
  allp = PETSC_FALSE;
  schp = PETSC_FALSE;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"FETI-DP options","PC");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-ksp_fetidp_pressure_field","Field id for pressures for saddle-point problems",NULL,fid,&fid,NULL));
  CHKERRQ(PetscOptionsBool("-ksp_fetidp_pressure_all","Use the whole pressure set instead of just that at the interface",NULL,allp,&allp,NULL));
  CHKERRQ(PetscOptionsBool("-ksp_fetidp_saddlepoint_flip","Flip the sign of the pressure-velocity (lower-left) block",NULL,flip,&flip,NULL));
  CHKERRQ(PetscOptionsBool("-ksp_fetidp_pressure_schur","Use a BDDC solver for pressure",NULL,schp,&schp,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ksp),&size));
  fetidp->saddlepoint = (fid >= 0 ? PETSC_TRUE : fetidp->saddlepoint);
  if (size == 1) fetidp->saddlepoint = PETSC_FALSE;

  CHKERRQ(KSPGetOperators(ksp,&A,&Ap));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis));
  PetscCheck(ismatis,PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Amat should be of type MATIS");

  /* Quiet return if the matrix states are unchanged.
     Needed only for the saddle point case since it uses MatZeroRows
     on a matrix that may not have changed */
  CHKERRQ(PetscObjectStateGet((PetscObject)A,&matstate));
  CHKERRQ(MatGetNonzeroState(A,&matnnzstate));
  if (matstate == fetidp->matstate && matnnzstate == fetidp->matnnzstate) PetscFunctionReturn(0);
  fetidp->matstate     = matstate;
  fetidp->matnnzstate  = matnnzstate;
  fetidp->statechanged = fetidp->saddlepoint;

  /* see if we have some fields attached */
  if (!pcbddc->n_ISForDofsLocal && !pcbddc->n_ISForDofs) {
    DM             dm;
    PetscContainer c;

    CHKERRQ(KSPGetDM(ksp,&dm));
    CHKERRQ(PetscObjectQuery((PetscObject)A,"_convert_nest_lfields",(PetscObject*)&c));
    if (dm) {
      IS      *fields;
      PetscInt nf,i;

      CHKERRQ(DMCreateFieldDecomposition(dm,&nf,NULL,&fields,NULL));
      CHKERRQ(PCBDDCSetDofsSplitting(fetidp->innerbddc,nf,fields));
      for (i=0;i<nf;i++) {
        CHKERRQ(ISDestroy(&fields[i]));
      }
      CHKERRQ(PetscFree(fields));
    } else if (c) {
      MatISLocalFields lf;

      CHKERRQ(PetscContainerGetPointer(c,(void**)&lf));
      CHKERRQ(PCBDDCSetDofsSplittingLocal(fetidp->innerbddc,lf->nr,lf->rf));
    }
  }

  if (!fetidp->saddlepoint) {
    CHKERRQ(PCSetOperators(fetidp->innerbddc,A,A));
  } else {
    Mat          nA,lA,PPmat;
    MatNullSpace nnsp;
    IS           pP;
    PetscInt     totP;

    CHKERRQ(MatISGetLocalMat(A,&lA));
    CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lA",(PetscObject)lA));

    pP = fetidp->pP;
    if (!pP) { /* first time, need to compute pressure dofs */
      PC_IS                  *pcis = (PC_IS*)fetidp->innerbddc->data;
      Mat_IS                 *matis = (Mat_IS*)(A->data);
      ISLocalToGlobalMapping l2g;
      IS                     lP = NULL,II,pII,lPall,Pall,is1,is2;
      const PetscInt         *idxs;
      PetscInt               nl,ni,*widxs;
      PetscInt               i,j,n_neigh,*neigh,*n_shared,**shared,*count;
      PetscInt               rst,ren,n;
      PetscBool              ploc;

      CHKERRQ(MatGetLocalSize(A,&nl,NULL));
      CHKERRQ(MatGetOwnershipRange(A,&rst,&ren));
      CHKERRQ(MatGetLocalSize(lA,&n,NULL));
      CHKERRQ(MatISGetLocalToGlobalMapping(A,&l2g,NULL));

      if (!pcis->is_I_local) { /* need to compute interior dofs */
        CHKERRQ(PetscCalloc1(n,&count));
        CHKERRQ(ISLocalToGlobalMappingGetInfo(l2g,&n_neigh,&neigh,&n_shared,&shared));
        for (i=1;i<n_neigh;i++)
          for (j=0;j<n_shared[i];j++)
            count[shared[i][j]] += 1;
        for (i=0,j=0;i<n;i++) if (!count[i]) count[j++] = i;
        CHKERRQ(ISLocalToGlobalMappingRestoreInfo(l2g,&n_neigh,&neigh,&n_shared,&shared));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,j,count,PETSC_OWN_POINTER,&II));
      } else {
        CHKERRQ(PetscObjectReference((PetscObject)pcis->is_I_local));
        II   = pcis->is_I_local;
      }

      /* interior dofs in layout */
      CHKERRQ(PetscArrayzero(matis->sf_leafdata,n));
      CHKERRQ(PetscArrayzero(matis->sf_rootdata,nl));
      CHKERRQ(ISGetLocalSize(II,&ni));
      CHKERRQ(ISGetIndices(II,&idxs));
      for (i=0;i<ni;i++) matis->sf_leafdata[idxs[i]] = 1;
      CHKERRQ(ISRestoreIndices(II,&idxs));
      CHKERRQ(PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE));
      CHKERRQ(PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE));
      CHKERRQ(PetscMalloc1(PetscMax(nl,n),&widxs));
      for (i=0,ni=0;i<nl;i++) if (matis->sf_rootdata[i]) widxs[ni++] = i+rst;
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&pII));

      /* pressure dofs */
      Pall  = NULL;
      lPall = NULL;
      ploc  = PETSC_FALSE;
      if (fid < 0) { /* zero pressure block */
        PetscInt np;

        CHKERRQ(MatFindZeroDiagonals(A,&Pall));
        CHKERRQ(ISGetSize(Pall,&np));
        if (!np) { /* zero-block not found, defaults to last field (if set) */
          fid  = pcbddc->n_ISForDofsLocal ? pcbddc->n_ISForDofsLocal - 1 : pcbddc->n_ISForDofs - 1;
          CHKERRQ(ISDestroy(&Pall));
        } else if (!pcbddc->n_ISForDofsLocal && !pcbddc->n_ISForDofs) {
          CHKERRQ(PCBDDCSetDofsSplitting(fetidp->innerbddc,1,&Pall));
        }
      }
      if (!Pall) { /* look for registered fields */
        if (pcbddc->n_ISForDofsLocal) {
          PetscInt np;

          PetscCheckFalse(fid < 0 || fid >= pcbddc->n_ISForDofsLocal,PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Invalid field id for pressure %D, max %D",fid,pcbddc->n_ISForDofsLocal);
          /* need a sequential IS */
          CHKERRQ(ISGetLocalSize(pcbddc->ISForDofsLocal[fid],&np));
          CHKERRQ(ISGetIndices(pcbddc->ISForDofsLocal[fid],&idxs));
          CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,np,idxs,PETSC_COPY_VALUES,&lPall));
          CHKERRQ(ISRestoreIndices(pcbddc->ISForDofsLocal[fid],&idxs));
          ploc = PETSC_TRUE;
        } else if (pcbddc->n_ISForDofs) {
          PetscCheckFalse(fid < 0 || fid >= pcbddc->n_ISForDofs,PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Invalid field id for pressure %D, max %D",fid,pcbddc->n_ISForDofs);
          CHKERRQ(PetscObjectReference((PetscObject)pcbddc->ISForDofs[fid]));
          Pall = pcbddc->ISForDofs[fid];
        } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Cannot detect pressure field! Use KSPFETIDPGetInnerBDDC() + PCBDDCSetDofsSplitting or PCBDDCSetDofsSplittingLocal");
      }

      /* if the user requested the entire pressure,
         remove the interior pressure dofs from II (or pII) */
      if (allp) {
        if (ploc) {
          IS nII;
          CHKERRQ(ISDifference(II,lPall,&nII));
          CHKERRQ(ISDestroy(&II));
          II   = nII;
        } else {
          IS nII;
          CHKERRQ(ISDifference(pII,Pall,&nII));
          CHKERRQ(ISDestroy(&pII));
          pII  = nII;
        }
      }
      if (ploc) {
        CHKERRQ(ISDifference(lPall,II,&lP));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lP",(PetscObject)lP));
      } else {
        CHKERRQ(ISDifference(Pall,pII,&pP));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP",(PetscObject)pP));
        /* need all local pressure dofs */
        CHKERRQ(PetscArrayzero(matis->sf_leafdata,n));
        CHKERRQ(PetscArrayzero(matis->sf_rootdata,nl));
        CHKERRQ(ISGetLocalSize(Pall,&ni));
        CHKERRQ(ISGetIndices(Pall,&idxs));
        for (i=0;i<ni;i++) matis->sf_rootdata[idxs[i]-rst] = 1;
        CHKERRQ(ISRestoreIndices(Pall,&idxs));
        CHKERRQ(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
        CHKERRQ(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
        for (i=0,ni=0;i<n;i++) if (matis->sf_leafdata[i]) widxs[ni++] = i;
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,ni,widxs,PETSC_COPY_VALUES,&lPall));
      }

      if (!Pall) {
        CHKERRQ(PetscArrayzero(matis->sf_leafdata,n));
        CHKERRQ(PetscArrayzero(matis->sf_rootdata,nl));
        CHKERRQ(ISGetLocalSize(lPall,&ni));
        CHKERRQ(ISGetIndices(lPall,&idxs));
        for (i=0;i<ni;i++) matis->sf_leafdata[idxs[i]] = 1;
        CHKERRQ(ISRestoreIndices(lPall,&idxs));
        CHKERRQ(PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE));
        CHKERRQ(PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE));
        for (i=0,ni=0;i<nl;i++) if (matis->sf_rootdata[i]) widxs[ni++] = i+rst;
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&Pall));
      }
      CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_aP",(PetscObject)Pall));

      if (flip) {
        PetscInt npl;
        CHKERRQ(ISGetLocalSize(Pall,&npl));
        CHKERRQ(ISGetIndices(Pall,&idxs));
        CHKERRQ(MatCreateVecs(A,NULL,&fetidp->rhs_flip));
        CHKERRQ(VecSet(fetidp->rhs_flip,1.));
        CHKERRQ(VecSetOption(fetidp->rhs_flip,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));
        for (i=0;i<npl;i++) {
          CHKERRQ(VecSetValue(fetidp->rhs_flip,idxs[i],-1.,INSERT_VALUES));
        }
        CHKERRQ(VecAssemblyBegin(fetidp->rhs_flip));
        CHKERRQ(VecAssemblyEnd(fetidp->rhs_flip));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_flip",(PetscObject)fetidp->rhs_flip));
        CHKERRQ(ISRestoreIndices(Pall,&idxs));
      }
      CHKERRQ(ISDestroy(&Pall));
      CHKERRQ(ISDestroy(&pII));

      /* local selected pressures in subdomain-wise and global ordering */
      CHKERRQ(PetscArrayzero(matis->sf_leafdata,n));
      CHKERRQ(PetscArrayzero(matis->sf_rootdata,nl));
      if (!ploc) {
        PetscInt *widxs2;

        PetscCheck(pP,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Missing parallel pressure IS");
        CHKERRQ(ISGetLocalSize(pP,&ni));
        CHKERRQ(ISGetIndices(pP,&idxs));
        for (i=0;i<ni;i++) matis->sf_rootdata[idxs[i]-rst] = 1;
        CHKERRQ(ISRestoreIndices(pP,&idxs));
        CHKERRQ(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
        CHKERRQ(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
        for (i=0,ni=0;i<n;i++) if (matis->sf_leafdata[i]) widxs[ni++] = i;
        CHKERRQ(PetscMalloc1(ni,&widxs2));
        CHKERRQ(ISLocalToGlobalMappingApply(l2g,ni,widxs,widxs2));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,ni,widxs,PETSC_COPY_VALUES,&lP));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lP",(PetscObject)lP));
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs2,PETSC_OWN_POINTER,&is1));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_gP",(PetscObject)is1));
        CHKERRQ(ISDestroy(&is1));
      } else {
        PetscCheck(lP,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing sequential pressure IS");
        CHKERRQ(ISGetLocalSize(lP,&ni));
        CHKERRQ(ISGetIndices(lP,&idxs));
        for (i=0;i<ni;i++)
          if (idxs[i] >=0 && idxs[i] < n)
            matis->sf_leafdata[idxs[i]] = 1;
        CHKERRQ(ISRestoreIndices(lP,&idxs));
        CHKERRQ(PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE));
        CHKERRQ(ISLocalToGlobalMappingApply(l2g,ni,idxs,widxs));
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&is1));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_gP",(PetscObject)is1));
        CHKERRQ(ISDestroy(&is1));
        CHKERRQ(PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE));
        for (i=0,ni=0;i<nl;i++) if (matis->sf_rootdata[i]) widxs[ni++] = i+rst;
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&pP));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP",(PetscObject)pP));
      }
      CHKERRQ(PetscFree(widxs));

      /* If there's any "interior pressure",
         we may want to use a discrete harmonic solver instead
         of a Stokes harmonic for the Dirichlet preconditioner
         Need to extract the interior velocity dofs in interior dofs ordering (iV)
         and interior pressure dofs in local ordering (iP) */
      if (!allp) {
        ISLocalToGlobalMapping l2g_t;

        CHKERRQ(ISDifference(lPall,lP,&is1));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_iP",(PetscObject)is1));
        CHKERRQ(ISDifference(II,is1,&is2));
        CHKERRQ(ISDestroy(&is1));
        CHKERRQ(ISLocalToGlobalMappingCreateIS(II,&l2g_t));
        CHKERRQ(ISGlobalToLocalMappingApplyIS(l2g_t,IS_GTOLM_DROP,is2,&is1));
        CHKERRQ(ISGetLocalSize(is1,&i));
        CHKERRQ(ISGetLocalSize(is2,&j));
        PetscCheckFalse(i != j,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent local sizes %D and %D for iV",i,j);
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_iV",(PetscObject)is1));
        CHKERRQ(ISLocalToGlobalMappingDestroy(&l2g_t));
        CHKERRQ(ISDestroy(&is1));
        CHKERRQ(ISDestroy(&is2));
      }
      CHKERRQ(ISDestroy(&II));

      /* exclude selected pressures from the inner BDDC */
      if (pcbddc->DirichletBoundariesLocal) {
        IS       list[2],plP,isout;
        PetscInt np;

        /* need a parallel IS */
        CHKERRQ(ISGetLocalSize(lP,&np));
        CHKERRQ(ISGetIndices(lP,&idxs));
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ksp),np,idxs,PETSC_USE_POINTER,&plP));
        list[0] = plP;
        list[1] = pcbddc->DirichletBoundariesLocal;
        CHKERRQ(ISConcatenate(PetscObjectComm((PetscObject)ksp),2,list,&isout));
        CHKERRQ(ISSortRemoveDups(isout));
        CHKERRQ(ISDestroy(&plP));
        CHKERRQ(ISRestoreIndices(lP,&idxs));
        CHKERRQ(PCBDDCSetDirichletBoundariesLocal(fetidp->innerbddc,isout));
        CHKERRQ(ISDestroy(&isout));
      } else if (pcbddc->DirichletBoundaries) {
        IS list[2],isout;

        list[0] = pP;
        list[1] = pcbddc->DirichletBoundaries;
        CHKERRQ(ISConcatenate(PetscObjectComm((PetscObject)ksp),2,list,&isout));
        CHKERRQ(ISSortRemoveDups(isout));
        CHKERRQ(PCBDDCSetDirichletBoundaries(fetidp->innerbddc,isout));
        CHKERRQ(ISDestroy(&isout));
      } else {
        IS       plP;
        PetscInt np;

        /* need a parallel IS */
        CHKERRQ(ISGetLocalSize(lP,&np));
        CHKERRQ(ISGetIndices(lP,&idxs));
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ksp),np,idxs,PETSC_COPY_VALUES,&plP));
        CHKERRQ(PCBDDCSetDirichletBoundariesLocal(fetidp->innerbddc,plP));
        CHKERRQ(ISDestroy(&plP));
        CHKERRQ(ISRestoreIndices(lP,&idxs));
      }

      /* save CSR information for the pressure BDDC solver (if any) */
      if (schp) {
        PetscInt np,nt;

        CHKERRQ(MatGetSize(matis->A,&nt,NULL));
        CHKERRQ(ISGetLocalSize(lP,&np));
        if (np) {
          PetscInt *xadj = pcbddc->mat_graph->xadj;
          PetscInt *adjn = pcbddc->mat_graph->adjncy;
          PetscInt nv = pcbddc->mat_graph->nvtxs_csr;

          if (nv && nv == nt) {
            ISLocalToGlobalMapping pmap;
            PetscInt               *schp_csr,*schp_xadj,*schp_adjn,p;
            PetscContainer         c;

            CHKERRQ(ISLocalToGlobalMappingCreateIS(lPall,&pmap));
            CHKERRQ(ISGetIndices(lPall,&idxs));
            for (p = 0, nv = 0; p < np; p++) {
              PetscInt x,n = idxs[p];

              CHKERRQ(ISGlobalToLocalMappingApply(pmap,IS_GTOLM_DROP,xadj[n+1]-xadj[n],adjn+xadj[n],&x,NULL));
              nv  += x;
            }
            CHKERRQ(PetscMalloc1(np + 1 + nv,&schp_csr));
            schp_xadj = schp_csr;
            schp_adjn = schp_csr + np + 1;
            for (p = 0, schp_xadj[0] = 0; p < np; p++) {
              PetscInt x,n = idxs[p];

              CHKERRQ(ISGlobalToLocalMappingApply(pmap,IS_GTOLM_DROP,xadj[n+1]-xadj[n],adjn+xadj[n],&x,schp_adjn + schp_xadj[p]));
              schp_xadj[p+1] = schp_xadj[p] + x;
            }
            CHKERRQ(ISRestoreIndices(lPall,&idxs));
            CHKERRQ(ISLocalToGlobalMappingDestroy(&pmap));
            CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF,&c));
            CHKERRQ(PetscContainerSetPointer(c,schp_csr));
            CHKERRQ(PetscContainerSetUserDestroy(c,PetscContainerUserDestroyDefault));
            CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pCSR",(PetscObject)c));
            CHKERRQ(PetscContainerDestroy(&c));

          }
        }
      }
      CHKERRQ(ISDestroy(&lPall));
      CHKERRQ(ISDestroy(&lP));
      fetidp->pP = pP;
    }

    /* total number of selected pressure dofs */
    CHKERRQ(ISGetSize(fetidp->pP,&totP));

    /* Set operator for inner BDDC */
    if (totP || fetidp->rhs_flip) {
      CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&nA));
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)A));
      nA   = A;
    }
    if (fetidp->rhs_flip) {
      CHKERRQ(MatDiagonalScale(nA,fetidp->rhs_flip,NULL));
      if (totP) {
        Mat lA2;

        CHKERRQ(MatISGetLocalMat(nA,&lA));
        CHKERRQ(MatDuplicate(lA,MAT_COPY_VALUES,&lA2));
        CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lA",(PetscObject)lA2));
        CHKERRQ(MatDestroy(&lA2));
      }
    }

    if (totP) {
      CHKERRQ(MatSetOption(nA,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
      CHKERRQ(MatZeroRowsColumnsIS(nA,fetidp->pP,1.,NULL,NULL));
    } else {
      CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lA",NULL));
    }
    CHKERRQ(MatGetNearNullSpace(Ap,&nnsp));
    if (!nnsp) {
      CHKERRQ(MatGetNullSpace(Ap,&nnsp));
    }
    if (!nnsp) {
      CHKERRQ(MatGetNearNullSpace(A,&nnsp));
    }
    if (!nnsp) {
      CHKERRQ(MatGetNullSpace(A,&nnsp));
    }
    CHKERRQ(MatSetNearNullSpace(nA,nnsp));
    CHKERRQ(PCSetOperators(fetidp->innerbddc,nA,nA));
    CHKERRQ(MatDestroy(&nA));

    /* non-zero rhs on interior dofs when applying the preconditioner */
    if (totP) pcbddc->switch_static = PETSC_TRUE;

    /* if there are no interface pressures, set inner bddc flag for benign saddle point */
    if (!totP) {
      pcbddc->benign_saddle_point = PETSC_TRUE;
      pcbddc->compute_nonetflux   = PETSC_TRUE;
    }

    /* Operators for pressure preconditioner */
    if (totP) {
      /* Extract pressure block if needed */
      if (!pisz) {
        Mat C;
        IS  nzrows = NULL;

        CHKERRQ(MatCreateSubMatrix(A,fetidp->pP,fetidp->pP,MAT_INITIAL_MATRIX,&C));
        CHKERRQ(MatFindNonzeroRows(C,&nzrows));
        if (nzrows) {
          PetscInt i;

          CHKERRQ(ISGetSize(nzrows,&i));
          CHKERRQ(ISDestroy(&nzrows));
          if (!i) pisz = PETSC_TRUE;
        }
        if (!pisz) {
          CHKERRQ(MatScale(C,-1.)); /* i.e. Almost Incompressible Elasticity, Stokes discretized with Q1xQ1_stabilized */
          CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_C",(PetscObject)C));
        }
        CHKERRQ(MatDestroy(&C));
      }
      /* Divergence mat */
      if (!pcbddc->divudotp) {
        Mat       B;
        IS        P;
        IS        l2l = NULL;
        PetscBool save;

        CHKERRQ(PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_aP",(PetscObject*)&P));
        if (!pisz) {
          IS       F,V;
          PetscInt m,M;

          CHKERRQ(MatGetOwnershipRange(A,&m,&M));
          CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)A),M-m,m,1,&F));
          CHKERRQ(ISComplement(P,m,M,&V));
          CHKERRQ(MatCreateSubMatrix(A,P,V,MAT_INITIAL_MATRIX,&B));
          {
            Mat_IS *Bmatis = (Mat_IS*)B->data;
            CHKERRQ(PetscObjectReference((PetscObject)Bmatis->getsub_cis));
            l2l  = Bmatis->getsub_cis;
          }
          CHKERRQ(ISDestroy(&V));
          CHKERRQ(ISDestroy(&F));
        } else {
          CHKERRQ(MatCreateSubMatrix(A,P,NULL,MAT_INITIAL_MATRIX,&B));
        }
        save = pcbddc->compute_nonetflux; /* SetDivergenceMat activates nonetflux computation */
        CHKERRQ(PCBDDCSetDivergenceMat(fetidp->innerbddc,B,PETSC_FALSE,l2l));
        pcbddc->compute_nonetflux = save;
        CHKERRQ(MatDestroy(&B));
        CHKERRQ(ISDestroy(&l2l));
      }
      if (A != Ap) { /* user has provided a different Pmat, this always superseeds the setter (TODO: is it OK?) */
        /* use monolithic operator, we restrict later */
        CHKERRQ(KSPFETIDPSetPressureOperator(ksp,Ap));
      }
      CHKERRQ(PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_PPmat",(PetscObject*)&PPmat));

      /* PPmat not present, use some default choice */
      if (!PPmat) {
        Mat C;

        CHKERRQ(PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_C",(PetscObject*)&C));
        if (!schp && C) { /* non-zero pressure block, most likely Almost Incompressible Elasticity */
          CHKERRQ(KSPFETIDPSetPressureOperator(ksp,C));
        } else if (!pisz && schp) { /* we need the whole pressure mass matrix to define the interface BDDC */
          IS  P;

          CHKERRQ(PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_aP",(PetscObject*)&P));
          CHKERRQ(MatCreateSubMatrix(A,P,P,MAT_INITIAL_MATRIX,&C));
          CHKERRQ(MatScale(C,-1.));
          CHKERRQ(KSPFETIDPSetPressureOperator(ksp,C));
          CHKERRQ(MatDestroy(&C));
        } else { /* identity (need to be scaled properly by the user using e.g. a Richardson method */
          PetscInt nl;

          CHKERRQ(ISGetLocalSize(fetidp->pP,&nl));
          CHKERRQ(MatCreate(PetscObjectComm((PetscObject)ksp),&C));
          CHKERRQ(MatSetSizes(C,nl,nl,totP,totP));
          CHKERRQ(MatSetType(C,MATAIJ));
          CHKERRQ(MatMPIAIJSetPreallocation(C,1,NULL,0,NULL));
          CHKERRQ(MatSeqAIJSetPreallocation(C,1,NULL));
          CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
          CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
          CHKERRQ(MatShift(C,1.));
          CHKERRQ(KSPFETIDPSetPressureOperator(ksp,C));
          CHKERRQ(MatDestroy(&C));
        }
      }

      /* Preconditioned operator for the pressure block */
      CHKERRQ(PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_PPmat",(PetscObject*)&PPmat));
      if (PPmat) {
        Mat      C;
        IS       Pall;
        PetscInt AM,PAM,PAN,pam,pan,am,an,pl,pIl,pAg,pIg;

        CHKERRQ(PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_aP",(PetscObject*)&Pall));
        CHKERRQ(MatGetSize(A,&AM,NULL));
        CHKERRQ(MatGetSize(PPmat,&PAM,&PAN));
        CHKERRQ(ISGetSize(Pall,&pAg));
        CHKERRQ(ISGetSize(fetidp->pP,&pIg));
        CHKERRQ(MatGetLocalSize(PPmat,&pam,&pan));
        CHKERRQ(MatGetLocalSize(A,&am,&an));
        CHKERRQ(ISGetLocalSize(Pall,&pIl));
        CHKERRQ(ISGetLocalSize(fetidp->pP,&pl));
        PetscCheckFalse(PAM != PAN,PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Pressure matrix must be square, unsupported %D x %D",PAM,PAN);
        PetscCheckFalse(pam != pan,PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Local sizes of pressure matrix must be equal, unsupported %D x %D",pam,pan);
        PetscCheckFalse(pam != am && pam != pl && pam != pIl,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local rows %D for pressure matrix! Supported are %D, %D or %D",pam,am,pl,pIl);
        PetscCheckFalse(pan != an && pan != pl && pan != pIl,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local columns %D for pressure matrix! Supported are %D, %D or %D",pan,an,pl,pIl);
        if (PAM == AM) { /* monolithic ordering, restrict to pressure */
          if (schp) {
            CHKERRQ(MatCreateSubMatrix(PPmat,Pall,Pall,MAT_INITIAL_MATRIX,&C));
          } else {
            CHKERRQ(MatCreateSubMatrix(PPmat,fetidp->pP,fetidp->pP,MAT_INITIAL_MATRIX,&C));
          }
        } else if (pAg == PAM) { /* global ordering for pressure only */
          if (!allp && !schp) { /* solving for interface pressure only */
            IS restr;

            CHKERRQ(ISRenumber(fetidp->pP,NULL,NULL,&restr));
            CHKERRQ(MatCreateSubMatrix(PPmat,restr,restr,MAT_INITIAL_MATRIX,&C));
            CHKERRQ(ISDestroy(&restr));
          } else {
            CHKERRQ(PetscObjectReference((PetscObject)PPmat));
            C    = PPmat;
          }
        } else if (pIg == PAM) { /* global ordering for selected pressure only */
          PetscCheck(!schp,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Need the entire matrix");
          CHKERRQ(PetscObjectReference((PetscObject)PPmat));
          C    = PPmat;
        } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Unable to use the pressure matrix");

        CHKERRQ(KSPFETIDPSetPressureOperator(ksp,C));
        CHKERRQ(MatDestroy(&C));
      } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Missing Pmat for pressure block");
    } else { /* totP == 0 */
      CHKERRQ(PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP",NULL));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(KSPFETIDPSetUpOperators(ksp));
  /* set up BDDC */
  CHKERRQ(PCSetErrorIfFailure(fetidp->innerbddc,ksp->errorifnotconverged));
  CHKERRQ(PCSetUp(fetidp->innerbddc));
  /* FETI-DP as it is implemented needs an exact coarse solver */
  if (pcbddc->coarse_ksp) {
    CHKERRQ(KSPSetTolerances(pcbddc->coarse_ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,1000));
    CHKERRQ(KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_DEFAULT));
  }
  /* FETI-DP as it is implemented needs exact local Neumann solvers */
  CHKERRQ(KSPSetTolerances(pcbddc->ksp_R,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,1000));
  CHKERRQ(KSPSetNormType(pcbddc->ksp_R,KSP_NORM_DEFAULT));

  /* setup FETI-DP operators
     If fetidp->statechanged is true, we need to update the operators
     needed in the saddle-point case. This should be replaced
     by a better logic when the FETI-DP matrix and preconditioner will
     have their own classes */
  if (pcbddc->new_primal_space || fetidp->statechanged) {
    Mat F; /* the FETI-DP matrix */
    PC  D; /* the FETI-DP preconditioner */
    CHKERRQ(KSPReset(fetidp->innerksp));
    CHKERRQ(PCBDDCCreateFETIDPOperators(fetidp->innerbddc,fetidp->fully_redundant,((PetscObject)ksp)->prefix,&F,&D));
    CHKERRQ(KSPSetOperators(fetidp->innerksp,F,F));
    CHKERRQ(KSPSetTolerances(fetidp->innerksp,ksp->rtol,ksp->abstol,ksp->divtol,ksp->max_it));
    CHKERRQ(KSPSetPC(fetidp->innerksp,D));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)D,(PetscObject)fetidp->innerksp,0));
    CHKERRQ(KSPSetFromOptions(fetidp->innerksp));
    CHKERRQ(MatCreateVecs(F,&(fetidp->innerksp)->vec_rhs,&(fetidp->innerksp)->vec_sol));
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(PCDestroy(&D));
    if (fetidp->check) {
      PetscViewer viewer;

      if (!pcbddc->dbg_viewer) {
        viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
      } else {
        viewer = pcbddc->dbg_viewer;
      }
      CHKERRQ(KSPFETIDPCheckOperators(ksp,viewer));
    }
  }
  fetidp->statechanged     = PETSC_FALSE;
  pcbddc->new_primal_space = PETSC_FALSE;

  /* propagate settings to the inner solve */
  CHKERRQ(KSPGetComputeSingularValues(ksp,&flg));
  CHKERRQ(KSPSetComputeSingularValues(fetidp->innerksp,flg));
  if (ksp->res_hist) {
    CHKERRQ(KSPSetResidualHistory(fetidp->innerksp,ksp->res_hist,ksp->res_hist_max,ksp->res_hist_reset));
  }
  CHKERRQ(KSPSetErrorIfNotConverged(fetidp->innerksp,ksp->errorifnotconverged));
  CHKERRQ(KSPSetUp(fetidp->innerksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_FETIDP(KSP ksp)
{
  Mat                F,A;
  MatNullSpace       nsp;
  Vec                X,B,Xl,Bl;
  KSP_FETIDP         *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC            *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  KSPConvergedReason reason;
  PC                 pc;
  PCFailedReason     pcreason;
  PetscInt           hist_len;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(citation,&cited));
  if (fetidp->saddlepoint) {
    CHKERRQ(PetscCitationsRegister(citation2,&cited2));
  }
  CHKERRQ(KSPGetOperators(ksp,&A,NULL));
  CHKERRQ(KSPGetRhs(ksp,&B));
  CHKERRQ(KSPGetSolution(ksp,&X));
  CHKERRQ(KSPGetOperators(fetidp->innerksp,&F,NULL));
  CHKERRQ(KSPGetRhs(fetidp->innerksp,&Bl));
  CHKERRQ(KSPGetSolution(fetidp->innerksp,&Xl));
  CHKERRQ(PCBDDCMatFETIDPGetRHS(F,B,Bl));
  if (ksp->transpose_solve) {
    CHKERRQ(KSPSolveTranspose(fetidp->innerksp,Bl,Xl));
  } else {
    CHKERRQ(KSPSolve(fetidp->innerksp,Bl,Xl));
  }
  CHKERRQ(KSPGetConvergedReason(fetidp->innerksp,&reason));
  CHKERRQ(KSPGetPC(fetidp->innerksp,&pc));
  CHKERRQ(PCGetFailedReason(pc,&pcreason));
  if ((reason < 0 && reason != KSP_DIVERGED_ITS) || pcreason) {
    PetscInt its;
    CHKERRQ(KSPGetIterationNumber(fetidp->innerksp,&its));
    ksp->reason = KSP_DIVERGED_PC_FAILED;
    CHKERRQ(VecSetInf(Xl));
    CHKERRQ(PetscInfo(ksp,"Inner KSP solve failed: %s %s at iteration %D",KSPConvergedReasons[reason],PCFailedReasons[pcreason],its));
  }
  CHKERRQ(PCBDDCMatFETIDPGetSolution(F,Xl,X));
  CHKERRQ(MatGetNullSpace(A,&nsp));
  if (nsp) {
    CHKERRQ(MatNullSpaceRemove(nsp,X));
  }
  /* update ksp with stats from inner ksp */
  CHKERRQ(KSPGetConvergedReason(fetidp->innerksp,&ksp->reason));
  CHKERRQ(KSPGetIterationNumber(fetidp->innerksp,&ksp->its));
  ksp->totalits += ksp->its;
  CHKERRQ(KSPGetResidualHistory(fetidp->innerksp,NULL,&hist_len));
  ksp->res_hist_len = (size_t) hist_len;
  /* restore defaults for inner BDDC (Pre/PostSolve flags) */
  pcbddc->temp_solution_used        = PETSC_FALSE;
  pcbddc->rhs_change                = PETSC_FALSE;
  pcbddc->exact_dirichlet_trick_app = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPReset_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC        *pcbddc;

  PetscFunctionBegin;
  CHKERRQ(ISDestroy(&fetidp->pP));
  CHKERRQ(VecDestroy(&fetidp->rhs_flip));
  /* avoid PCReset that does not take into account ref counting */
  CHKERRQ(PCDestroy(&fetidp->innerbddc));
  CHKERRQ(PCCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerbddc));
  CHKERRQ(PCSetType(fetidp->innerbddc,PCBDDC));
  pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  pcbddc->symmetric_primal = PETSC_FALSE;
  CHKERRQ(PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerbddc));
  CHKERRQ(KSPDestroy(&fetidp->innerksp));
  fetidp->saddlepoint  = PETSC_FALSE;
  fetidp->matstate     = -1;
  fetidp->matnnzstate  = -1;
  fetidp->statechanged = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  CHKERRQ(KSPReset_FETIDP(ksp));
  CHKERRQ(PCDestroy(&fetidp->innerbddc));
  CHKERRQ(KSPDestroy(&fetidp->innerksp));
  CHKERRQ(PetscFree(fetidp->monctx));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetInnerBDDC_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerBDDC_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerKSP_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetPressureOperator_C",NULL));
  CHKERRQ(PetscFree(ksp->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_FETIDP(KSP ksp,PetscViewer viewer)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  fully redundant: %d\n",fetidp->fully_redundant));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  saddle point:    %d\n",fetidp->saddlepoint));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Inner KSP solver details\n"));
  }
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(KSPView(fetidp->innerksp,viewer));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Inner BDDC solver details\n"));
  }
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PCView(fetidp->innerbddc,viewer));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_FETIDP(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  /* set options prefixes for the inner objects, since the parent prefix will be valid at this point */
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerksp,((PetscObject)ksp)->prefix));
  CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerksp,"fetidp_"));
  if (!fetidp->userbddc) {
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerbddc,((PetscObject)ksp)->prefix));
    CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerbddc,"fetidp_bddc_"));
  }
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"KSP FETIDP options"));
  CHKERRQ(PetscOptionsBool("-ksp_fetidp_fullyredundant","Use fully redundant multipliers","none",fetidp->fully_redundant,&fetidp->fully_redundant,NULL));
  CHKERRQ(PetscOptionsBool("-ksp_fetidp_saddlepoint","Activates support for saddle-point problems",NULL,fetidp->saddlepoint,&fetidp->saddlepoint,NULL));
  CHKERRQ(PetscOptionsBool("-ksp_fetidp_check","Activates verbose debugging output FETI-DP operators",NULL,fetidp->check,&fetidp->check,NULL));
  CHKERRQ(PetscOptionsTail());
  CHKERRQ(PCSetFromOptions(fetidp->innerbddc));
  PetscFunctionReturn(0);
}

/*MC
     KSPFETIDP - The FETI-DP method

   This class implements the FETI-DP method [1].
   The matrix for the KSP must be of type MATIS.
   The FETI-DP linear system (automatically generated constructing an internal PCBDDC object) is solved using an internal KSP object.

   Options Database Keys:
+   -ksp_fetidp_fullyredundant <false>   - use a fully redundant set of Lagrange multipliers
.   -ksp_fetidp_saddlepoint <false>      - activates support for saddle point problems, see [2]
.   -ksp_fetidp_saddlepoint_flip <false> - usually, an incompressible Stokes problem is written as
                                           | A B^T | | v | = | f |
                                           | B 0   | | p | = | g |
                                           with B representing -\int_\Omega \nabla \cdot u q.
                                           If -ksp_fetidp_saddlepoint_flip is true, the code assumes that the user provides it as
                                           | A B^T | | v | = | f |
                                           |-B 0   | | p | = |-g |
.   -ksp_fetidp_pressure_field <-1>      - activates support for saddle point problems, and identifies the pressure field id.
                                           If this information is not provided, the pressure field is detected by using MatFindZeroDiagonals().
-   -ksp_fetidp_pressure_all <false>     - if false, uses the interface pressures, as described in [2]. If true, uses the entire pressure field.

   Level: Advanced

   Notes:
    Options for the inner KSP and for the customization of the PCBDDC object can be specified at command line by using the prefixes -fetidp_ and -fetidp_bddc_. E.g.,
.vb
      -fetidp_ksp_type gmres -fetidp_bddc_pc_bddc_symmetric false
.ve
   will use GMRES for the solution of the linear system on the Lagrange multipliers, generated using a non-symmetric PCBDDC.

   For saddle point problems with continuous pressures, the preconditioned operator for the pressure solver can be specified with KSPFETIDPSetPressureOperator().
   Alternatively, the pressure operator is extracted from the precondioned matrix (if it is different from the linear solver matrix).
   If none of the above, an identity matrix will be created; the user then needs to scale it through a Richardson solver.
   Options for the pressure solver can be prefixed with -fetidp_fielsplit_p_, E.g.
.vb
      -fetidp_fielsplit_p_ksp_type preonly -fetidp_fielsplit_p_pc_type lu -fetidp_fielsplit_p_pc_factor_mat_solver_type mumps
.ve
   In order to use the deluxe version of FETI-DP, you must customize the inner BDDC operator with -fetidp_bddc_pc_bddc_use_deluxe_scaling -fetidp_bddc_pc_bddc_deluxe_singlemat and use
   non-redundant multipliers, i.e. -ksp_fetidp_fullyredundant false. Options for the scaling solver are prefixed by -fetidp_bddelta_, E.g.
.vb
      -fetidp_bddelta_pc_factor_mat_solver_type mumps -fetidp_bddelta_pc_type lu
.ve

   Some of the basic options such as the maximum number of iterations and tolerances are automatically passed from this KSP to the inner KSP that actually performs the iterations.

   The converged reason and number of iterations computed are passed from the inner KSP to this KSP at the end of the solution.

   Developer Notes:
    Even though this method does not directly use any norms, the user is allowed to set the KSPNormType to any value.
    This is so users do not have to change KSPNormType options when they switch from other KSP methods to this one.

   References:
+  * - C. Farhat, M. Lesoinne, P. LeTallec, K. Pierson, and D. Rixen, FETI-DP: a dual-primal unified FETI method. I. A faster alternative to the two-level FETI method, Internat. J. Numer. Methods Engrg., 50 (2001), pp. 1523--1544
-  * - X. Tu, J. Li, A FETI-DP type domain decomposition algorithm for three-dimensional incompressible Stokes equations, SIAM J. Numer. Anal., 53 (2015), pp. 720-742

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC(), KSPFETIDPGetInnerBDDC(), KSPFETIDPGetInnerKSP()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp;
  KSP_FETIDPMon  *monctx;
  PC_BDDC        *pcbddc;
  PC             pc;

  PetscFunctionBegin;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));

  CHKERRQ(PetscNewLog(ksp,&fetidp));
  fetidp->matstate     = -1;
  fetidp->matnnzstate  = -1;
  fetidp->statechanged = PETSC_TRUE;

  ksp->data = (void*)fetidp;
  ksp->ops->setup                        = KSPSetUp_FETIDP;
  ksp->ops->solve                        = KSPSolve_FETIDP;
  ksp->ops->destroy                      = KSPDestroy_FETIDP;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_FETIDP;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_FETIDP;
  ksp->ops->view                         = KSPView_FETIDP;
  ksp->ops->setfromoptions               = KSPSetFromOptions_FETIDP;
  ksp->ops->buildsolution                = KSPBuildSolution_FETIDP;
  ksp->ops->buildresidual                = KSPBuildResidualDefault;
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  /* create the inner KSP for the Lagrange multipliers */
  CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerksp));
  CHKERRQ(KSPGetPC(fetidp->innerksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerksp));
  /* monitor */
  CHKERRQ(PetscNew(&monctx));
  monctx->parentksp = ksp;
  fetidp->monctx = monctx;
  CHKERRQ(KSPMonitorSet(fetidp->innerksp,KSPMonitor_FETIDP,fetidp->monctx,NULL));
  /* create the inner BDDC */
  CHKERRQ(PCCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerbddc));
  CHKERRQ(PCSetType(fetidp->innerbddc,PCBDDC));
  /* make sure we always obtain a consistent FETI-DP matrix
     for symmetric problems, the user can always customize it through the command line */
  pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  pcbddc->symmetric_primal = PETSC_FALSE;
  CHKERRQ(PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerbddc));
  /* composed functions */
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetInnerBDDC_C",KSPFETIDPSetInnerBDDC_FETIDP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerBDDC_C",KSPFETIDPGetInnerBDDC_FETIDP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerKSP_C",KSPFETIDPGetInnerKSP_FETIDP));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetPressureOperator_C",KSPFETIDPSetPressureOperator_FETIDP));
  /* need to call KSPSetUp_FETIDP even with KSP_SETUP_NEWMATRIX */
  ksp->setupnewmatrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}

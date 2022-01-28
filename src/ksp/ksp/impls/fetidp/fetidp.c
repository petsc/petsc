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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (P) fetidp->saddlepoint = PETSC_TRUE;
  ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_PPmat",(PetscObject)P);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (P) PetscValidHeaderSpecific(P,MAT_CLASSID,2);
  ierr = PetscTryMethod(ksp,"KSPFETIDPSetPressureOperator_C",(KSP,Mat),(ksp,P));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(innerksp,2);
  ierr = PetscUseMethod(ksp,"KSPFETIDPGetInnerKSP_C",(KSP,KSP*),(ksp,innerksp));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(pc,2);
  ierr = PetscUseMethod(ksp,"KSPFETIDPGetInnerBDDC_C",(KSP,PC*),(ksp,pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPSetInnerBDDC_FETIDP(KSP ksp, PC pc)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
  ierr = PCDestroy(&fetidp->innerbddc);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCBDDC,&isbddc);CHKERRQ(ierr);
  if (!isbddc) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"KSPFETIDPSetInnerBDDC need a PCBDDC preconditioner");
  ierr = PetscTryMethod(ksp,"KSPFETIDPSetInnerBDDC_C",(KSP,PC),(ksp,pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPBuildSolution_FETIDP(KSP ksp,Vec v,Vec *V)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  Mat            F;
  Vec            Xl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetOperators(fetidp->innerksp,&F,NULL);CHKERRQ(ierr);
  ierr = KSPBuildSolution(fetidp->innerksp,NULL,&Xl);CHKERRQ(ierr);
  if (v) {
    ierr = PCBDDCMatFETIDPGetSolution(F,Xl,v);CHKERRQ(ierr);
    *V   = v;
  } else {
    ierr = PCBDDCMatFETIDPGetSolution(F,Xl,*V);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMonitor_FETIDP(KSP ksp,PetscInt it,PetscReal rnorm,void* ctx)
{
  KSP_FETIDPMon  *monctx = (KSP_FETIDPMon*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitor(monctx->parentksp,it,rnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPComputeEigenvalues_FETIDP(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPComputeEigenvalues(fetidp->innerksp,nmax,r,c,neig);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPComputeExtremeSingularValues_FETIDP(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPComputeExtremeSingularValues(fetidp->innerksp,emax,emin);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckSameComm(ksp,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(comm,PETSC_ERR_SUP,"Unsupported viewer");
  ierr = PetscViewerASCIIPrintf(viewer,"----------FETI-DP MAT  --------------\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,2);CHKERRQ(ierr);
  ierr = KSPGetOperators(fetidp->innerksp,&F,NULL);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(F,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,2);CHKERRQ(ierr);
  ierr = MatShellGetContext(F,&fetidpmat_ctx);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"----------FETI-DP TESTS--------------\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"All tests should return zero!\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"FETIDP MAT context in the ");CHKERRQ(ierr);
  if (fetidp->fully_redundant) {
    ierr = PetscViewerASCIIPrintf(viewer,"fully redundant case for lagrange multipliers.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"Non-fully redundant case for lagrange multiplier.\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

  /* Get Vertices used to define the BDDC */
  ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isvert,&n_vertices);CHKERRQ(ierr);
  ierr = ISGetIndices(isvert,&vertex_indices);CHKERRQ(ierr);

  /******************************************************************/
  /* TEST A/B: Test numbering of global fetidp dofs                 */
  /******************************************************************/
  ierr = MatCreateVecs(F,&fetidp_global,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(fetidpmat_ctx->lambda_local,&test_vec);CHKERRQ(ierr);
  ierr = VecSet(fetidp_global,1.0);CHKERRQ(ierr);
  ierr = VecSet(test_vec,1.);CHKERRQ(ierr);
  ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (fetidpmat_ctx->l2g_p) {
    ierr = VecDuplicate(fetidpmat_ctx->vP,&test_vec_p);CHKERRQ(ierr);
    ierr = VecSet(test_vec_p,1.);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_p,fetidp_global,fetidpmat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_p,fetidp_global,fetidpmat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  ierr = VecAXPY(test_vec,-1.0,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecNorm(test_vec,NORM_INFINITY,&val);CHKERRQ(ierr);
  ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
  ierr = MPI_Reduce(&val,&rval,1,MPIU_REAL,MPI_MAX,0,comm);CHKERRMPI(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"A: CHECK glob to loc: % 1.14e\n",rval);CHKERRQ(ierr);

  if (fetidpmat_ctx->l2g_p) {
    ierr = VecAXPY(test_vec_p,-1.0,fetidpmat_ctx->vP);CHKERRQ(ierr);
    ierr = VecNorm(test_vec_p,NORM_INFINITY,&val);CHKERRQ(ierr);
    ierr = MPI_Reduce(&val,&rval,1,MPIU_REAL,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"A: CHECK glob to loc (p): % 1.14e\n",rval);CHKERRQ(ierr);
  }

  if (fetidp->fully_redundant) {
    ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
    ierr = VecSet(fetidpmat_ctx->lambda_local,0.5);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecSum(fetidp_global,&sval);CHKERRQ(ierr);
    val  = PetscRealPart(sval)-fetidpmat_ctx->n_lambda;
    ierr = MPI_Reduce(&val,&rval,1,MPIU_REAL,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"B: CHECK loc to glob: % 1.14e\n",rval);CHKERRQ(ierr);
  }

  if (fetidpmat_ctx->l2g_p) {
    ierr = VecSet(pcis->vec1_N,1.0);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
    ierr = VecSet(fetidpmat_ctx->vP,-1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->g2g_p,fetidp_global,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->g2g_p,fetidp_global,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->g2g_p,pcis->vec1_global,fetidp_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->g2g_p,pcis->vec1_global,fetidp_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecSum(fetidp_global,&sval);CHKERRQ(ierr);
    val  = PetscRealPart(sval);
    ierr = MPI_Reduce(&val,&rval,1,MPIU_REAL,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"B: CHECK loc to glob (p): % 1.14e\n",rval);CHKERRQ(ierr);
  }

  /******************************************************************/
  /* TEST C: It should hold B_delta*w=0, w\in\widehat{W}            */
  /* This is the meaning of the B matrix                            */
  /******************************************************************/

  ierr = VecSetRandom(pcis->vec1_N,NULL);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Action of B_delta */
  ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecNorm(fetidp_global,NORM_INFINITY,&val);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"C: CHECK infty norm of B_delta*w (w continuous): % 1.14e\n",val);CHKERRQ(ierr);

  /******************************************************************/
  /* TEST D: It should hold E_Dw = w - P_Dw w\in\widetilde{W}       */
  /* E_D = R_D^TR                                                   */
  /* P_D = B_{D,delta}^T B_{delta}                                  */
  /* eq.44 Mandel Tezaur and Dohrmann 2005                          */
  /******************************************************************/

  /* compute a random vector in \widetilde{W} */
  ierr = VecSetRandom(pcis->vec1_N,NULL);CHKERRQ(ierr);
  /* set zero at vertices and essential dofs */
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<n_vertices;i++) array[vertex_indices[i]] = 0.0;
  ierr = PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph,&dirdofs);CHKERRQ(ierr);
  if (dirdofs) {
    const PetscInt *idxs;
    PetscInt       ndir;

    ierr = ISGetLocalSize(dirdofs,&ndir);CHKERRQ(ierr);
    ierr = ISGetIndices(dirdofs,&idxs);CHKERRQ(ierr);
    for (i=0;i<ndir;i++) array[idxs[i]] = 0.0;
    ierr = ISRestoreIndices(dirdofs,&idxs);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  /* store w for final comparison */
  ierr = VecDuplicate(pcis->vec1_B,&test_vec);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,test_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,test_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* Jump operator P_D : results stored in pcis->vec1_B */
  /* Action of B_delta */
  ierr = MatMult(fetidpmat_ctx->B_delta,test_vec,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Action of B_Ddelta^T */
  ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);

  /* Average operator E_D : results stored in pcis->vec2_B */
  ierr = PCBDDCScalingExtension(fetidpmat_ctx->pc,test_vec,pcis->vec1_global);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* test E_D=I-P_D */
  ierr = VecAXPY(pcis->vec1_B,1.0,pcis->vec2_B);CHKERRQ(ierr);
  ierr = VecAXPY(pcis->vec1_B,-1.0,test_vec);CHKERRQ(ierr);
  ierr = VecNorm(pcis->vec1_B,NORM_INFINITY,&val);CHKERRQ(ierr);
  ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
  ierr = MPI_Reduce(&val,&rval,1,MPIU_REAL,MPI_MAX,0,comm);CHKERRMPI(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"D: CHECK infty norm of E_D + P_D - I: % 1.14e\n",PetscGlobalRank,val);CHKERRQ(ierr);

  /******************************************************************/
  /* TEST E: It should hold R_D^TP_Dw=0 w\in\widetilde{W}           */
  /* eq.48 Mandel Tezaur and Dohrmann 2005                          */
  /******************************************************************/

  ierr = VecSetRandom(pcis->vec1_N,NULL);CHKERRQ(ierr);
  /* set zero at vertices and essential dofs */
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<n_vertices;i++) array[vertex_indices[i]] = 0.0;
  if (dirdofs) {
    const PetscInt *idxs;
    PetscInt       ndir;

    ierr = ISGetLocalSize(dirdofs,&ndir);CHKERRQ(ierr);
    ierr = ISGetIndices(dirdofs,&idxs);CHKERRQ(ierr);
    for (i=0;i<ndir;i++) array[idxs[i]] = 0.0;
    ierr = ISRestoreIndices(dirdofs,&idxs);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);

  /* Jump operator P_D : results stored in pcis->vec1_B */

  ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Action of B_delta */
  ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Action of B_Ddelta^T */
  ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
  /* scaling */
  ierr = PCBDDCScalingExtension(fetidpmat_ctx->pc,pcis->vec1_B,pcis->vec1_global);CHKERRQ(ierr);
  ierr = VecNorm(pcis->vec1_global,NORM_INFINITY,&val);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"E: CHECK infty norm of R^T_D P_D: % 1.14e\n",val);CHKERRQ(ierr);

  if (!fetidp->fully_redundant) {
    /******************************************************************/
    /* TEST F: It should holds B_{delta}B^T_{D,delta}=I               */
    /* Corollary thm 14 Mandel Tezaur and Dohrmann 2005               */
    /******************************************************************/
    ierr = VecDuplicate(fetidp_global,&test_vec);CHKERRQ(ierr);
    ierr = VecSetRandom(fetidp_global,NULL);CHKERRQ(ierr);
    if (fetidpmat_ctx->l2g_p) {
      ierr = VecSet(fetidpmat_ctx->vP,0.);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
    /* Action of B_Ddelta^T */
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
    /* Action of B_delta */
    ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecSet(test_vec,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecAXPY(fetidp_global,-1.,test_vec);CHKERRQ(ierr);
    ierr = VecNorm(fetidp_global,NORM_INFINITY,&val);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"E: CHECK infty norm of P^T_D - I: % 1.14e\n",val);CHKERRQ(ierr);
    ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"-------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&test_vec_p);CHKERRQ(ierr);
  ierr = ISDestroy(&dirdofs);CHKERRQ(ierr);
  ierr = VecDestroy(&fetidp_global);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isvert,&vertex_indices);CHKERRQ(ierr);
  ierr = PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert);CHKERRQ(ierr);
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
  ierr = PetscOptionsInt("-ksp_fetidp_pressure_field","Field id for pressures for saddle-point problems",NULL,fid,&fid,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_fetidp_pressure_all","Use the whole pressure set instead of just that at the interface",NULL,allp,&allp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_fetidp_saddlepoint_flip","Flip the sign of the pressure-velocity (lower-left) block",NULL,flip,&flip,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_fetidp_pressure_schur","Use a BDDC solver for pressure",NULL,schp,&schp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)ksp),&size);CHKERRMPI(ierr);
  fetidp->saddlepoint = (fid >= 0 ? PETSC_TRUE : fetidp->saddlepoint);
  if (size == 1) fetidp->saddlepoint = PETSC_FALSE;

  ierr = KSPGetOperators(ksp,&A,&Ap);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis);CHKERRQ(ierr);
  if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Amat should be of type MATIS");

  /* Quiet return if the matrix states are unchanged.
     Needed only for the saddle point case since it uses MatZeroRows
     on a matrix that may not have changed */
  ierr = PetscObjectStateGet((PetscObject)A,&matstate);CHKERRQ(ierr);
  ierr = MatGetNonzeroState(A,&matnnzstate);CHKERRQ(ierr);
  if (matstate == fetidp->matstate && matnnzstate == fetidp->matnnzstate) PetscFunctionReturn(0);
  fetidp->matstate     = matstate;
  fetidp->matnnzstate  = matnnzstate;
  fetidp->statechanged = fetidp->saddlepoint;

  /* see if we have some fields attached */
  if (!pcbddc->n_ISForDofsLocal && !pcbddc->n_ISForDofs) {
    DM             dm;
    PetscContainer c;

    ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)A,"_convert_nest_lfields",(PetscObject*)&c);CHKERRQ(ierr);
    if (dm) {
      IS      *fields;
      PetscInt nf,i;

      ierr = DMCreateFieldDecomposition(dm,&nf,NULL,&fields,NULL);CHKERRQ(ierr);
      ierr = PCBDDCSetDofsSplitting(fetidp->innerbddc,nf,fields);CHKERRQ(ierr);
      for (i=0;i<nf;i++) {
        ierr = ISDestroy(&fields[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(fields);CHKERRQ(ierr);
    } else if (c) {
      MatISLocalFields lf;

      ierr = PetscContainerGetPointer(c,(void**)&lf);CHKERRQ(ierr);
      ierr = PCBDDCSetDofsSplittingLocal(fetidp->innerbddc,lf->nr,lf->rf);CHKERRQ(ierr);
    }
  }

  if (!fetidp->saddlepoint) {
    ierr = PCSetOperators(fetidp->innerbddc,A,A);CHKERRQ(ierr);
  } else {
    Mat          nA,lA,PPmat;
    MatNullSpace nnsp;
    IS           pP;
    PetscInt     totP;

    ierr = MatISGetLocalMat(A,&lA);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lA",(PetscObject)lA);CHKERRQ(ierr);

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

      ierr = MatGetLocalSize(A,&nl,NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(A,&rst,&ren);CHKERRQ(ierr);
      ierr = MatGetLocalSize(lA,&n,NULL);CHKERRQ(ierr);
      ierr = MatGetLocalToGlobalMapping(A,&l2g,NULL);CHKERRQ(ierr);

      if (!pcis->is_I_local) { /* need to compute interior dofs */
        ierr = PetscCalloc1(n,&count);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetInfo(l2g,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
        for (i=1;i<n_neigh;i++)
          for (j=0;j<n_shared[i];j++)
            count[shared[i][j]] += 1;
        for (i=0,j=0;i<n;i++) if (!count[i]) count[j++] = i;
        ierr = ISLocalToGlobalMappingRestoreInfo(l2g,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,j,count,PETSC_OWN_POINTER,&II);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)pcis->is_I_local);CHKERRQ(ierr);
        II   = pcis->is_I_local;
      }

      /* interior dofs in layout */
      ierr = PetscArrayzero(matis->sf_leafdata,n);CHKERRQ(ierr);
      ierr = PetscArrayzero(matis->sf_rootdata,nl);CHKERRQ(ierr);
      ierr = ISGetLocalSize(II,&ni);CHKERRQ(ierr);
      ierr = ISGetIndices(II,&idxs);CHKERRQ(ierr);
      for (i=0;i<ni;i++) matis->sf_leafdata[idxs[i]] = 1;
      ierr = ISRestoreIndices(II,&idxs);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscMalloc1(PetscMax(nl,n),&widxs);CHKERRQ(ierr);
      for (i=0,ni=0;i<nl;i++) if (matis->sf_rootdata[i]) widxs[ni++] = i+rst;
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&pII);CHKERRQ(ierr);

      /* pressure dofs */
      Pall  = NULL;
      lPall = NULL;
      ploc  = PETSC_FALSE;
      if (fid < 0) { /* zero pressure block */
        PetscInt np;

        ierr = MatFindZeroDiagonals(A,&Pall);CHKERRQ(ierr);
        ierr = ISGetSize(Pall,&np);CHKERRQ(ierr);
        if (!np) { /* zero-block not found, defaults to last field (if set) */
          fid  = pcbddc->n_ISForDofsLocal ? pcbddc->n_ISForDofsLocal - 1 : pcbddc->n_ISForDofs - 1;
          ierr = ISDestroy(&Pall);CHKERRQ(ierr);
        } else if (!pcbddc->n_ISForDofsLocal && !pcbddc->n_ISForDofs) {
          ierr = PCBDDCSetDofsSplitting(fetidp->innerbddc,1,&Pall);CHKERRQ(ierr);
        }
      }
      if (!Pall) { /* look for registered fields */
        if (pcbddc->n_ISForDofsLocal) {
          PetscInt np;

          if (fid < 0 || fid >= pcbddc->n_ISForDofsLocal) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Invalid field id for pressure %D, max %D",fid,pcbddc->n_ISForDofsLocal);
          /* need a sequential IS */
          ierr = ISGetLocalSize(pcbddc->ISForDofsLocal[fid],&np);CHKERRQ(ierr);
          ierr = ISGetIndices(pcbddc->ISForDofsLocal[fid],&idxs);CHKERRQ(ierr);
          ierr = ISCreateGeneral(PETSC_COMM_SELF,np,idxs,PETSC_COPY_VALUES,&lPall);CHKERRQ(ierr);
          ierr = ISRestoreIndices(pcbddc->ISForDofsLocal[fid],&idxs);CHKERRQ(ierr);
          ploc = PETSC_TRUE;
        } else if (pcbddc->n_ISForDofs) {
          if (fid < 0 || fid >= pcbddc->n_ISForDofs) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Invalid field id for pressure %D, max %D",fid,pcbddc->n_ISForDofs);
          ierr = PetscObjectReference((PetscObject)pcbddc->ISForDofs[fid]);CHKERRQ(ierr);
          Pall = pcbddc->ISForDofs[fid];
        } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Cannot detect pressure field! Use KSPFETIDPGetInnerBDDC() + PCBDDCSetDofsSplitting or PCBDDCSetDofsSplittingLocal");
      }

      /* if the user requested the entire pressure,
         remove the interior pressure dofs from II (or pII) */
      if (allp) {
        if (ploc) {
          IS nII;
          ierr = ISDifference(II,lPall,&nII);CHKERRQ(ierr);
          ierr = ISDestroy(&II);CHKERRQ(ierr);
          II   = nII;
        } else {
          IS nII;
          ierr = ISDifference(pII,Pall,&nII);CHKERRQ(ierr);
          ierr = ISDestroy(&pII);CHKERRQ(ierr);
          pII  = nII;
        }
      }
      if (ploc) {
        ierr = ISDifference(lPall,II,&lP);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lP",(PetscObject)lP);CHKERRQ(ierr);
      } else {
        ierr = ISDifference(Pall,pII,&pP);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP",(PetscObject)pP);CHKERRQ(ierr);
        /* need all local pressure dofs */
        ierr = PetscArrayzero(matis->sf_leafdata,n);CHKERRQ(ierr);
        ierr = PetscArrayzero(matis->sf_rootdata,nl);CHKERRQ(ierr);
        ierr = ISGetLocalSize(Pall,&ni);CHKERRQ(ierr);
        ierr = ISGetIndices(Pall,&idxs);CHKERRQ(ierr);
        for (i=0;i<ni;i++) matis->sf_rootdata[idxs[i]-rst] = 1;
        ierr = ISRestoreIndices(Pall,&idxs);CHKERRQ(ierr);
        ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
        for (i=0,ni=0;i<n;i++) if (matis->sf_leafdata[i]) widxs[ni++] = i;
        ierr = ISCreateGeneral(PETSC_COMM_SELF,ni,widxs,PETSC_COPY_VALUES,&lPall);CHKERRQ(ierr);
      }

      if (!Pall) {
        ierr = PetscArrayzero(matis->sf_leafdata,n);CHKERRQ(ierr);
        ierr = PetscArrayzero(matis->sf_rootdata,nl);CHKERRQ(ierr);
        ierr = ISGetLocalSize(lPall,&ni);CHKERRQ(ierr);
        ierr = ISGetIndices(lPall,&idxs);CHKERRQ(ierr);
        for (i=0;i<ni;i++) matis->sf_leafdata[idxs[i]] = 1;
        ierr = ISRestoreIndices(lPall,&idxs);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE);CHKERRQ(ierr);
        for (i=0,ni=0;i<nl;i++) if (matis->sf_rootdata[i]) widxs[ni++] = i+rst;
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&Pall);CHKERRQ(ierr);
      }
      ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_aP",(PetscObject)Pall);CHKERRQ(ierr);

      if (flip) {
        PetscInt npl;
        ierr = ISGetLocalSize(Pall,&npl);CHKERRQ(ierr);
        ierr = ISGetIndices(Pall,&idxs);CHKERRQ(ierr);
        ierr = MatCreateVecs(A,NULL,&fetidp->rhs_flip);CHKERRQ(ierr);
        ierr = VecSet(fetidp->rhs_flip,1.);CHKERRQ(ierr);
        ierr = VecSetOption(fetidp->rhs_flip,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
        for (i=0;i<npl;i++) {
          ierr = VecSetValue(fetidp->rhs_flip,idxs[i],-1.,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(fetidp->rhs_flip);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(fetidp->rhs_flip);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_flip",(PetscObject)fetidp->rhs_flip);CHKERRQ(ierr);
        ierr = ISRestoreIndices(Pall,&idxs);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&Pall);CHKERRQ(ierr);
      ierr = ISDestroy(&pII);CHKERRQ(ierr);

      /* local selected pressures in subdomain-wise and global ordering */
      ierr = PetscArrayzero(matis->sf_leafdata,n);CHKERRQ(ierr);
      ierr = PetscArrayzero(matis->sf_rootdata,nl);CHKERRQ(ierr);
      if (!ploc) {
        PetscInt *widxs2;

        if (!pP) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Missing parallel pressure IS");
        ierr = ISGetLocalSize(pP,&ni);CHKERRQ(ierr);
        ierr = ISGetIndices(pP,&idxs);CHKERRQ(ierr);
        for (i=0;i<ni;i++) matis->sf_rootdata[idxs[i]-rst] = 1;
        ierr = ISRestoreIndices(pP,&idxs);CHKERRQ(ierr);
        ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
        for (i=0,ni=0;i<n;i++) if (matis->sf_leafdata[i]) widxs[ni++] = i;
        ierr = PetscMalloc1(ni,&widxs2);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingApply(l2g,ni,widxs,widxs2);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,ni,widxs,PETSC_COPY_VALUES,&lP);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lP",(PetscObject)lP);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs2,PETSC_OWN_POINTER,&is1);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_gP",(PetscObject)is1);CHKERRQ(ierr);
        ierr = ISDestroy(&is1);CHKERRQ(ierr);
      } else {
        if (!lP) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing sequential pressure IS");
        ierr = ISGetLocalSize(lP,&ni);CHKERRQ(ierr);
        ierr = ISGetIndices(lP,&idxs);CHKERRQ(ierr);
        for (i=0;i<ni;i++)
          if (idxs[i] >=0 && idxs[i] < n)
            matis->sf_leafdata[idxs[i]] = 1;
        ierr = ISRestoreIndices(lP,&idxs);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingApply(l2g,ni,idxs,widxs);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_gP",(PetscObject)is1);CHKERRQ(ierr);
        ierr = ISDestroy(&is1);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_REPLACE);CHKERRQ(ierr);
        for (i=0,ni=0;i<nl;i++) if (matis->sf_rootdata[i]) widxs[ni++] = i+rst;
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),ni,widxs,PETSC_COPY_VALUES,&pP);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP",(PetscObject)pP);CHKERRQ(ierr);
      }
      ierr = PetscFree(widxs);CHKERRQ(ierr);

      /* If there's any "interior pressure",
         we may want to use a discrete harmonic solver instead
         of a Stokes harmonic for the Dirichlet preconditioner
         Need to extract the interior velocity dofs in interior dofs ordering (iV)
         and interior pressure dofs in local ordering (iP) */
      if (!allp) {
        ISLocalToGlobalMapping l2g_t;

        ierr = ISDifference(lPall,lP,&is1);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_iP",(PetscObject)is1);CHKERRQ(ierr);
        ierr = ISDifference(II,is1,&is2);CHKERRQ(ierr);
        ierr = ISDestroy(&is1);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingCreateIS(II,&l2g_t);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApplyIS(l2g_t,IS_GTOLM_DROP,is2,&is1);CHKERRQ(ierr);
        ierr = ISGetLocalSize(is1,&i);CHKERRQ(ierr);
        ierr = ISGetLocalSize(is2,&j);CHKERRQ(ierr);
        if (i != j) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent local sizes %D and %D for iV",i,j);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_iV",(PetscObject)is1);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&l2g_t);CHKERRQ(ierr);
        ierr = ISDestroy(&is1);CHKERRQ(ierr);
        ierr = ISDestroy(&is2);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&II);CHKERRQ(ierr);

      /* exclude selected pressures from the inner BDDC */
      if (pcbddc->DirichletBoundariesLocal) {
        IS       list[2],plP,isout;
        PetscInt np;

        /* need a parallel IS */
        ierr = ISGetLocalSize(lP,&np);CHKERRQ(ierr);
        ierr = ISGetIndices(lP,&idxs);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),np,idxs,PETSC_USE_POINTER,&plP);CHKERRQ(ierr);
        list[0] = plP;
        list[1] = pcbddc->DirichletBoundariesLocal;
        ierr = ISConcatenate(PetscObjectComm((PetscObject)ksp),2,list,&isout);CHKERRQ(ierr);
        ierr = ISSortRemoveDups(isout);CHKERRQ(ierr);
        ierr = ISDestroy(&plP);CHKERRQ(ierr);
        ierr = ISRestoreIndices(lP,&idxs);CHKERRQ(ierr);
        ierr = PCBDDCSetDirichletBoundariesLocal(fetidp->innerbddc,isout);CHKERRQ(ierr);
        ierr = ISDestroy(&isout);CHKERRQ(ierr);
      } else if (pcbddc->DirichletBoundaries) {
        IS list[2],isout;

        list[0] = pP;
        list[1] = pcbddc->DirichletBoundaries;
        ierr = ISConcatenate(PetscObjectComm((PetscObject)ksp),2,list,&isout);CHKERRQ(ierr);
        ierr = ISSortRemoveDups(isout);CHKERRQ(ierr);
        ierr = PCBDDCSetDirichletBoundaries(fetidp->innerbddc,isout);CHKERRQ(ierr);
        ierr = ISDestroy(&isout);CHKERRQ(ierr);
      } else {
        IS       plP;
        PetscInt np;

        /* need a parallel IS */
        ierr = ISGetLocalSize(lP,&np);CHKERRQ(ierr);
        ierr = ISGetIndices(lP,&idxs);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ksp),np,idxs,PETSC_COPY_VALUES,&plP);CHKERRQ(ierr);
        ierr = PCBDDCSetDirichletBoundariesLocal(fetidp->innerbddc,plP);CHKERRQ(ierr);
        ierr = ISDestroy(&plP);CHKERRQ(ierr);
        ierr = ISRestoreIndices(lP,&idxs);CHKERRQ(ierr);
      }

      /* save CSR information for the pressure BDDC solver (if any) */
      if (schp) {
        PetscInt np,nt;

        ierr = MatGetSize(matis->A,&nt,NULL);CHKERRQ(ierr);
        ierr = ISGetLocalSize(lP,&np);CHKERRQ(ierr);
        if (np) {
          PetscInt *xadj = pcbddc->mat_graph->xadj;
          PetscInt *adjn = pcbddc->mat_graph->adjncy;
          PetscInt nv = pcbddc->mat_graph->nvtxs_csr;

          if (nv && nv == nt) {
            ISLocalToGlobalMapping pmap;
            PetscInt               *schp_csr,*schp_xadj,*schp_adjn,p;
            PetscContainer         c;

            ierr = ISLocalToGlobalMappingCreateIS(lPall,&pmap);CHKERRQ(ierr);
            ierr = ISGetIndices(lPall,&idxs);CHKERRQ(ierr);
            for (p = 0, nv = 0; p < np; p++) {
              PetscInt x,n = idxs[p];

              ierr = ISGlobalToLocalMappingApply(pmap,IS_GTOLM_DROP,xadj[n+1]-xadj[n],adjn+xadj[n],&x,NULL);CHKERRQ(ierr);
              nv  += x;
            }
            ierr = PetscMalloc1(np + 1 + nv,&schp_csr);CHKERRQ(ierr);
            schp_xadj = schp_csr;
            schp_adjn = schp_csr + np + 1;
            for (p = 0, schp_xadj[0] = 0; p < np; p++) {
              PetscInt x,n = idxs[p];

              ierr = ISGlobalToLocalMappingApply(pmap,IS_GTOLM_DROP,xadj[n+1]-xadj[n],adjn+xadj[n],&x,schp_adjn + schp_xadj[p]);CHKERRQ(ierr);
              schp_xadj[p+1] = schp_xadj[p] + x;
            }
            ierr = ISRestoreIndices(lPall,&idxs);CHKERRQ(ierr);
            ierr = ISLocalToGlobalMappingDestroy(&pmap);CHKERRQ(ierr);
            ierr = PetscContainerCreate(PETSC_COMM_SELF,&c);CHKERRQ(ierr);
            ierr = PetscContainerSetPointer(c,schp_csr);CHKERRQ(ierr);
            ierr = PetscContainerSetUserDestroy(c,PetscContainerUserDestroyDefault);CHKERRQ(ierr);
            ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pCSR",(PetscObject)c);CHKERRQ(ierr);
            ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);

          }
        }
      }
      ierr = ISDestroy(&lPall);CHKERRQ(ierr);
      ierr = ISDestroy(&lP);CHKERRQ(ierr);
      fetidp->pP = pP;
    }

    /* total number of selected pressure dofs */
    ierr = ISGetSize(fetidp->pP,&totP);CHKERRQ(ierr);

    /* Set operator for inner BDDC */
    if (totP || fetidp->rhs_flip) {
      ierr = MatDuplicate(A,MAT_COPY_VALUES,&nA);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
      nA   = A;
    }
    if (fetidp->rhs_flip) {
      ierr = MatDiagonalScale(nA,fetidp->rhs_flip,NULL);CHKERRQ(ierr);
      if (totP) {
        Mat lA2;

        ierr = MatISGetLocalMat(nA,&lA);CHKERRQ(ierr);
        ierr = MatDuplicate(lA,MAT_COPY_VALUES,&lA2);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lA",(PetscObject)lA2);CHKERRQ(ierr);
        ierr = MatDestroy(&lA2);CHKERRQ(ierr);
      }
    }

    if (totP) {
      ierr = MatSetOption(nA,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatZeroRowsColumnsIS(nA,fetidp->pP,1.,NULL,NULL);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_lA",NULL);CHKERRQ(ierr);
    }
    ierr = MatGetNearNullSpace(Ap,&nnsp);CHKERRQ(ierr);
    if (!nnsp) {
      ierr = MatGetNullSpace(Ap,&nnsp);CHKERRQ(ierr);
    }
    if (!nnsp) {
      ierr = MatGetNearNullSpace(A,&nnsp);CHKERRQ(ierr);
    }
    if (!nnsp) {
      ierr = MatGetNullSpace(A,&nnsp);CHKERRQ(ierr);
    }
    ierr = MatSetNearNullSpace(nA,nnsp);CHKERRQ(ierr);
    ierr = PCSetOperators(fetidp->innerbddc,nA,nA);CHKERRQ(ierr);
    ierr = MatDestroy(&nA);CHKERRQ(ierr);

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

        ierr = MatCreateSubMatrix(A,fetidp->pP,fetidp->pP,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
        ierr = MatFindNonzeroRows(C,&nzrows);CHKERRQ(ierr);
        if (nzrows) {
          PetscInt i;

          ierr = ISGetSize(nzrows,&i);CHKERRQ(ierr);
          ierr = ISDestroy(&nzrows);CHKERRQ(ierr);
          if (!i) pisz = PETSC_TRUE;
        }
        if (!pisz) {
          ierr = MatScale(C,-1.);CHKERRQ(ierr); /* i.e. Almost Incompressible Elasticity, Stokes discretized with Q1xQ1_stabilized */
          ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_C",(PetscObject)C);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&C);CHKERRQ(ierr);
      }
      /* Divergence mat */
      if (!pcbddc->divudotp) {
        Mat       B;
        IS        P;
        IS        l2l = NULL;
        PetscBool save;

        ierr = PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_aP",(PetscObject*)&P);CHKERRQ(ierr);
        if (!pisz) {
          IS       F,V;
          PetscInt m,M;

          ierr = MatGetOwnershipRange(A,&m,&M);CHKERRQ(ierr);
          ierr = ISCreateStride(PetscObjectComm((PetscObject)A),M-m,m,1,&F);CHKERRQ(ierr);
          ierr = ISComplement(P,m,M,&V);CHKERRQ(ierr);
          ierr = MatCreateSubMatrix(A,P,V,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
          {
            Mat_IS *Bmatis = (Mat_IS*)B->data;
            ierr = PetscObjectReference((PetscObject)Bmatis->getsub_cis);CHKERRQ(ierr);
            l2l  = Bmatis->getsub_cis;
          }
          ierr = ISDestroy(&V);CHKERRQ(ierr);
          ierr = ISDestroy(&F);CHKERRQ(ierr);
        } else {
          ierr = MatCreateSubMatrix(A,P,NULL,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
        }
        save = pcbddc->compute_nonetflux; /* SetDivergenceMat activates nonetflux computation */
        ierr = PCBDDCSetDivergenceMat(fetidp->innerbddc,B,PETSC_FALSE,l2l);CHKERRQ(ierr);
        pcbddc->compute_nonetflux = save;
        ierr = MatDestroy(&B);CHKERRQ(ierr);
        ierr = ISDestroy(&l2l);CHKERRQ(ierr);
      }
      if (A != Ap) { /* user has provided a different Pmat, this always superseeds the setter (TODO: is it OK?) */
        /* use monolithic operator, we restrict later */
        ierr = KSPFETIDPSetPressureOperator(ksp,Ap);CHKERRQ(ierr);
      }
      ierr = PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_PPmat",(PetscObject*)&PPmat);CHKERRQ(ierr);

      /* PPmat not present, use some default choice */
      if (!PPmat) {
        Mat C;

        ierr = PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_C",(PetscObject*)&C);CHKERRQ(ierr);
        if (!schp && C) { /* non-zero pressure block, most likely Almost Incompressible Elasticity */
          ierr = KSPFETIDPSetPressureOperator(ksp,C);CHKERRQ(ierr);
        } else if (!pisz && schp) { /* we need the whole pressure mass matrix to define the interface BDDC */
          IS  P;

          ierr = PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_aP",(PetscObject*)&P);CHKERRQ(ierr);
          ierr = MatCreateSubMatrix(A,P,P,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
          ierr = MatScale(C,-1.);CHKERRQ(ierr);
          ierr = KSPFETIDPSetPressureOperator(ksp,C);CHKERRQ(ierr);
          ierr = MatDestroy(&C);CHKERRQ(ierr);
        } else { /* identity (need to be scaled properly by the user using e.g. a Richardson method */
          PetscInt nl;

          ierr = ISGetLocalSize(fetidp->pP,&nl);CHKERRQ(ierr);
          ierr = MatCreate(PetscObjectComm((PetscObject)ksp),&C);CHKERRQ(ierr);
          ierr = MatSetSizes(C,nl,nl,totP,totP);CHKERRQ(ierr);
          ierr = MatSetType(C,MATAIJ);CHKERRQ(ierr);
          ierr = MatMPIAIJSetPreallocation(C,1,NULL,0,NULL);CHKERRQ(ierr);
          ierr = MatSeqAIJSetPreallocation(C,1,NULL);CHKERRQ(ierr);
          ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
          ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
          ierr = MatShift(C,1.);CHKERRQ(ierr);
          ierr = KSPFETIDPSetPressureOperator(ksp,C);CHKERRQ(ierr);
          ierr = MatDestroy(&C);CHKERRQ(ierr);
        }
      }

      /* Preconditioned operator for the pressure block */
      ierr = PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_PPmat",(PetscObject*)&PPmat);CHKERRQ(ierr);
      if (PPmat) {
        Mat      C;
        IS       Pall;
        PetscInt AM,PAM,PAN,pam,pan,am,an,pl,pIl,pAg,pIg;

        ierr = PetscObjectQuery((PetscObject)fetidp->innerbddc,"__KSPFETIDP_aP",(PetscObject*)&Pall);CHKERRQ(ierr);
        ierr = MatGetSize(A,&AM,NULL);CHKERRQ(ierr);
        ierr = MatGetSize(PPmat,&PAM,&PAN);CHKERRQ(ierr);
        ierr = ISGetSize(Pall,&pAg);CHKERRQ(ierr);
        ierr = ISGetSize(fetidp->pP,&pIg);CHKERRQ(ierr);
        ierr = MatGetLocalSize(PPmat,&pam,&pan);CHKERRQ(ierr);
        ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
        ierr = ISGetLocalSize(Pall,&pIl);CHKERRQ(ierr);
        ierr = ISGetLocalSize(fetidp->pP,&pl);CHKERRQ(ierr);
        if (PAM != PAN) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Pressure matrix must be square, unsupported %D x %D",PAM,PAN);
        if (pam != pan) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Local sizes of pressure matrix must be equal, unsupported %D x %D",pam,pan);
        if (pam != am && pam != pl && pam != pIl) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local rows %D for pressure matrix! Supported are %D, %D or %D",pam,am,pl,pIl);
        if (pan != an && pan != pl && pan != pIl) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local columns %D for pressure matrix! Supported are %D, %D or %D",pan,an,pl,pIl);
        if (PAM == AM) { /* monolithic ordering, restrict to pressure */
          if (schp) {
            ierr = MatCreateSubMatrix(PPmat,Pall,Pall,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
          } else {
            ierr = MatCreateSubMatrix(PPmat,fetidp->pP,fetidp->pP,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
          }
        } else if (pAg == PAM) { /* global ordering for pressure only */
          if (!allp && !schp) { /* solving for interface pressure only */
            IS restr;

            ierr = ISRenumber(fetidp->pP,NULL,NULL,&restr);CHKERRQ(ierr);
            ierr = MatCreateSubMatrix(PPmat,restr,restr,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
            ierr = ISDestroy(&restr);CHKERRQ(ierr);
          } else {
            ierr = PetscObjectReference((PetscObject)PPmat);CHKERRQ(ierr);
            C    = PPmat;
          }
        } else if (pIg == PAM) { /* global ordering for selected pressure only */
          if (schp) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Need the entire matrix");
          ierr = PetscObjectReference((PetscObject)PPmat);CHKERRQ(ierr);
          C    = PPmat;
        } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Unable to use the pressure matrix");

        ierr = KSPFETIDPSetPressureOperator(ksp,C);CHKERRQ(ierr);
        ierr = MatDestroy(&C);CHKERRQ(ierr);
      } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Missing Pmat for pressure block");
    } else { /* totP == 0 */
      ierr = PetscObjectCompose((PetscObject)fetidp->innerbddc,"__KSPFETIDP_pP",NULL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPFETIDPSetUpOperators(ksp);CHKERRQ(ierr);
  /* set up BDDC */
  ierr = PCSetErrorIfFailure(fetidp->innerbddc,ksp->errorifnotconverged);CHKERRQ(ierr);
  ierr = PCSetUp(fetidp->innerbddc);CHKERRQ(ierr);
  /* FETI-DP as it is implemented needs an exact coarse solver */
  if (pcbddc->coarse_ksp) {
    ierr = KSPSetTolerances(pcbddc->coarse_ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,1000);CHKERRQ(ierr);
    ierr = KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_DEFAULT);CHKERRQ(ierr);
  }
  /* FETI-DP as it is implemented needs exact local Neumann solvers */
  ierr = KSPSetTolerances(pcbddc->ksp_R,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,1000);CHKERRQ(ierr);
  ierr = KSPSetNormType(pcbddc->ksp_R,KSP_NORM_DEFAULT);CHKERRQ(ierr);

  /* setup FETI-DP operators
     If fetidp->statechanged is true, we need to update the operators
     needed in the saddle-point case. This should be replaced
     by a better logic when the FETI-DP matrix and preconditioner will
     have their own classes */
  if (pcbddc->new_primal_space || fetidp->statechanged) {
    Mat F; /* the FETI-DP matrix */
    PC  D; /* the FETI-DP preconditioner */
    ierr = KSPReset(fetidp->innerksp);CHKERRQ(ierr);
    ierr = PCBDDCCreateFETIDPOperators(fetidp->innerbddc,fetidp->fully_redundant,((PetscObject)ksp)->prefix,&F,&D);CHKERRQ(ierr);
    ierr = KSPSetOperators(fetidp->innerksp,F,F);CHKERRQ(ierr);
    ierr = KSPSetTolerances(fetidp->innerksp,ksp->rtol,ksp->abstol,ksp->divtol,ksp->max_it);CHKERRQ(ierr);
    ierr = KSPSetPC(fetidp->innerksp,D);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)D,(PetscObject)fetidp->innerksp,0);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(fetidp->innerksp);CHKERRQ(ierr);
    ierr = MatCreateVecs(F,&(fetidp->innerksp)->vec_rhs,&(fetidp->innerksp)->vec_sol);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = PCDestroy(&D);CHKERRQ(ierr);
    if (fetidp->check) {
      PetscViewer viewer;

      if (!pcbddc->dbg_viewer) {
        viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
      } else {
        viewer = pcbddc->dbg_viewer;
      }
      ierr = KSPFETIDPCheckOperators(ksp,viewer);CHKERRQ(ierr);
    }
  }
  fetidp->statechanged     = PETSC_FALSE;
  pcbddc->new_primal_space = PETSC_FALSE;

  /* propagate settings to the inner solve */
  ierr = KSPGetComputeSingularValues(ksp,&flg);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues(fetidp->innerksp,flg);CHKERRQ(ierr);
  if (ksp->res_hist) {
    ierr = KSPSetResidualHistory(fetidp->innerksp,ksp->res_hist,ksp->res_hist_max,ksp->res_hist_reset);CHKERRQ(ierr);
  }
  ierr = KSPSetErrorIfNotConverged(fetidp->innerksp,ksp->errorifnotconverged);CHKERRQ(ierr);
  ierr = KSPSetUp(fetidp->innerksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_FETIDP(KSP ksp)
{
  PetscErrorCode     ierr;
  Mat                F,A;
  MatNullSpace       nsp;
  Vec                X,B,Xl,Bl;
  KSP_FETIDP         *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC            *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  KSPConvergedReason reason;
  PC                 pc;
  PCFailedReason     pcreason;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  if (fetidp->saddlepoint) {
    ierr = PetscCitationsRegister(citation2,&cited2);CHKERRQ(ierr);
  }
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&B);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&X);CHKERRQ(ierr);
  ierr = KSPGetOperators(fetidp->innerksp,&F,NULL);CHKERRQ(ierr);
  ierr = KSPGetRhs(fetidp->innerksp,&Bl);CHKERRQ(ierr);
  ierr = KSPGetSolution(fetidp->innerksp,&Xl);CHKERRQ(ierr);
  ierr = PCBDDCMatFETIDPGetRHS(F,B,Bl);CHKERRQ(ierr);
  if (ksp->transpose_solve) {
    ierr = KSPSolveTranspose(fetidp->innerksp,Bl,Xl);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(fetidp->innerksp,Bl,Xl);CHKERRQ(ierr);
  }
  ierr = KSPGetConvergedReason(fetidp->innerksp,&reason);CHKERRQ(ierr);
  ierr = KSPGetPC(fetidp->innerksp,&pc);CHKERRQ(ierr);
  ierr = PCGetFailedReason(pc,&pcreason);CHKERRQ(ierr);
  if ((reason < 0 && reason != KSP_DIVERGED_ITS) || pcreason) {
    PetscInt its;
    ierr = KSPGetIterationNumber(fetidp->innerksp,&its);CHKERRQ(ierr);
    ksp->reason = KSP_DIVERGED_PC_FAILED;
    ierr = VecSetInf(Xl);CHKERRQ(ierr);
    ierr = PetscInfo3(ksp,"Inner KSP solve failed: %s %s at iteration %D",KSPConvergedReasons[reason],PCFailedReasons[pcreason],its);CHKERRQ(ierr);
  }
  ierr = PCBDDCMatFETIDPGetSolution(F,Xl,X);CHKERRQ(ierr);
  ierr = MatGetNullSpace(A,&nsp);CHKERRQ(ierr);
  if (nsp) {
    ierr = MatNullSpaceRemove(nsp,X);CHKERRQ(ierr);
  }
  /* update ksp with stats from inner ksp */
  ierr = KSPGetConvergedReason(fetidp->innerksp,&ksp->reason);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(fetidp->innerksp,&ksp->its);CHKERRQ(ierr);
  ksp->totalits += ksp->its;
  ierr = KSPGetResidualHistory(fetidp->innerksp,NULL,&ksp->res_hist_len);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&fetidp->pP);CHKERRQ(ierr);
  ierr = VecDestroy(&fetidp->rhs_flip);CHKERRQ(ierr);
  /* avoid PCReset that does not take into account ref counting */
  ierr = PCDestroy(&fetidp->innerbddc);CHKERRQ(ierr);
  ierr = PCCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerbddc);CHKERRQ(ierr);
  ierr = PCSetType(fetidp->innerbddc,PCBDDC);CHKERRQ(ierr);
  pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  pcbddc->symmetric_primal = PETSC_FALSE;
  ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerbddc);CHKERRQ(ierr);
  ierr = KSPDestroy(&fetidp->innerksp);CHKERRQ(ierr);
  fetidp->saddlepoint  = PETSC_FALSE;
  fetidp->matstate     = -1;
  fetidp->matnnzstate  = -1;
  fetidp->statechanged = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_FETIDP(ksp);CHKERRQ(ierr);
  ierr = PCDestroy(&fetidp->innerbddc);CHKERRQ(ierr);
  ierr = KSPDestroy(&fetidp->innerksp);CHKERRQ(ierr);
  ierr = PetscFree(fetidp->monctx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetInnerBDDC_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerBDDC_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetPressureOperator_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_FETIDP(KSP ksp,PetscViewer viewer)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  fully redundant: %d\n",fetidp->fully_redundant);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  saddle point:    %d\n",fetidp->saddlepoint);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Inner KSP solver details\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = KSPView(fetidp->innerksp,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Inner BDDC solver details\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PCView(fetidp->innerbddc,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_FETIDP(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* set options prefixes for the inner objects, since the parent prefix will be valid at this point */
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerksp,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerksp,"fetidp_");CHKERRQ(ierr);
  if (!fetidp->userbddc) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerbddc,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
    ierr = PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerbddc,"fetidp_bddc_");CHKERRQ(ierr);
  }
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP FETIDP options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_fetidp_fullyredundant","Use fully redundant multipliers","none",fetidp->fully_redundant,&fetidp->fully_redundant,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_fetidp_saddlepoint","Activates support for saddle-point problems",NULL,fetidp->saddlepoint,&fetidp->saddlepoint,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_fetidp_check","Activates verbose debugging output FETI-DP operators",NULL,fetidp->check,&fetidp->check,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PCSetFromOptions(fetidp->innerbddc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPFETIDP - The FETI-DP method

   This class implements the FETI-DP method [1].
   The matrix for the KSP must be of type MATIS.
   The FETI-DP linear system (automatically generated constructing an internal PCBDDC object) is solved using an internal KSP object.

   Options Database Keys:
+   -ksp_fetidp_fullyredundant <false>   : use a fully redundant set of Lagrange multipliers
.   -ksp_fetidp_saddlepoint <false>      : activates support for saddle point problems, see [2]
.   -ksp_fetidp_saddlepoint_flip <false> : usually, an incompressible Stokes problem is written as
                                           | A B^T | | v | = | f |
                                           | B 0   | | p | = | g |
                                           with B representing -\int_\Omega \nabla \cdot u q.
                                           If -ksp_fetidp_saddlepoint_flip is true, the code assumes that the user provides it as
                                           | A B^T | | v | = | f |
                                           |-B 0   | | p | = |-g |
.   -ksp_fetidp_pressure_field <-1>      : activates support for saddle point problems, and identifies the pressure field id.
                                           If this information is not provided, the pressure field is detected by using MatFindZeroDiagonals().
-   -ksp_fetidp_pressure_all <false>     : if false, uses the interface pressures, as described in [2]. If true, uses the entire pressure field.

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
.vb
.  [1] - C. Farhat, M. Lesoinne, P. LeTallec, K. Pierson, and D. Rixen, FETI-DP: a dual-primal unified FETI method. I. A faster alternative to the two-level FETI method, Internat. J. Numer. Methods Engrg., 50 (2001), pp. 1523--1544
.  [2] - X. Tu, J. Li, A FETI-DP type domain decomposition algorithm for three-dimensional incompressible Stokes equations, SIAM J. Numer. Anal., 53 (2015), pp. 720-742
.ve

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC(), KSPFETIDPGetInnerBDDC(), KSPFETIDPGetInnerKSP()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_FETIDP(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_FETIDP     *fetidp;
  KSP_FETIDPMon  *monctx;
  PC_BDDC        *pcbddc;
  PC             pc;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);

  ierr = PetscNewLog(ksp,&fetidp);CHKERRQ(ierr);
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
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  /* create the inner KSP for the Lagrange multipliers */
  ierr = KSPCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerksp);CHKERRQ(ierr);
  ierr = KSPGetPC(fetidp->innerksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerksp);CHKERRQ(ierr);
  /* monitor */
  ierr = PetscNew(&monctx);CHKERRQ(ierr);
  monctx->parentksp = ksp;
  fetidp->monctx = monctx;
  ierr = KSPMonitorSet(fetidp->innerksp,KSPMonitor_FETIDP,fetidp->monctx,NULL);CHKERRQ(ierr);
  /* create the inner BDDC */
  ierr = PCCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerbddc);CHKERRQ(ierr);
  ierr = PCSetType(fetidp->innerbddc,PCBDDC);CHKERRQ(ierr);
  /* make sure we always obtain a consistent FETI-DP matrix
     for symmetric problems, the user can always customize it through the command line */
  pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  pcbddc->symmetric_primal = PETSC_FALSE;
  ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerbddc);CHKERRQ(ierr);
  /* composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetInnerBDDC_C",KSPFETIDPSetInnerBDDC_FETIDP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerBDDC_C",KSPFETIDPGetInnerBDDC_FETIDP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerKSP_C",KSPFETIDPGetInnerKSP_FETIDP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetPressureOperator_C",KSPFETIDPSetPressureOperator_FETIDP);CHKERRQ(ierr);
  /* need to call KSPSetUp_FETIDP even with KSP_SETUP_NEWMATRIX */
  ksp->setupnewmatrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}

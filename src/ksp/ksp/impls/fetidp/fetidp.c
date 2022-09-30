#include <petsc/private/kspimpl.h> /*I <petscksp.h> I*/
#include <petsc/private/pcbddcimpl.h>
#include <petsc/private/pcbddcprivateimpl.h>
#include <petscdm.h>

static PetscBool  cited       = PETSC_FALSE;
static PetscBool  cited2      = PETSC_FALSE;
static const char citation[]  = "@article{ZampiniPCBDDC,\n"
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
static const char citation2[] = "@article{li2013nonoverlapping,\n"
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
  KSP              innerksp;        /* the KSP for the Lagrange multipliers */
  PC               innerbddc;       /* the inner BDDC object */
  PetscBool        fully_redundant; /* true for using a fully redundant set of multipliers */
  PetscBool        userbddc;        /* true if the user provided the PCBDDC object */
  PetscBool        saddlepoint;     /* support for saddle point problems */
  IS               pP;              /* index set for pressure variables */
  Vec              rhs_flip;        /* see KSPFETIDPSetUpOperators */
  KSP_FETIDPMon   *monctx;          /* monitor context, used to pass user defined monitors
                                        in the physical space */
  PetscObjectState matstate;        /* these are needed just in the saddle point case */
  PetscObjectState matnnzstate;     /* where we are going to use MatZeroRows on pmat */
  PetscBool        statechanged;
  PetscBool        check;
} KSP_FETIDP;

static PetscErrorCode KSPFETIDPSetPressureOperator_FETIDP(KSP ksp, Mat P)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;

  PetscFunctionBegin;
  if (P) fetidp->saddlepoint = PETSC_TRUE;
  PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_PPmat", (PetscObject)P));
  PetscFunctionReturn(0);
}

/*@
 KSPFETIDPSetPressureOperator - Sets the operator used to setup the pressure preconditioner for the saddle point `KSPFETIDP` solver,

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

.seealso: [](chapter_ksp), `KSPFETIDP`, `MATIS`, `PCBDDC`, `KSPFETIDPGetInnerBDDC()`, `KSPFETIDPGetInnerKSP()`, `KSPSetOperators()`
@*/
PetscErrorCode KSPFETIDPSetPressureOperator(KSP ksp, Mat P)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (P) PetscValidHeaderSpecific(P, MAT_CLASSID, 2);
  PetscTryMethod(ksp, "KSPFETIDPSetPressureOperator_C", (KSP, Mat), (ksp, P));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPGetInnerKSP_FETIDP(KSP ksp, KSP *innerksp)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;

  PetscFunctionBegin;
  *innerksp = fetidp->innerksp;
  PetscFunctionReturn(0);
}

/*@
 KSPFETIDPGetInnerKSP - Gets the `KSP` object for the Lagrange multipliers from inside a `KSPFETIDP`

   Input Parameters:
+  ksp - the `KSPFETIDP`
-  innerksp - the `KSP` for the multipliers

   Level: advanced

.seealso: [](chapter_ksp), `KSPFETIDP`, `MATIS`, `PCBDDC`, `KSPFETIDPSetInnerBDDC()`, `KSPFETIDPGetInnerBDDC()`
@*/
PetscErrorCode KSPFETIDPGetInnerKSP(KSP ksp, KSP *innerksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(innerksp, 2);
  PetscUseMethod(ksp, "KSPFETIDPGetInnerKSP_C", (KSP, KSP *), (ksp, innerksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPGetInnerBDDC_FETIDP(KSP ksp, PC *pc)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;

  PetscFunctionBegin;
  *pc = fetidp->innerbddc;
  PetscFunctionReturn(0);
}

/*@
  KSPFETIDPGetInnerBDDC - Gets the `PCBDDC` preconditioner used to setup the `KSPFETIDP` matrix for the Lagrange multipliers

   Input Parameters:
+  ksp - the `KSPFETIDP` Krylov solver
-  pc - the `PCBDDC` preconditioner

   Level: advanced

.seealso: [](chapter_ksp), `MATIS`, `PCBDDC`, `KSPFETIDPSetInnerBDDC()`, `KSPFETIDPGetInnerKSP()`
@*/
PetscErrorCode KSPFETIDPGetInnerBDDC(KSP ksp, PC *pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(pc, 2);
  PetscUseMethod(ksp, "KSPFETIDPGetInnerBDDC_C", (KSP, PC *), (ksp, pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPSetInnerBDDC_FETIDP(KSP ksp, PC pc)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)pc));
  PetscCall(PCDestroy(&fetidp->innerbddc));
  fetidp->innerbddc = pc;
  fetidp->userbddc  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  KSPFETIDPSetInnerBDDC - Provides the `PCBDDC` preconditioner used to setup the `KSPFETIDP` matrix for the Lagrange multipliers

   Collective on ksp

   Input Parameters:
+  ksp - the `KSPFETIDP` Krylov solver
-  pc - the `PCBDDC` preconditioner

   Level: advanced

   Note:
   A `PC` is automatically created for the `KSPFETIDP` and can be accessed to change options with  `KSPFETIDPGetInnerBDDC()` hence this routine is rarely needed

.seealso: [](chapter_ksp), `MATIS`, `PCBDDC`, `KSPFETIDPGetInnerBDDC()`, `KSPFETIDPGetInnerKSP()`
@*/
PetscErrorCode KSPFETIDPSetInnerBDDC(KSP ksp, PC pc)
{
  PetscBool isbddc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(pc, PC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBDDC, &isbddc));
  PetscCheck(isbddc, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_WRONG, "KSPFETIDPSetInnerBDDC need a PCBDDC preconditioner");
  PetscTryMethod(ksp, "KSPFETIDPSetInnerBDDC_C", (KSP, PC), (ksp, pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPBuildSolution_FETIDP(KSP ksp, Vec v, Vec *V)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;
  Mat         F;
  Vec         Xl;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(fetidp->innerksp, &F, NULL));
  PetscCall(KSPBuildSolution(fetidp->innerksp, NULL, &Xl));
  if (v) {
    PetscCall(PCBDDCMatFETIDPGetSolution(F, Xl, v));
    *V = v;
  } else {
    PetscCall(PCBDDCMatFETIDPGetSolution(F, Xl, *V));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMonitor_FETIDP(KSP ksp, PetscInt it, PetscReal rnorm, void *ctx)
{
  KSP_FETIDPMon *monctx = (KSP_FETIDPMon *)ctx;

  PetscFunctionBegin;
  PetscCall(KSPMonitor(monctx->parentksp, it, rnorm));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPComputeEigenvalues_FETIDP(KSP ksp, PetscInt nmax, PetscReal *r, PetscReal *c, PetscInt *neig)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPComputeEigenvalues(fetidp->innerksp, nmax, r, c, neig));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPComputeExtremeSingularValues_FETIDP(KSP ksp, PetscReal *emax, PetscReal *emin)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPComputeExtremeSingularValues(fetidp->innerksp, emax, emin));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPCheckOperators(KSP ksp, PetscViewer viewer)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP *)ksp->data;
  PC_BDDC        *pcbddc = (PC_BDDC *)fetidp->innerbddc->data;
  PC_IS          *pcis   = (PC_IS *)fetidp->innerbddc->data;
  Mat_IS         *matis  = (Mat_IS *)fetidp->innerbddc->pmat->data;
  Mat             F;
  FETIDPMat_ctx   fetidpmat_ctx;
  Vec             test_vec, test_vec_p = NULL, fetidp_global;
  IS              dirdofs, isvert;
  MPI_Comm        comm = PetscObjectComm((PetscObject)ksp);
  PetscScalar     sval, *array;
  PetscReal       val, rval;
  const PetscInt *vertex_indices;
  PetscInt        i, n_vertices;
  PetscBool       isascii;

  PetscFunctionBegin;
  PetscCheckSameComm(ksp, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(isascii, comm, PETSC_ERR_SUP, "Unsupported viewer");
  PetscCall(PetscViewerASCIIPrintf(viewer, "----------FETI-DP MAT  --------------\n"));
  PetscCall(PetscViewerASCIIAddTab(viewer, 2));
  PetscCall(KSPGetOperators(fetidp->innerksp, &F, NULL));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
  PetscCall(MatView(F, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerASCIISubtractTab(viewer, 2));
  PetscCall(MatShellGetContext(F, &fetidpmat_ctx));
  PetscCall(PetscViewerASCIIPrintf(viewer, "----------FETI-DP TESTS--------------\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "All tests should return zero!\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "FETIDP MAT context in the "));
  if (fetidp->fully_redundant) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "fully redundant case for lagrange multipliers.\n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Non-fully redundant case for lagrange multiplier.\n"));
  }
  PetscCall(PetscViewerFlush(viewer));

  /* Get Vertices used to define the BDDC */
  PetscCall(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph, NULL, NULL, NULL, NULL, &isvert));
  PetscCall(ISGetLocalSize(isvert, &n_vertices));
  PetscCall(ISGetIndices(isvert, &vertex_indices));

  /******************************************************************/
  /* TEST A/B: Test numbering of global fetidp dofs                 */
  /******************************************************************/
  PetscCall(MatCreateVecs(F, &fetidp_global, NULL));
  PetscCall(VecDuplicate(fetidpmat_ctx->lambda_local, &test_vec));
  PetscCall(VecSet(fetidp_global, 1.0));
  PetscCall(VecSet(test_vec, 1.));
  PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidp_global, fetidpmat_ctx->lambda_local, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidp_global, fetidpmat_ctx->lambda_local, INSERT_VALUES, SCATTER_REVERSE));
  if (fetidpmat_ctx->l2g_p) {
    PetscCall(VecDuplicate(fetidpmat_ctx->vP, &test_vec_p));
    PetscCall(VecSet(test_vec_p, 1.));
    PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_p, fetidp_global, fetidpmat_ctx->vP, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_p, fetidp_global, fetidpmat_ctx->vP, INSERT_VALUES, SCATTER_REVERSE));
  }
  PetscCall(VecAXPY(test_vec, -1.0, fetidpmat_ctx->lambda_local));
  PetscCall(VecNorm(test_vec, NORM_INFINITY, &val));
  PetscCall(VecDestroy(&test_vec));
  PetscCallMPI(MPI_Reduce(&val, &rval, 1, MPIU_REAL, MPIU_MAX, 0, comm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "A: CHECK glob to loc: % 1.14e\n", (double)rval));

  if (fetidpmat_ctx->l2g_p) {
    PetscCall(VecAXPY(test_vec_p, -1.0, fetidpmat_ctx->vP));
    PetscCall(VecNorm(test_vec_p, NORM_INFINITY, &val));
    PetscCallMPI(MPI_Reduce(&val, &rval, 1, MPIU_REAL, MPIU_MAX, 0, comm));
    PetscCall(PetscViewerASCIIPrintf(viewer, "A: CHECK glob to loc (p): % 1.14e\n", (double)rval));
  }

  if (fetidp->fully_redundant) {
    PetscCall(VecSet(fetidp_global, 0.0));
    PetscCall(VecSet(fetidpmat_ctx->lambda_local, 0.5));
    PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
    PetscCall(VecSum(fetidp_global, &sval));
    val = PetscRealPart(sval) - fetidpmat_ctx->n_lambda;
    PetscCallMPI(MPI_Reduce(&val, &rval, 1, MPIU_REAL, MPIU_MAX, 0, comm));
    PetscCall(PetscViewerASCIIPrintf(viewer, "B: CHECK loc to glob: % 1.14e\n", (double)rval));
  }

  if (fetidpmat_ctx->l2g_p) {
    PetscCall(VecSet(pcis->vec1_N, 1.0));
    PetscCall(VecSet(pcis->vec1_global, 0.0));
    PetscCall(VecScatterBegin(matis->rctx, pcis->vec1_N, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(matis->rctx, pcis->vec1_N, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));

    PetscCall(VecSet(fetidp_global, 0.0));
    PetscCall(VecSet(fetidpmat_ctx->vP, -1.0));
    PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_p, fetidpmat_ctx->vP, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_p, fetidpmat_ctx->vP, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(fetidpmat_ctx->g2g_p, fetidp_global, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(fetidpmat_ctx->g2g_p, fetidp_global, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterBegin(fetidpmat_ctx->g2g_p, pcis->vec1_global, fetidp_global, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(fetidpmat_ctx->g2g_p, pcis->vec1_global, fetidp_global, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecSum(fetidp_global, &sval));
    val = PetscRealPart(sval);
    PetscCallMPI(MPI_Reduce(&val, &rval, 1, MPIU_REAL, MPIU_MAX, 0, comm));
    PetscCall(PetscViewerASCIIPrintf(viewer, "B: CHECK loc to glob (p): % 1.14e\n", (double)rval));
  }

  /******************************************************************/
  /* TEST C: It should hold B_delta*w=0, w\in\widehat{W}            */
  /* This is the meaning of the B matrix                            */
  /******************************************************************/

  PetscCall(VecSetRandom(pcis->vec1_N, NULL));
  PetscCall(VecSet(pcis->vec1_global, 0.0));
  PetscCall(VecScatterBegin(matis->rctx, pcis->vec1_N, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(matis->rctx, pcis->vec1_N, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(matis->rctx, pcis->vec1_global, pcis->vec1_N, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(matis->rctx, pcis->vec1_global, pcis->vec1_N, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(pcis->N_to_B, pcis->vec1_N, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->N_to_B, pcis->vec1_N, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
  /* Action of B_delta */
  PetscCall(MatMult(fetidpmat_ctx->B_delta, pcis->vec1_B, fetidpmat_ctx->lambda_local));
  PetscCall(VecSet(fetidp_global, 0.0));
  PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecNorm(fetidp_global, NORM_INFINITY, &val));
  PetscCall(PetscViewerASCIIPrintf(viewer, "C: CHECK infty norm of B_delta*w (w continuous): % 1.14e\n", (double)val));

  /******************************************************************/
  /* TEST D: It should hold E_Dw = w - P_Dw w\in\widetilde{W}       */
  /* E_D = R_D^TR                                                   */
  /* P_D = B_{D,delta}^T B_{delta}                                  */
  /* eq.44 Mandel Tezaur and Dohrmann 2005                          */
  /******************************************************************/

  /* compute a random vector in \widetilde{W} */
  PetscCall(VecSetRandom(pcis->vec1_N, NULL));
  /* set zero at vertices and essential dofs */
  PetscCall(VecGetArray(pcis->vec1_N, &array));
  for (i = 0; i < n_vertices; i++) array[vertex_indices[i]] = 0.0;
  PetscCall(PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph, &dirdofs));
  if (dirdofs) {
    const PetscInt *idxs;
    PetscInt        ndir;

    PetscCall(ISGetLocalSize(dirdofs, &ndir));
    PetscCall(ISGetIndices(dirdofs, &idxs));
    for (i = 0; i < ndir; i++) array[idxs[i]] = 0.0;
    PetscCall(ISRestoreIndices(dirdofs, &idxs));
  }
  PetscCall(VecRestoreArray(pcis->vec1_N, &array));
  /* store w for final comparison */
  PetscCall(VecDuplicate(pcis->vec1_B, &test_vec));
  PetscCall(VecScatterBegin(pcis->N_to_B, pcis->vec1_N, test_vec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->N_to_B, pcis->vec1_N, test_vec, INSERT_VALUES, SCATTER_FORWARD));

  /* Jump operator P_D : results stored in pcis->vec1_B */
  /* Action of B_delta */
  PetscCall(MatMult(fetidpmat_ctx->B_delta, test_vec, fetidpmat_ctx->lambda_local));
  PetscCall(VecSet(fetidp_global, 0.0));
  PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
  /* Action of B_Ddelta^T */
  PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidp_global, fetidpmat_ctx->lambda_local, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidp_global, fetidpmat_ctx->lambda_local, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(MatMultTranspose(fetidpmat_ctx->B_Ddelta, fetidpmat_ctx->lambda_local, pcis->vec1_B));

  /* Average operator E_D : results stored in pcis->vec2_B */
  PetscCall(PCBDDCScalingExtension(fetidpmat_ctx->pc, test_vec, pcis->vec1_global));
  PetscCall(VecScatterBegin(pcis->global_to_B, pcis->vec1_global, pcis->vec2_B, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_B, pcis->vec1_global, pcis->vec2_B, INSERT_VALUES, SCATTER_FORWARD));

  /* test E_D=I-P_D */
  PetscCall(VecAXPY(pcis->vec1_B, 1.0, pcis->vec2_B));
  PetscCall(VecAXPY(pcis->vec1_B, -1.0, test_vec));
  PetscCall(VecNorm(pcis->vec1_B, NORM_INFINITY, &val));
  PetscCall(VecDestroy(&test_vec));
  PetscCallMPI(MPI_Reduce(&val, &rval, 1, MPIU_REAL, MPIU_MAX, 0, comm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%d: CHECK infty norm of E_D + P_D - I: %1.14e\n", PetscGlobalRank, (double)val));

  /******************************************************************/
  /* TEST E: It should hold R_D^TP_Dw=0 w\in\widetilde{W}           */
  /* eq.48 Mandel Tezaur and Dohrmann 2005                          */
  /******************************************************************/

  PetscCall(VecSetRandom(pcis->vec1_N, NULL));
  /* set zero at vertices and essential dofs */
  PetscCall(VecGetArray(pcis->vec1_N, &array));
  for (i = 0; i < n_vertices; i++) array[vertex_indices[i]] = 0.0;
  if (dirdofs) {
    const PetscInt *idxs;
    PetscInt        ndir;

    PetscCall(ISGetLocalSize(dirdofs, &ndir));
    PetscCall(ISGetIndices(dirdofs, &idxs));
    for (i = 0; i < ndir; i++) array[idxs[i]] = 0.0;
    PetscCall(ISRestoreIndices(dirdofs, &idxs));
  }
  PetscCall(VecRestoreArray(pcis->vec1_N, &array));

  /* Jump operator P_D : results stored in pcis->vec1_B */

  PetscCall(VecScatterBegin(pcis->N_to_B, pcis->vec1_N, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->N_to_B, pcis->vec1_N, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
  /* Action of B_delta */
  PetscCall(MatMult(fetidpmat_ctx->B_delta, pcis->vec1_B, fetidpmat_ctx->lambda_local));
  PetscCall(VecSet(fetidp_global, 0.0));
  PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, fetidp_global, ADD_VALUES, SCATTER_FORWARD));
  /* Action of B_Ddelta^T */
  PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidp_global, fetidpmat_ctx->lambda_local, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidp_global, fetidpmat_ctx->lambda_local, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(MatMultTranspose(fetidpmat_ctx->B_Ddelta, fetidpmat_ctx->lambda_local, pcis->vec1_B));
  /* scaling */
  PetscCall(PCBDDCScalingExtension(fetidpmat_ctx->pc, pcis->vec1_B, pcis->vec1_global));
  PetscCall(VecNorm(pcis->vec1_global, NORM_INFINITY, &val));
  PetscCall(PetscViewerASCIIPrintf(viewer, "E: CHECK infty norm of R^T_D P_D: % 1.14e\n", (double)val));

  if (!fetidp->fully_redundant) {
    /******************************************************************/
    /* TEST F: It should holds B_{delta}B^T_{D,delta}=I               */
    /* Corollary thm 14 Mandel Tezaur and Dohrmann 2005               */
    /******************************************************************/
    PetscCall(VecDuplicate(fetidp_global, &test_vec));
    PetscCall(VecSetRandom(fetidp_global, NULL));
    if (fetidpmat_ctx->l2g_p) {
      PetscCall(VecSet(fetidpmat_ctx->vP, 0.));
      PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_p, fetidpmat_ctx->vP, fetidp_global, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_p, fetidpmat_ctx->vP, fetidp_global, INSERT_VALUES, SCATTER_FORWARD));
    }
    /* Action of B_Ddelta^T */
    PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidp_global, fetidpmat_ctx->lambda_local, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidp_global, fetidpmat_ctx->lambda_local, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(MatMultTranspose(fetidpmat_ctx->B_Ddelta, fetidpmat_ctx->lambda_local, pcis->vec1_B));
    /* Action of B_delta */
    PetscCall(MatMult(fetidpmat_ctx->B_delta, pcis->vec1_B, fetidpmat_ctx->lambda_local));
    PetscCall(VecSet(test_vec, 0.0));
    PetscCall(VecScatterBegin(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, test_vec, ADD_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(fetidpmat_ctx->l2g_lambda, fetidpmat_ctx->lambda_local, test_vec, ADD_VALUES, SCATTER_FORWARD));
    PetscCall(VecAXPY(fetidp_global, -1., test_vec));
    PetscCall(VecNorm(fetidp_global, NORM_INFINITY, &val));
    PetscCall(PetscViewerASCIIPrintf(viewer, "E: CHECK infty norm of P^T_D - I: % 1.14e\n", (double)val));
    PetscCall(VecDestroy(&test_vec));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "-------------------------------------\n"));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(VecDestroy(&test_vec_p));
  PetscCall(ISDestroy(&dirdofs));
  PetscCall(VecDestroy(&fetidp_global));
  PetscCall(ISRestoreIndices(isvert, &vertex_indices));
  PetscCall(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph, NULL, NULL, NULL, NULL, &isvert));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPFETIDPSetUpOperators(KSP ksp)
{
  KSP_FETIDP      *fetidp = (KSP_FETIDP *)ksp->data;
  PC_BDDC         *pcbddc = (PC_BDDC *)fetidp->innerbddc->data;
  Mat              A, Ap;
  PetscInt         fid = -1;
  PetscMPIInt      size;
  PetscBool        ismatis, pisz = PETSC_FALSE, allp = PETSC_FALSE, schp = PETSC_FALSE;
  PetscBool        flip = PETSC_FALSE; /* Usually, Stokes is written (B = -\int_\Omega \nabla \cdot u q)
                           | A B'| | v | = | f |
                           | B 0 | | p | = | g |
                            If -ksp_fetidp_saddlepoint_flip is true, the code assumes it is written as
                           | A B'| | v | = | f |
                           |-B 0 | | p | = |-g |
                         */
  PetscObjectState matstate, matnnzstate;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)ksp), ((PetscObject)ksp)->prefix, "FETI-DP options", "PC");
  PetscCall(PetscOptionsInt("-ksp_fetidp_pressure_field", "Field id for pressures for saddle-point problems", NULL, fid, &fid, NULL));
  PetscCall(PetscOptionsBool("-ksp_fetidp_pressure_all", "Use the whole pressure set instead of just that at the interface", NULL, allp, &allp, NULL));
  PetscCall(PetscOptionsBool("-ksp_fetidp_saddlepoint_flip", "Flip the sign of the pressure-velocity (lower-left) block", NULL, flip, &flip, NULL));
  PetscCall(PetscOptionsBool("-ksp_fetidp_pressure_schur", "Use a BDDC solver for pressure", NULL, schp, &schp, NULL));
  PetscOptionsEnd();

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ksp), &size));
  fetidp->saddlepoint = (fid >= 0 ? PETSC_TRUE : fetidp->saddlepoint);
  if (size == 1) fetidp->saddlepoint = PETSC_FALSE;

  PetscCall(KSPGetOperators(ksp, &A, &Ap));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  PetscCheck(ismatis, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Amat should be of type MATIS");

  /* Quiet return if the matrix states are unchanged.
     Needed only for the saddle point case since it uses MatZeroRows
     on a matrix that may not have changed */
  PetscCall(PetscObjectStateGet((PetscObject)A, &matstate));
  PetscCall(MatGetNonzeroState(A, &matnnzstate));
  if (matstate == fetidp->matstate && matnnzstate == fetidp->matnnzstate) PetscFunctionReturn(0);
  fetidp->matstate     = matstate;
  fetidp->matnnzstate  = matnnzstate;
  fetidp->statechanged = fetidp->saddlepoint;

  /* see if we have some fields attached */
  if (!pcbddc->n_ISForDofsLocal && !pcbddc->n_ISForDofs) {
    DM             dm;
    PetscContainer c;

    PetscCall(KSPGetDM(ksp, &dm));
    PetscCall(PetscObjectQuery((PetscObject)A, "_convert_nest_lfields", (PetscObject *)&c));
    if (dm) {
      IS      *fields;
      PetscInt nf, i;

      PetscCall(DMCreateFieldDecomposition(dm, &nf, NULL, &fields, NULL));
      PetscCall(PCBDDCSetDofsSplitting(fetidp->innerbddc, nf, fields));
      for (i = 0; i < nf; i++) PetscCall(ISDestroy(&fields[i]));
      PetscCall(PetscFree(fields));
    } else if (c) {
      MatISLocalFields lf;

      PetscCall(PetscContainerGetPointer(c, (void **)&lf));
      PetscCall(PCBDDCSetDofsSplittingLocal(fetidp->innerbddc, lf->nr, lf->rf));
    }
  }

  if (!fetidp->saddlepoint) {
    PetscCall(PCSetOperators(fetidp->innerbddc, A, A));
  } else {
    Mat          nA, lA, PPmat;
    MatNullSpace nnsp;
    IS           pP;
    PetscInt     totP;

    PetscCall(MatISGetLocalMat(A, &lA));
    PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_lA", (PetscObject)lA));

    pP = fetidp->pP;
    if (!pP) { /* first time, need to compute pressure dofs */
      PC_IS                 *pcis  = (PC_IS *)fetidp->innerbddc->data;
      Mat_IS                *matis = (Mat_IS *)(A->data);
      ISLocalToGlobalMapping l2g;
      IS                     lP = NULL, II, pII, lPall, Pall, is1, is2;
      const PetscInt        *idxs;
      PetscInt               nl, ni, *widxs;
      PetscInt               i, j, n_neigh, *neigh, *n_shared, **shared, *count;
      PetscInt               rst, ren, n;
      PetscBool              ploc;

      PetscCall(MatGetLocalSize(A, &nl, NULL));
      PetscCall(MatGetOwnershipRange(A, &rst, &ren));
      PetscCall(MatGetLocalSize(lA, &n, NULL));
      PetscCall(MatISGetLocalToGlobalMapping(A, &l2g, NULL));

      if (!pcis->is_I_local) { /* need to compute interior dofs */
        PetscCall(PetscCalloc1(n, &count));
        PetscCall(ISLocalToGlobalMappingGetInfo(l2g, &n_neigh, &neigh, &n_shared, &shared));
        for (i = 1; i < n_neigh; i++)
          for (j = 0; j < n_shared[i]; j++) count[shared[i][j]] += 1;
        for (i = 0, j = 0; i < n; i++)
          if (!count[i]) count[j++] = i;
        PetscCall(ISLocalToGlobalMappingRestoreInfo(l2g, &n_neigh, &neigh, &n_shared, &shared));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, j, count, PETSC_OWN_POINTER, &II));
      } else {
        PetscCall(PetscObjectReference((PetscObject)pcis->is_I_local));
        II = pcis->is_I_local;
      }

      /* interior dofs in layout */
      PetscCall(PetscArrayzero(matis->sf_leafdata, n));
      PetscCall(PetscArrayzero(matis->sf_rootdata, nl));
      PetscCall(ISGetLocalSize(II, &ni));
      PetscCall(ISGetIndices(II, &idxs));
      for (i = 0; i < ni; i++) matis->sf_leafdata[idxs[i]] = 1;
      PetscCall(ISRestoreIndices(II, &idxs));
      PetscCall(PetscSFReduceBegin(matis->sf, MPIU_INT, matis->sf_leafdata, matis->sf_rootdata, MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(matis->sf, MPIU_INT, matis->sf_leafdata, matis->sf_rootdata, MPI_REPLACE));
      PetscCall(PetscMalloc1(PetscMax(nl, n), &widxs));
      for (i = 0, ni = 0; i < nl; i++)
        if (matis->sf_rootdata[i]) widxs[ni++] = i + rst;
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)ksp), ni, widxs, PETSC_COPY_VALUES, &pII));

      /* pressure dofs */
      Pall  = NULL;
      lPall = NULL;
      ploc  = PETSC_FALSE;
      if (fid < 0) { /* zero pressure block */
        PetscInt np;

        PetscCall(MatFindZeroDiagonals(A, &Pall));
        PetscCall(ISGetSize(Pall, &np));
        if (!np) { /* zero-block not found, defaults to last field (if set) */
          fid = pcbddc->n_ISForDofsLocal ? pcbddc->n_ISForDofsLocal - 1 : pcbddc->n_ISForDofs - 1;
          PetscCall(ISDestroy(&Pall));
        } else if (!pcbddc->n_ISForDofsLocal && !pcbddc->n_ISForDofs) {
          PetscCall(PCBDDCSetDofsSplitting(fetidp->innerbddc, 1, &Pall));
        }
      }
      if (!Pall) { /* look for registered fields */
        if (pcbddc->n_ISForDofsLocal) {
          PetscInt np;

          PetscCheck(fid >= 0 && fid < pcbddc->n_ISForDofsLocal, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Invalid field id for pressure %" PetscInt_FMT ", max %" PetscInt_FMT, fid, pcbddc->n_ISForDofsLocal);
          /* need a sequential IS */
          PetscCall(ISGetLocalSize(pcbddc->ISForDofsLocal[fid], &np));
          PetscCall(ISGetIndices(pcbddc->ISForDofsLocal[fid], &idxs));
          PetscCall(ISCreateGeneral(PETSC_COMM_SELF, np, idxs, PETSC_COPY_VALUES, &lPall));
          PetscCall(ISRestoreIndices(pcbddc->ISForDofsLocal[fid], &idxs));
          ploc = PETSC_TRUE;
        } else if (pcbddc->n_ISForDofs) {
          PetscCheck(fid >= 0 && fid < pcbddc->n_ISForDofs, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Invalid field id for pressure %" PetscInt_FMT ", max %" PetscInt_FMT, fid, pcbddc->n_ISForDofs);
          PetscCall(PetscObjectReference((PetscObject)pcbddc->ISForDofs[fid]));
          Pall = pcbddc->ISForDofs[fid];
        } else SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Cannot detect pressure field! Use KSPFETIDPGetInnerBDDC() + PCBDDCSetDofsSplitting or PCBDDCSetDofsSplittingLocal");
      }

      /* if the user requested the entire pressure,
         remove the interior pressure dofs from II (or pII) */
      if (allp) {
        if (ploc) {
          IS nII;
          PetscCall(ISDifference(II, lPall, &nII));
          PetscCall(ISDestroy(&II));
          II = nII;
        } else {
          IS nII;
          PetscCall(ISDifference(pII, Pall, &nII));
          PetscCall(ISDestroy(&pII));
          pII = nII;
        }
      }
      if (ploc) {
        PetscCall(ISDifference(lPall, II, &lP));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_lP", (PetscObject)lP));
      } else {
        PetscCall(ISDifference(Pall, pII, &pP));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_pP", (PetscObject)pP));
        /* need all local pressure dofs */
        PetscCall(PetscArrayzero(matis->sf_leafdata, n));
        PetscCall(PetscArrayzero(matis->sf_rootdata, nl));
        PetscCall(ISGetLocalSize(Pall, &ni));
        PetscCall(ISGetIndices(Pall, &idxs));
        for (i = 0; i < ni; i++) matis->sf_rootdata[idxs[i] - rst] = 1;
        PetscCall(ISRestoreIndices(Pall, &idxs));
        PetscCall(PetscSFBcastBegin(matis->sf, MPIU_INT, matis->sf_rootdata, matis->sf_leafdata, MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(matis->sf, MPIU_INT, matis->sf_rootdata, matis->sf_leafdata, MPI_REPLACE));
        for (i = 0, ni = 0; i < n; i++)
          if (matis->sf_leafdata[i]) widxs[ni++] = i;
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ni, widxs, PETSC_COPY_VALUES, &lPall));
      }

      if (!Pall) {
        PetscCall(PetscArrayzero(matis->sf_leafdata, n));
        PetscCall(PetscArrayzero(matis->sf_rootdata, nl));
        PetscCall(ISGetLocalSize(lPall, &ni));
        PetscCall(ISGetIndices(lPall, &idxs));
        for (i = 0; i < ni; i++) matis->sf_leafdata[idxs[i]] = 1;
        PetscCall(ISRestoreIndices(lPall, &idxs));
        PetscCall(PetscSFReduceBegin(matis->sf, MPIU_INT, matis->sf_leafdata, matis->sf_rootdata, MPI_REPLACE));
        PetscCall(PetscSFReduceEnd(matis->sf, MPIU_INT, matis->sf_leafdata, matis->sf_rootdata, MPI_REPLACE));
        for (i = 0, ni = 0; i < nl; i++)
          if (matis->sf_rootdata[i]) widxs[ni++] = i + rst;
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)ksp), ni, widxs, PETSC_COPY_VALUES, &Pall));
      }
      PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_aP", (PetscObject)Pall));

      if (flip) {
        PetscInt npl;
        PetscCall(ISGetLocalSize(Pall, &npl));
        PetscCall(ISGetIndices(Pall, &idxs));
        PetscCall(MatCreateVecs(A, NULL, &fetidp->rhs_flip));
        PetscCall(VecSet(fetidp->rhs_flip, 1.));
        PetscCall(VecSetOption(fetidp->rhs_flip, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
        for (i = 0; i < npl; i++) PetscCall(VecSetValue(fetidp->rhs_flip, idxs[i], -1., INSERT_VALUES));
        PetscCall(VecAssemblyBegin(fetidp->rhs_flip));
        PetscCall(VecAssemblyEnd(fetidp->rhs_flip));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_flip", (PetscObject)fetidp->rhs_flip));
        PetscCall(ISRestoreIndices(Pall, &idxs));
      }
      PetscCall(ISDestroy(&Pall));
      PetscCall(ISDestroy(&pII));

      /* local selected pressures in subdomain-wise and global ordering */
      PetscCall(PetscArrayzero(matis->sf_leafdata, n));
      PetscCall(PetscArrayzero(matis->sf_rootdata, nl));
      if (!ploc) {
        PetscInt *widxs2;

        PetscCheck(pP, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB, "Missing parallel pressure IS");
        PetscCall(ISGetLocalSize(pP, &ni));
        PetscCall(ISGetIndices(pP, &idxs));
        for (i = 0; i < ni; i++) matis->sf_rootdata[idxs[i] - rst] = 1;
        PetscCall(ISRestoreIndices(pP, &idxs));
        PetscCall(PetscSFBcastBegin(matis->sf, MPIU_INT, matis->sf_rootdata, matis->sf_leafdata, MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(matis->sf, MPIU_INT, matis->sf_rootdata, matis->sf_leafdata, MPI_REPLACE));
        for (i = 0, ni = 0; i < n; i++)
          if (matis->sf_leafdata[i]) widxs[ni++] = i;
        PetscCall(PetscMalloc1(ni, &widxs2));
        PetscCall(ISLocalToGlobalMappingApply(l2g, ni, widxs, widxs2));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ni, widxs, PETSC_COPY_VALUES, &lP));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_lP", (PetscObject)lP));
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)ksp), ni, widxs2, PETSC_OWN_POINTER, &is1));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_gP", (PetscObject)is1));
        PetscCall(ISDestroy(&is1));
      } else {
        PetscCheck(lP, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing sequential pressure IS");
        PetscCall(ISGetLocalSize(lP, &ni));
        PetscCall(ISGetIndices(lP, &idxs));
        for (i = 0; i < ni; i++)
          if (idxs[i] >= 0 && idxs[i] < n) matis->sf_leafdata[idxs[i]] = 1;
        PetscCall(ISRestoreIndices(lP, &idxs));
        PetscCall(PetscSFReduceBegin(matis->sf, MPIU_INT, matis->sf_leafdata, matis->sf_rootdata, MPI_REPLACE));
        PetscCall(ISLocalToGlobalMappingApply(l2g, ni, idxs, widxs));
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)ksp), ni, widxs, PETSC_COPY_VALUES, &is1));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_gP", (PetscObject)is1));
        PetscCall(ISDestroy(&is1));
        PetscCall(PetscSFReduceEnd(matis->sf, MPIU_INT, matis->sf_leafdata, matis->sf_rootdata, MPI_REPLACE));
        for (i = 0, ni = 0; i < nl; i++)
          if (matis->sf_rootdata[i]) widxs[ni++] = i + rst;
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)ksp), ni, widxs, PETSC_COPY_VALUES, &pP));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_pP", (PetscObject)pP));
      }
      PetscCall(PetscFree(widxs));

      /* If there's any "interior pressure",
         we may want to use a discrete harmonic solver instead
         of a Stokes harmonic for the Dirichlet preconditioner
         Need to extract the interior velocity dofs in interior dofs ordering (iV)
         and interior pressure dofs in local ordering (iP) */
      if (!allp) {
        ISLocalToGlobalMapping l2g_t;

        PetscCall(ISDifference(lPall, lP, &is1));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_iP", (PetscObject)is1));
        PetscCall(ISDifference(II, is1, &is2));
        PetscCall(ISDestroy(&is1));
        PetscCall(ISLocalToGlobalMappingCreateIS(II, &l2g_t));
        PetscCall(ISGlobalToLocalMappingApplyIS(l2g_t, IS_GTOLM_DROP, is2, &is1));
        PetscCall(ISGetLocalSize(is1, &i));
        PetscCall(ISGetLocalSize(is2, &j));
        PetscCheck(i == j, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent local sizes %" PetscInt_FMT " and %" PetscInt_FMT " for iV", i, j);
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_iV", (PetscObject)is1));
        PetscCall(ISLocalToGlobalMappingDestroy(&l2g_t));
        PetscCall(ISDestroy(&is1));
        PetscCall(ISDestroy(&is2));
      }
      PetscCall(ISDestroy(&II));

      /* exclude selected pressures from the inner BDDC */
      if (pcbddc->DirichletBoundariesLocal) {
        IS       list[2], plP, isout;
        PetscInt np;

        /* need a parallel IS */
        PetscCall(ISGetLocalSize(lP, &np));
        PetscCall(ISGetIndices(lP, &idxs));
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)ksp), np, idxs, PETSC_USE_POINTER, &plP));
        list[0] = plP;
        list[1] = pcbddc->DirichletBoundariesLocal;
        PetscCall(ISConcatenate(PetscObjectComm((PetscObject)ksp), 2, list, &isout));
        PetscCall(ISSortRemoveDups(isout));
        PetscCall(ISDestroy(&plP));
        PetscCall(ISRestoreIndices(lP, &idxs));
        PetscCall(PCBDDCSetDirichletBoundariesLocal(fetidp->innerbddc, isout));
        PetscCall(ISDestroy(&isout));
      } else if (pcbddc->DirichletBoundaries) {
        IS list[2], isout;

        list[0] = pP;
        list[1] = pcbddc->DirichletBoundaries;
        PetscCall(ISConcatenate(PetscObjectComm((PetscObject)ksp), 2, list, &isout));
        PetscCall(ISSortRemoveDups(isout));
        PetscCall(PCBDDCSetDirichletBoundaries(fetidp->innerbddc, isout));
        PetscCall(ISDestroy(&isout));
      } else {
        IS       plP;
        PetscInt np;

        /* need a parallel IS */
        PetscCall(ISGetLocalSize(lP, &np));
        PetscCall(ISGetIndices(lP, &idxs));
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)ksp), np, idxs, PETSC_COPY_VALUES, &plP));
        PetscCall(PCBDDCSetDirichletBoundariesLocal(fetidp->innerbddc, plP));
        PetscCall(ISDestroy(&plP));
        PetscCall(ISRestoreIndices(lP, &idxs));
      }

      /* save CSR information for the pressure BDDC solver (if any) */
      if (schp) {
        PetscInt np, nt;

        PetscCall(MatGetSize(matis->A, &nt, NULL));
        PetscCall(ISGetLocalSize(lP, &np));
        if (np) {
          PetscInt *xadj = pcbddc->mat_graph->xadj;
          PetscInt *adjn = pcbddc->mat_graph->adjncy;
          PetscInt  nv   = pcbddc->mat_graph->nvtxs_csr;

          if (nv && nv == nt) {
            ISLocalToGlobalMapping pmap;
            PetscInt              *schp_csr, *schp_xadj, *schp_adjn, p;
            PetscContainer         c;

            PetscCall(ISLocalToGlobalMappingCreateIS(lPall, &pmap));
            PetscCall(ISGetIndices(lPall, &idxs));
            for (p = 0, nv = 0; p < np; p++) {
              PetscInt x, n = idxs[p];

              PetscCall(ISGlobalToLocalMappingApply(pmap, IS_GTOLM_DROP, xadj[n + 1] - xadj[n], adjn + xadj[n], &x, NULL));
              nv += x;
            }
            PetscCall(PetscMalloc1(np + 1 + nv, &schp_csr));
            schp_xadj = schp_csr;
            schp_adjn = schp_csr + np + 1;
            for (p = 0, schp_xadj[0] = 0; p < np; p++) {
              PetscInt x, n = idxs[p];

              PetscCall(ISGlobalToLocalMappingApply(pmap, IS_GTOLM_DROP, xadj[n + 1] - xadj[n], adjn + xadj[n], &x, schp_adjn + schp_xadj[p]));
              schp_xadj[p + 1] = schp_xadj[p] + x;
            }
            PetscCall(ISRestoreIndices(lPall, &idxs));
            PetscCall(ISLocalToGlobalMappingDestroy(&pmap));
            PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &c));
            PetscCall(PetscContainerSetPointer(c, schp_csr));
            PetscCall(PetscContainerSetUserDestroy(c, PetscContainerUserDestroyDefault));
            PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_pCSR", (PetscObject)c));
            PetscCall(PetscContainerDestroy(&c));
          }
        }
      }
      PetscCall(ISDestroy(&lPall));
      PetscCall(ISDestroy(&lP));
      fetidp->pP = pP;
    }

    /* total number of selected pressure dofs */
    PetscCall(ISGetSize(fetidp->pP, &totP));

    /* Set operator for inner BDDC */
    if (totP || fetidp->rhs_flip) {
      PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &nA));
    } else {
      PetscCall(PetscObjectReference((PetscObject)A));
      nA = A;
    }
    if (fetidp->rhs_flip) {
      PetscCall(MatDiagonalScale(nA, fetidp->rhs_flip, NULL));
      if (totP) {
        Mat lA2;

        PetscCall(MatISGetLocalMat(nA, &lA));
        PetscCall(MatDuplicate(lA, MAT_COPY_VALUES, &lA2));
        PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_lA", (PetscObject)lA2));
        PetscCall(MatDestroy(&lA2));
      }
    }

    if (totP) {
      PetscCall(MatSetOption(nA, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
      PetscCall(MatZeroRowsColumnsIS(nA, fetidp->pP, 1., NULL, NULL));
    } else {
      PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_lA", NULL));
    }
    PetscCall(MatGetNearNullSpace(Ap, &nnsp));
    if (!nnsp) PetscCall(MatGetNullSpace(Ap, &nnsp));
    if (!nnsp) PetscCall(MatGetNearNullSpace(A, &nnsp));
    if (!nnsp) PetscCall(MatGetNullSpace(A, &nnsp));
    PetscCall(MatSetNearNullSpace(nA, nnsp));
    PetscCall(PCSetOperators(fetidp->innerbddc, nA, nA));
    PetscCall(MatDestroy(&nA));

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

        PetscCall(MatCreateSubMatrix(A, fetidp->pP, fetidp->pP, MAT_INITIAL_MATRIX, &C));
        PetscCall(MatFindNonzeroRows(C, &nzrows));
        if (nzrows) {
          PetscInt i;

          PetscCall(ISGetSize(nzrows, &i));
          PetscCall(ISDestroy(&nzrows));
          if (!i) pisz = PETSC_TRUE;
        }
        if (!pisz) {
          PetscCall(MatScale(C, -1.)); /* i.e. Almost Incompressible Elasticity, Stokes discretized with Q1xQ1_stabilized */
          PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_C", (PetscObject)C));
        }
        PetscCall(MatDestroy(&C));
      }
      /* Divergence mat */
      if (!pcbddc->divudotp) {
        Mat       B;
        IS        P;
        IS        l2l = NULL;
        PetscBool save;

        PetscCall(PetscObjectQuery((PetscObject)fetidp->innerbddc, "__KSPFETIDP_aP", (PetscObject *)&P));
        if (!pisz) {
          IS       F, V;
          PetscInt m, M;

          PetscCall(MatGetOwnershipRange(A, &m, &M));
          PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A), M - m, m, 1, &F));
          PetscCall(ISComplement(P, m, M, &V));
          PetscCall(MatCreateSubMatrix(A, P, V, MAT_INITIAL_MATRIX, &B));
          {
            Mat_IS *Bmatis = (Mat_IS *)B->data;
            PetscCall(PetscObjectReference((PetscObject)Bmatis->getsub_cis));
            l2l = Bmatis->getsub_cis;
          }
          PetscCall(ISDestroy(&V));
          PetscCall(ISDestroy(&F));
        } else {
          PetscCall(MatCreateSubMatrix(A, P, NULL, MAT_INITIAL_MATRIX, &B));
        }
        save = pcbddc->compute_nonetflux; /* SetDivergenceMat activates nonetflux computation */
        PetscCall(PCBDDCSetDivergenceMat(fetidp->innerbddc, B, PETSC_FALSE, l2l));
        pcbddc->compute_nonetflux = save;
        PetscCall(MatDestroy(&B));
        PetscCall(ISDestroy(&l2l));
      }
      if (A != Ap) { /* user has provided a different Pmat, this always superseeds the setter (TODO: is it OK?) */
        /* use monolithic operator, we restrict later */
        PetscCall(KSPFETIDPSetPressureOperator(ksp, Ap));
      }
      PetscCall(PetscObjectQuery((PetscObject)fetidp->innerbddc, "__KSPFETIDP_PPmat", (PetscObject *)&PPmat));

      /* PPmat not present, use some default choice */
      if (!PPmat) {
        Mat C;

        PetscCall(PetscObjectQuery((PetscObject)fetidp->innerbddc, "__KSPFETIDP_C", (PetscObject *)&C));
        if (!schp && C) { /* non-zero pressure block, most likely Almost Incompressible Elasticity */
          PetscCall(KSPFETIDPSetPressureOperator(ksp, C));
        } else if (!pisz && schp) { /* we need the whole pressure mass matrix to define the interface BDDC */
          IS P;

          PetscCall(PetscObjectQuery((PetscObject)fetidp->innerbddc, "__KSPFETIDP_aP", (PetscObject *)&P));
          PetscCall(MatCreateSubMatrix(A, P, P, MAT_INITIAL_MATRIX, &C));
          PetscCall(MatScale(C, -1.));
          PetscCall(KSPFETIDPSetPressureOperator(ksp, C));
          PetscCall(MatDestroy(&C));
        } else { /* identity (need to be scaled properly by the user using e.g. a Richardson method */
          PetscInt nl;

          PetscCall(ISGetLocalSize(fetidp->pP, &nl));
          PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &C));
          PetscCall(MatSetSizes(C, nl, nl, totP, totP));
          PetscCall(MatSetType(C, MATAIJ));
          PetscCall(MatMPIAIJSetPreallocation(C, 1, NULL, 0, NULL));
          PetscCall(MatSeqAIJSetPreallocation(C, 1, NULL));
          PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
          PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
          PetscCall(MatShift(C, 1.));
          PetscCall(KSPFETIDPSetPressureOperator(ksp, C));
          PetscCall(MatDestroy(&C));
        }
      }

      /* Preconditioned operator for the pressure block */
      PetscCall(PetscObjectQuery((PetscObject)fetidp->innerbddc, "__KSPFETIDP_PPmat", (PetscObject *)&PPmat));
      if (PPmat) {
        Mat      C;
        IS       Pall;
        PetscInt AM, PAM, PAN, pam, pan, am, an, pl, pIl, pAg, pIg;

        PetscCall(PetscObjectQuery((PetscObject)fetidp->innerbddc, "__KSPFETIDP_aP", (PetscObject *)&Pall));
        PetscCall(MatGetSize(A, &AM, NULL));
        PetscCall(MatGetSize(PPmat, &PAM, &PAN));
        PetscCall(ISGetSize(Pall, &pAg));
        PetscCall(ISGetSize(fetidp->pP, &pIg));
        PetscCall(MatGetLocalSize(PPmat, &pam, &pan));
        PetscCall(MatGetLocalSize(A, &am, &an));
        PetscCall(ISGetLocalSize(Pall, &pIl));
        PetscCall(ISGetLocalSize(fetidp->pP, &pl));
        PetscCheck(PAM == PAN, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Pressure matrix must be square, unsupported %" PetscInt_FMT " x %" PetscInt_FMT, PAM, PAN);
        PetscCheck(pam == pan, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Local sizes of pressure matrix must be equal, unsupported %" PetscInt_FMT " x %" PetscInt_FMT, pam, pan);
        PetscCheck(pam == am || pam == pl || pam == pIl, PETSC_COMM_SELF, PETSC_ERR_USER, "Invalid number of local rows %" PetscInt_FMT " for pressure matrix! Supported are %" PetscInt_FMT ", %" PetscInt_FMT " or %" PetscInt_FMT, pam, am, pl, pIl);
        PetscCheck(pan == an || pan == pl || pan == pIl, PETSC_COMM_SELF, PETSC_ERR_USER, "Invalid number of local columns %" PetscInt_FMT " for pressure matrix! Supported are %" PetscInt_FMT ", %" PetscInt_FMT " or %" PetscInt_FMT, pan, an, pl, pIl);
        if (PAM == AM) { /* monolithic ordering, restrict to pressure */
          if (schp) {
            PetscCall(MatCreateSubMatrix(PPmat, Pall, Pall, MAT_INITIAL_MATRIX, &C));
          } else {
            PetscCall(MatCreateSubMatrix(PPmat, fetidp->pP, fetidp->pP, MAT_INITIAL_MATRIX, &C));
          }
        } else if (pAg == PAM) { /* global ordering for pressure only */
          if (!allp && !schp) {  /* solving for interface pressure only */
            IS restr;

            PetscCall(ISRenumber(fetidp->pP, NULL, NULL, &restr));
            PetscCall(MatCreateSubMatrix(PPmat, restr, restr, MAT_INITIAL_MATRIX, &C));
            PetscCall(ISDestroy(&restr));
          } else {
            PetscCall(PetscObjectReference((PetscObject)PPmat));
            C = PPmat;
          }
        } else if (pIg == PAM) { /* global ordering for selected pressure only */
          PetscCheck(!schp, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB, "Need the entire matrix");
          PetscCall(PetscObjectReference((PetscObject)PPmat));
          C = PPmat;
        } else SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Unable to use the pressure matrix");

        PetscCall(KSPFETIDPSetPressureOperator(ksp, C));
        PetscCall(MatDestroy(&C));
      } else SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB, "Missing Pmat for pressure block");
    } else { /* totP == 0 */
      PetscCall(PetscObjectCompose((PetscObject)fetidp->innerbddc, "__KSPFETIDP_pP", NULL));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_FETIDP(KSP ksp)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;
  PC_BDDC    *pcbddc = (PC_BDDC *)fetidp->innerbddc->data;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscCall(KSPFETIDPSetUpOperators(ksp));
  /* set up BDDC */
  PetscCall(PCSetErrorIfFailure(fetidp->innerbddc, ksp->errorifnotconverged));
  PetscCall(PCSetUp(fetidp->innerbddc));
  /* FETI-DP as it is implemented needs an exact coarse solver */
  if (pcbddc->coarse_ksp) {
    PetscCall(KSPSetTolerances(pcbddc->coarse_ksp, PETSC_SMALL, PETSC_SMALL, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetNormType(pcbddc->coarse_ksp, KSP_NORM_DEFAULT));
  }
  /* FETI-DP as it is implemented needs exact local Neumann solvers */
  PetscCall(KSPSetTolerances(pcbddc->ksp_R, PETSC_SMALL, PETSC_SMALL, PETSC_DEFAULT, 1000));
  PetscCall(KSPSetNormType(pcbddc->ksp_R, KSP_NORM_DEFAULT));

  /* setup FETI-DP operators
     If fetidp->statechanged is true, we need to update the operators
     needed in the saddle-point case. This should be replaced
     by a better logic when the FETI-DP matrix and preconditioner will
     have their own classes */
  if (pcbddc->new_primal_space || fetidp->statechanged) {
    Mat F; /* the FETI-DP matrix */
    PC  D; /* the FETI-DP preconditioner */
    PetscCall(KSPReset(fetidp->innerksp));
    PetscCall(PCBDDCCreateFETIDPOperators(fetidp->innerbddc, fetidp->fully_redundant, ((PetscObject)ksp)->prefix, &F, &D));
    PetscCall(KSPSetOperators(fetidp->innerksp, F, F));
    PetscCall(KSPSetTolerances(fetidp->innerksp, ksp->rtol, ksp->abstol, ksp->divtol, ksp->max_it));
    PetscCall(KSPSetPC(fetidp->innerksp, D));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)D, (PetscObject)fetidp->innerksp, 0));
    PetscCall(KSPSetFromOptions(fetidp->innerksp));
    PetscCall(MatCreateVecs(F, &(fetidp->innerksp)->vec_rhs, &(fetidp->innerksp)->vec_sol));
    PetscCall(MatDestroy(&F));
    PetscCall(PCDestroy(&D));
    if (fetidp->check) {
      PetscViewer viewer;

      if (!pcbddc->dbg_viewer) {
        viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
      } else {
        viewer = pcbddc->dbg_viewer;
      }
      PetscCall(KSPFETIDPCheckOperators(ksp, viewer));
    }
  }
  fetidp->statechanged     = PETSC_FALSE;
  pcbddc->new_primal_space = PETSC_FALSE;

  /* propagate settings to the inner solve */
  PetscCall(KSPGetComputeSingularValues(ksp, &flg));
  PetscCall(KSPSetComputeSingularValues(fetidp->innerksp, flg));
  if (ksp->res_hist) PetscCall(KSPSetResidualHistory(fetidp->innerksp, ksp->res_hist, ksp->res_hist_max, ksp->res_hist_reset));
  PetscCall(KSPSetErrorIfNotConverged(fetidp->innerksp, ksp->errorifnotconverged));
  PetscCall(KSPSetUp(fetidp->innerksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_FETIDP(KSP ksp)
{
  Mat                F, A;
  MatNullSpace       nsp;
  Vec                X, B, Xl, Bl;
  KSP_FETIDP        *fetidp = (KSP_FETIDP *)ksp->data;
  PC_BDDC           *pcbddc = (PC_BDDC *)fetidp->innerbddc->data;
  KSPConvergedReason reason;
  PC                 pc;
  PCFailedReason     pcreason;
  PetscInt           hist_len;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation, &cited));
  if (fetidp->saddlepoint) PetscCall(PetscCitationsRegister(citation2, &cited2));
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(KSPGetRhs(ksp, &B));
  PetscCall(KSPGetSolution(ksp, &X));
  PetscCall(KSPGetOperators(fetidp->innerksp, &F, NULL));
  PetscCall(KSPGetRhs(fetidp->innerksp, &Bl));
  PetscCall(KSPGetSolution(fetidp->innerksp, &Xl));
  PetscCall(PCBDDCMatFETIDPGetRHS(F, B, Bl));
  if (ksp->transpose_solve) {
    PetscCall(KSPSolveTranspose(fetidp->innerksp, Bl, Xl));
  } else {
    PetscCall(KSPSolve(fetidp->innerksp, Bl, Xl));
  }
  PetscCall(KSPGetConvergedReason(fetidp->innerksp, &reason));
  PetscCall(KSPGetPC(fetidp->innerksp, &pc));
  PetscCall(PCGetFailedReason(pc, &pcreason));
  if ((reason < 0 && reason != KSP_DIVERGED_ITS) || pcreason) {
    PetscInt its;
    PetscCall(KSPGetIterationNumber(fetidp->innerksp, &its));
    ksp->reason = KSP_DIVERGED_PC_FAILED;
    PetscCall(VecSetInf(Xl));
    PetscCall(PetscInfo(ksp, "Inner KSP solve failed: %s %s at iteration %" PetscInt_FMT, KSPConvergedReasons[reason], PCFailedReasons[pcreason], its));
  }
  PetscCall(PCBDDCMatFETIDPGetSolution(F, Xl, X));
  PetscCall(MatGetNullSpace(A, &nsp));
  if (nsp) PetscCall(MatNullSpaceRemove(nsp, X));
  /* update ksp with stats from inner ksp */
  PetscCall(KSPGetConvergedReason(fetidp->innerksp, &ksp->reason));
  PetscCall(KSPGetIterationNumber(fetidp->innerksp, &ksp->its));
  ksp->totalits += ksp->its;
  PetscCall(KSPGetResidualHistory(fetidp->innerksp, NULL, &hist_len));
  ksp->res_hist_len = (size_t)hist_len;
  /* restore defaults for inner BDDC (Pre/PostSolve flags) */
  pcbddc->temp_solution_used        = PETSC_FALSE;
  pcbddc->rhs_change                = PETSC_FALSE;
  pcbddc->exact_dirichlet_trick_app = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPReset_FETIDP(KSP ksp)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;
  PC_BDDC    *pcbddc;

  PetscFunctionBegin;
  PetscCall(ISDestroy(&fetidp->pP));
  PetscCall(VecDestroy(&fetidp->rhs_flip));
  /* avoid PCReset that does not take into account ref counting */
  PetscCall(PCDestroy(&fetidp->innerbddc));
  PetscCall(PCCreate(PetscObjectComm((PetscObject)ksp), &fetidp->innerbddc));
  PetscCall(PCSetType(fetidp->innerbddc, PCBDDC));
  pcbddc                   = (PC_BDDC *)fetidp->innerbddc->data;
  pcbddc->symmetric_primal = PETSC_FALSE;
  PetscCall(KSPDestroy(&fetidp->innerksp));
  fetidp->saddlepoint  = PETSC_FALSE;
  fetidp->matstate     = -1;
  fetidp->matnnzstate  = -1;
  fetidp->statechanged = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_FETIDP(KSP ksp)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPReset_FETIDP(ksp));
  PetscCall(PCDestroy(&fetidp->innerbddc));
  PetscCall(KSPDestroy(&fetidp->innerksp));
  PetscCall(PetscFree(fetidp->monctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPFETIDPSetInnerBDDC_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPFETIDPGetInnerBDDC_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPFETIDPGetInnerKSP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPFETIDPSetPressureOperator_C", NULL));
  PetscCall(PetscFree(ksp->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_FETIDP(KSP ksp, PetscViewer viewer)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;
  PetscBool   iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  fully redundant: %d\n", fetidp->fully_redundant));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  saddle point:    %d\n", fetidp->saddlepoint));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Inner KSP solver details\n"));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(KSPView(fetidp->innerksp, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "Inner BDDC solver details\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PCView(fetidp->innerbddc, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_FETIDP(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  KSP_FETIDP *fetidp = (KSP_FETIDP *)ksp->data;

  PetscFunctionBegin;
  /* set options prefixes for the inner objects, since the parent prefix will be valid at this point */
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerksp, ((PetscObject)ksp)->prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerksp, "fetidp_"));
  if (!fetidp->userbddc) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerbddc, ((PetscObject)ksp)->prefix));
    PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerbddc, "fetidp_bddc_"));
  }
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP FETIDP options");
  PetscCall(PetscOptionsBool("-ksp_fetidp_fullyredundant", "Use fully redundant multipliers", "none", fetidp->fully_redundant, &fetidp->fully_redundant, NULL));
  PetscCall(PetscOptionsBool("-ksp_fetidp_saddlepoint", "Activates support for saddle-point problems", NULL, fetidp->saddlepoint, &fetidp->saddlepoint, NULL));
  PetscCall(PetscOptionsBool("-ksp_fetidp_check", "Activates verbose debugging output FETI-DP operators", NULL, fetidp->check, &fetidp->check, NULL));
  PetscOptionsHeadEnd();
  PetscCall(PCSetFromOptions(fetidp->innerbddc));
  PetscFunctionReturn(0);
}

/*MC
     KSPFETIDP - The FETI-DP method [1]

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
   The matrix for the KSP must be of type `MATIS`.

   The FETI-DP linear system (automatically generated constructing an internal `PCBDDC` object) is solved using an internal `KSP` object.

    Options for the inner `KSP` and for the customization of the `PCBDDC` object can be specified at command line by using the prefixes -fetidp_ and -fetidp_bddc_. E.g.,
.vb
      -fetidp_ksp_type gmres -fetidp_bddc_pc_bddc_symmetric false
.ve
   will use `KSPGMRES` for the solution of the linear system on the Lagrange multipliers, generated using a non-symmetric `PCBDDC`.

   For saddle point problems with continuous pressures, the preconditioned operator for the pressure solver can be specified with `KSPFETIDPSetPressureOperator()`.
   Alternatively, the pressure operator is extracted from the precondioned matrix (if it is different from the linear solver matrix).
   If none of the above, an identity matrix will be created; the user then needs to scale it through a Richardson solver.
   Options for the pressure solver can be prefixed with -fetidp_fielsplit_p_, E.g.
.vb
      -fetidp_fielsplit_p_ksp_type preonly -fetidp_fielsplit_p_pc_type lu -fetidp_fielsplit_p_pc_factor_mat_solver_type mumps
.ve
   In order to use the deluxe version of FETI-DP, you must customize the inner `PCBDDC` operator with -fetidp_bddc_pc_bddc_use_deluxe_scaling -fetidp_bddc_pc_bddc_deluxe_singlemat and use
   non-redundant multipliers, i.e. -ksp_fetidp_fullyredundant false. Options for the scaling solver are prefixed by -fetidp_bddelta_, E.g.
.vb
      -fetidp_bddelta_pc_factor_mat_solver_type mumps -fetidp_bddelta_pc_type lu
.ve

   Some of the basic options such as the maximum number of iterations and tolerances are automatically passed from this `KSP` to the inner `KSP` that actually performs the iterations.

   The converged reason and number of iterations computed are passed from the inner `KSP` to this `KSP` at the end of the solution.

   Developer Note:
   Even though this method does not directly use any norms, the user is allowed to set the `KSPNormType` to any value.
   This is so users do not have to change `KSPNormTyp`e options when they switch from other `KSP` methods to this one.

   References:
+  [1] - C. Farhat, M. Lesoinne, P. LeTallec, K. Pierson, and D. Rixen, FETI-DP: a dual-primal unified FETI method. I. A faster alternative to the two-level FETI method, Internat. J. Numer. Methods Engrg., 50 (2001), pp. 1523--1544
-  [2] - X. Tu, J. Li, A FETI-DP type domain decomposition algorithm for three-dimensional incompressible Stokes equations, SIAM J. Numer. Anal., 53 (2015), pp. 720-742

.seealso: [](chapter_ksp), `MATIS`, `PCBDDC`, `KSPFETIDPSetInnerBDDC()`, `KSPFETIDPGetInnerBDDC()`, `KSPFETIDPGetInnerKSP()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_FETIDP(KSP ksp)
{
  KSP_FETIDP    *fetidp;
  KSP_FETIDPMon *monctx;
  PC_BDDC       *pcbddc;
  PC             pc;

  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NATURAL, PC_LEFT, 2));

  PetscCall(PetscNew(&fetidp));
  fetidp->matstate     = -1;
  fetidp->matnnzstate  = -1;
  fetidp->statechanged = PETSC_TRUE;

  ksp->data                              = (void *)fetidp;
  ksp->ops->setup                        = KSPSetUp_FETIDP;
  ksp->ops->solve                        = KSPSolve_FETIDP;
  ksp->ops->destroy                      = KSPDestroy_FETIDP;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_FETIDP;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_FETIDP;
  ksp->ops->view                         = KSPView_FETIDP;
  ksp->ops->setfromoptions               = KSPSetFromOptions_FETIDP;
  ksp->ops->buildsolution                = KSPBuildSolution_FETIDP;
  ksp->ops->buildresidual                = KSPBuildResidualDefault;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  /* create the inner KSP for the Lagrange multipliers */
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)ksp), &fetidp->innerksp));
  PetscCall(KSPGetPC(fetidp->innerksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  /* monitor */
  PetscCall(PetscNew(&monctx));
  monctx->parentksp = ksp;
  fetidp->monctx    = monctx;
  PetscCall(KSPMonitorSet(fetidp->innerksp, KSPMonitor_FETIDP, fetidp->monctx, NULL));
  /* create the inner BDDC */
  PetscCall(PCCreate(PetscObjectComm((PetscObject)ksp), &fetidp->innerbddc));
  PetscCall(PCSetType(fetidp->innerbddc, PCBDDC));
  /* make sure we always obtain a consistent FETI-DP matrix
     for symmetric problems, the user can always customize it through the command line */
  pcbddc                   = (PC_BDDC *)fetidp->innerbddc->data;
  pcbddc->symmetric_primal = PETSC_FALSE;
  /* composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPFETIDPSetInnerBDDC_C", KSPFETIDPSetInnerBDDC_FETIDP));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPFETIDPGetInnerBDDC_C", KSPFETIDPGetInnerBDDC_FETIDP));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPFETIDPGetInnerKSP_C", KSPFETIDPGetInnerKSP_FETIDP));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPFETIDPSetPressureOperator_C", KSPFETIDPSetPressureOperator_FETIDP));
  /* need to call KSPSetUp_FETIDP even with KSP_SETUP_NEWMATRIX */
  ksp->setupnewmatrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}

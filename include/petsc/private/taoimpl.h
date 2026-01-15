#pragma once

#include <petsctao.h>
#include <petsctaolinesearch.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      TaoRegisterAllCalled;
PETSC_EXTERN PetscErrorCode TaoRegisterAll(void);

typedef struct _TaoOps *TaoOps;

struct _TaoOps {
  /* Methods set by application */
  PetscErrorCode (*computeresidual)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computeresidualjacobian)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computeconstraints)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computeinequalityconstraints)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computeequalityconstraints)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computejacobian)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computejacobianstate)(Tao, Vec, Mat, Mat, Mat, void *);
  PetscErrorCode (*computejacobiandesign)(Tao, Vec, Mat, void *);
  PetscErrorCode (*computejacobianinequality)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computejacobianequality)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computebounds)(Tao, Vec, Vec, void *);
  PetscErrorCode (*update)(Tao, PetscInt, void *);
  PetscErrorCode (*convergencetest)(Tao, void *);
  PetscErrorCode (*convergencedestroy)(void *);

  /* Methods set by solver */
  PetscErrorCode (*computedual)(Tao, Vec, Vec);
  PetscErrorCode (*setup)(Tao);
  PetscErrorCode (*solve)(Tao);
  PetscErrorCode (*view)(Tao, PetscViewer);
  PetscErrorCode (*setfromoptions)(Tao, PetscOptionItems);
  PetscErrorCode (*destroy)(Tao);
};

#define MAXTAOMONITORS 10

typedef struct _n_TaoTermMapping TaoTermMapping;

/*S
   TaoTermMapping - Object held by either `Tao` or `TaoTerm` with `TAOTERMSUM` type
   that contain necessary information regarding mapping matrix

   Level: developer

   Notes:
   This object is internally handled in `TaoAddTerm()`, which couples `TaoTerm` with
   appropriately mapped gradients, mapped Hessians, and other necessary information.

   This object is also used in `TAOTERMSUM` to handle mapping of summands of `TAOTERMSUM`.

   Users would not need to work directly with the `TaoTermMapping`, rather, they work with
   `Tao` with added terms via `TaoAddTerm()`, or with `TaoTerm` with type `TAOTERMSUM`.

   Developer Note:
   Currently, as `MatPtAP` does not support diagonal matrices, internal work matrices was added
   for a workaround.

.seealso: [](ch_tao), `Tao`, `TaoAddTerm()`, `TAOTERMSUM`,
S*/
struct _n_TaoTermMapping {
  char       *prefix;
  TaoTerm     term;
  PetscReal   scale;
  Mat         map;
  Vec         _map_output;
  Vec         _unmapped_gradient;
  Vec         _mapped_gradient;
  Mat         _unmapped_H;
  Mat         _unmapped_Hpre;
  Mat         _mapped_H;
  Mat         _mapped_Hpre;
  Mat         _mapped_H_work; /* Temporary work matrices for PtAP for diagonal A */
  Mat         _mapped_Hpre_work;
  TaoTermMask mask;
};

#define TaoTermObjectiveMasked(a) ((a) & TAOTERM_MASK_OBJECTIVE)
#define TaoTermGradientMasked(a)  ((a) & TAOTERM_MASK_GRADIENT)
#define TaoTermHessianMasked(a)   ((a) & TAOTERM_MASK_HESSIAN)

struct _p_Tao {
  PETSCHEADER(struct _TaoOps);
  PetscCtx ctx; /* user provided context */
  void    *user_lsresP;
  void    *user_lsjacP;
  void    *user_conP;
  void    *user_con_equalityP;
  void    *user_con_inequalityP;
  void    *user_jacP;
  void    *user_jac_equalityP;
  void    *user_jac_inequalityP;
  void    *user_jac_stateP;
  void    *user_jac_designP;
  void    *user_boundsP;
  void    *user_update;

  PetscErrorCode (*monitor[MAXTAOMONITORS])(Tao, void *);
  PetscCtxDestroyFn *monitordestroy[MAXTAOMONITORS];
  void              *monitorcontext[MAXTAOMONITORS];
  PetscInt           numbermonitors;
  void              *cnvP;
  TaoConvergedReason reason;

  PetscBool setupcalled;
  void     *data;

  Vec        solution;
  Vec        gradient;
  Vec        stepdirection;
  Vec        XL;
  Vec        XU;
  Vec        IL;
  Vec        IU;
  Vec        DI;
  Vec        DE;
  Mat        hessian;
  Mat        hessian_pre;
  Mat        gradient_norm;
  Vec        gradient_norm_tmp;
  Vec        ls_res;
  Mat        ls_jac;
  Mat        ls_jac_pre;
  Vec        res_weights_v;
  PetscInt   res_weights_n;
  PetscInt  *res_weights_rows;
  PetscInt  *res_weights_cols;
  PetscReal *res_weights_w;
  Vec        constraints;
  Vec        constraints_equality;
  Vec        constraints_inequality;
  Mat        jacobian;
  Mat        jacobian_pre;
  Mat        jacobian_inequality;
  Mat        jacobian_inequality_pre;
  Mat        jacobian_equality;
  Mat        jacobian_equality_pre;
  Mat        jacobian_state;
  Mat        jacobian_state_inv;
  Mat        jacobian_design;
  Mat        jacobian_state_pre;
  Mat        jacobian_design_pre;
  IS         state_is;
  IS         design_is;
  PetscReal  step;
  PetscReal  residual;
  PetscReal  gnorm0;
  PetscReal  cnorm;
  PetscReal  cnorm0;
  PetscReal  fc;

  PetscInt max_constraints;
  PetscInt nfuncs;
  PetscInt ngrads;
  PetscInt nfuncgrads;
  PetscInt nhess;
  PetscInt niter;
  PetscInt ntotalits;
  PetscInt nconstraints;
  PetscInt niconstraints;
  PetscInt neconstraints;
  PetscInt nres;
  PetscInt njac;
  PetscInt njac_equality;
  PetscInt njac_inequality;
  PetscInt njac_state;
  PetscInt njac_design;

  PetscInt ksp_its;     /* KSP iterations for this solver iteration */
  PetscInt ksp_tot_its; /* Total (cumulative) KSP iterations */

  TaoLineSearch linesearch;
  PetscBool     lsflag; /* goes up when line search fails */
  KSP           ksp;
  PetscReal     trust; /* Current trust region */

  /* EW type forcing term */
  PetscBool ksp_ewconv;
  SNES      snes_ewdummy;

  PetscObjectParameterDeclare(PetscReal, gatol);
  PetscObjectParameterDeclare(PetscReal, grtol);
  PetscObjectParameterDeclare(PetscReal, gttol);
  PetscObjectParameterDeclare(PetscReal, catol);
  PetscObjectParameterDeclare(PetscReal, crtol);
  PetscObjectParameterDeclare(PetscReal, steptol);
  PetscObjectParameterDeclare(PetscReal, fmin);
  PetscObjectParameterDeclare(PetscInt, max_it);
  PetscObjectParameterDeclare(PetscInt, max_funcs);
  PetscObjectParameterDeclare(PetscReal, trust0); /* initial trust region radius */

  PetscBool printreason;
  PetscBool viewsolution;
  PetscBool viewgradient;
  PetscBool viewconstraints;
  PetscBool viewhessian;
  PetscBool viewjacobian;
  PetscBool bounded;
  PetscBool constrained;
  PetscBool eq_constrained;
  PetscBool ineq_constrained;
  PetscBool ineq_doublesided;
  PetscBool header_printed;
  PetscBool recycle;

  TaoSubsetType subset_type;
  PetscInt      hist_max;   /* Number of iteration histories to keep */
  PetscReal    *hist_obj;   /* obj value at each iteration */
  PetscReal    *hist_resid; /* residual at each iteration */
  PetscReal    *hist_cnorm; /* constraint norm at each iteration */
  PetscInt     *hist_lits;  /* number of ksp its at each TAO iteration */
  PetscInt      hist_len;
  PetscBool     hist_reset;
  PetscBool     hist_malloc;

  TaoTermMapping objective_term; /* TaoTerm in use */
  Vec            objective_parameters;
  PetscInt       num_terms;
  PetscBool      term_set;

  TaoTerm   callbacks; /* TAOTERMCALLBACKS for the original callbacks */
  PetscBool uses_hessian_matrices;
  PetscBool uses_gradient;
};

PETSC_EXTERN PetscLogEvent TAO_Solve;
PETSC_EXTERN PetscLogEvent TAO_ConstraintsEval;
PETSC_EXTERN PetscLogEvent TAO_JacobianEval;
PETSC_INTERN PetscLogEvent TAO_ResidualEval;

PETSC_INTERN PetscLogEvent TAOTERM_ObjectiveEval;
PETSC_INTERN PetscLogEvent TAOTERM_GradientEval;
PETSC_INTERN PetscLogEvent TAOTERM_ObjGradEval;
PETSC_INTERN PetscLogEvent TAOTERM_HessianEval;

static inline PetscErrorCode TaoLogConvergenceHistory(Tao tao, PetscReal obj, PetscReal resid, PetscReal cnorm, PetscInt totits)
{
  PetscFunctionBegin;
  if (tao->hist_max > tao->hist_len) {
    if (tao->hist_obj) tao->hist_obj[tao->hist_len] = obj;
    if (tao->hist_resid) tao->hist_resid[tao->hist_len] = resid;
    if (tao->hist_cnorm) tao->hist_cnorm[tao->hist_len] = cnorm;
    if (tao->hist_lits) {
      PetscInt sits = totits;
      PetscCheck(tao->hist_len >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "History length cannot be negative");
      for (PetscInt i = 0; i < tao->hist_len; i++) sits -= tao->hist_lits[i];
      tao->hist_lits[tao->hist_len] = sits;
    }
    tao->hist_len++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTestGradient_Internal(Tao, Vec, Vec, PetscViewer, PetscViewer);

typedef struct _TaoTermOps *TaoTermOps;

struct _TaoTermOps {
  PetscErrorCode (*setfromoptions)(TaoTerm, PetscOptionItems);
  PetscErrorCode (*setup)(TaoTerm);
  PetscErrorCode (*view)(TaoTerm, PetscViewer);
  PetscErrorCode (*destroy)(TaoTerm);

  TaoTermObjectiveFn            *objective;
  TaoTermObjectiveAndGradientFn *objectiveandgradient;
  TaoTermGradientFn             *gradient;
  TaoTermHessianFn              *hessian;

  PetscErrorCode (*isobjectivedefined)(TaoTerm, PetscBool *);
  PetscErrorCode (*isgradientdefined)(TaoTerm, PetscBool *);
  PetscErrorCode (*isobjectiveandgradientdefined)(TaoTerm, PetscBool *);
  PetscErrorCode (*ishessiandefined)(TaoTerm, PetscBool *);
  PetscErrorCode (*iscreatehessianmatricesdefined)(TaoTerm, PetscBool *);
  PetscErrorCode (*iscomputehessianfdpossible)(TaoTerm, PetscBool3 *);

  PetscErrorCode (*createsolutionvec)(TaoTerm, Vec *);
  PetscErrorCode (*createparametersvec)(TaoTerm, Vec *);
  PetscErrorCode (*createhessianmatrices)(TaoTerm, Mat *, Mat *);
};

struct _p_TaoTerm {
  PETSCHEADER(struct _TaoTermOps);
  void                 *data;
  PetscBool             setup_called;
  Mat                   solution_factory; // dummies used to create vectors
  Mat                   parameters_factory;
  Mat                   parameters_factory_orig; // copy so that parameters_factory can be made a reference of solution_factory if parameter space == vector space
  TaoTermParametersMode parameters_mode;
  PetscBool             Hpre_is_H; // Hessian mode data
  MatType               H_mattype;
  MatType               Hpre_mattype;

  MatType H_mattype_pre_fd_push;
  MatType Hpre_mattype_pre_fd_push;

  PetscInt ngrad_mffd;

  PetscReal fd_delta;      // increment for TaoTermGradientFD()
  PetscInt  fd_grad_level; // push/pop using finite difference for the gradient
  PetscInt  fd_hess_level; // push/pop using finite difference for the Hessian
};

PETSC_INTERN PetscErrorCode TaoTermRegisterAll(void);

PETSC_INTERN PetscErrorCode TaoTermCreateCallbacks(Tao, TaoTerm *);

PETSC_INTERN PetscErrorCode TaoTermCreate_ElementwiseDivergence_Internal(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermDestroy_ElementwiseDivergence_Internal(TaoTerm);

PETSC_INTERN PetscErrorCode TaoTermCallbacksSetObjective(TaoTerm, PetscErrorCode (*)(Tao, Vec, PetscReal *, PetscCtx), PetscCtx);
PETSC_INTERN PetscErrorCode TaoTermCallbacksSetGradient(TaoTerm, PetscErrorCode (*)(Tao, Vec, Vec, PetscCtx), PetscCtx);
PETSC_INTERN PetscErrorCode TaoTermCallbacksSetObjectiveAndGradient(TaoTerm, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx);
PETSC_INTERN PetscErrorCode TaoTermCallbacksSetHessian(TaoTerm, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);

PETSC_INTERN PetscErrorCode TaoTermCallbacksGetObjective(TaoTerm, PetscErrorCode (**)(Tao, Vec, PetscReal *, PetscCtx), PetscCtxRt);
PETSC_INTERN PetscErrorCode TaoTermCallbacksGetGradient(TaoTerm, PetscErrorCode (**)(Tao, Vec, Vec, PetscCtx), PetscCtxRt);
PETSC_INTERN PetscErrorCode TaoTermCallbacksGetObjectiveAndGradient(TaoTerm, PetscErrorCode (**)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtxRt);
PETSC_INTERN PetscErrorCode TaoTermCallbacksGetHessian(TaoTerm, PetscErrorCode (**)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtxRt);

PETSC_INTERN PetscErrorCode TaoTermMappingSetData(TaoTermMapping *, const char *, PetscReal, TaoTerm, Mat);
PETSC_INTERN PetscErrorCode TaoTermMappingGetData(TaoTermMapping *, const char **, PetscReal *, TaoTerm *, Mat *);
PETSC_INTERN PetscErrorCode TaoTermMappingReset(TaoTermMapping *);
PETSC_INTERN PetscErrorCode TaoTermMappingComputeObjective(TaoTermMapping *, Vec, Vec, InsertMode, PetscReal *);
PETSC_INTERN PetscErrorCode TaoTermMappingComputeGradient(TaoTermMapping *, Vec, Vec, InsertMode, Vec);
PETSC_INTERN PetscErrorCode TaoTermMappingComputeObjectiveAndGradient(TaoTermMapping *, Vec, Vec, InsertMode, PetscReal *, Vec);
PETSC_INTERN PetscErrorCode TaoTermMappingComputeHessian(TaoTermMapping *, Vec, Vec, InsertMode, Mat, Mat);
PETSC_INTERN PetscErrorCode TaoTermMappingSetUp(TaoTermMapping *);
PETSC_INTERN PetscErrorCode TaoTermMappingCreateSolutionVec(TaoTermMapping *, Vec *);
PETSC_INTERN PetscErrorCode TaoTermMappingCreateParametersVec(TaoTermMapping *, Vec *);
PETSC_INTERN PetscErrorCode TaoTermMappingCreateHessianMatrices(TaoTermMapping *, Mat *, Mat *);

PETSC_INTERN PetscErrorCode VecIfNotCongruentGetSameLayoutVec(Vec, Vec *);

PETSC_INTERN PetscErrorCode TaoTermCreateHessianMatricesDefault_H_Internal(TaoTerm, Mat *, Mat *, PetscBool, MatType);
PETSC_INTERN PetscErrorCode TaoTermCreateHessianMatricesDefault_Hpre_Internal(TaoTerm, Mat *, Mat *, PetscBool, MatType);

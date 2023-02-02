#include <../src/tao/unconstrained/impls/bmrm/bmrm.h>

static PetscErrorCode init_df_solver(TAO_DF *);
static PetscErrorCode ensure_df_space(PetscInt, TAO_DF *);
static PetscErrorCode destroy_df_solver(TAO_DF *);
static PetscReal      phi(PetscReal *, PetscInt, PetscReal, PetscReal *, PetscReal, PetscReal *, PetscReal *, PetscReal *);
static PetscInt       project(PetscInt, PetscReal *, PetscReal, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, TAO_DF *);
static PetscErrorCode solve(TAO_DF *);

/*------------------------------------------------------------*/
/* The main solver function

   f = Remp(W)          This is what the user provides us from the application layer
   So the ComputeGradient function for instance should get us back the subgradient of Remp(W)

   Regularizer assumed to be L2 norm = lambda*0.5*W'W ()
*/

static PetscErrorCode make_grad_node(Vec X, Vec_Chain **p)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(p));
  PetscCall(VecDuplicate(X, &(*p)->V));
  PetscCall(VecCopy(X, (*p)->V));
  (*p)->next = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroy_grad_list(Vec_Chain *head)
{
  Vec_Chain *p = head->next, *q;

  PetscFunctionBegin;
  while (p) {
    q = p->next;
    PetscCall(VecDestroy(&p->V));
    PetscCall(PetscFree(p));
    p = q;
  }
  head->next = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_BMRM(Tao tao)
{
  TAO_DF    df;
  TAO_BMRM *bmrm = (TAO_BMRM *)tao->data;

  /* Values and pointers to parts of the optimization problem */
  PetscReal   f = 0.0;
  Vec         W = tao->solution;
  Vec         G = tao->gradient;
  PetscReal   lambda;
  PetscReal   bt;
  Vec_Chain   grad_list, *tail_glist, *pgrad;
  PetscInt    i;
  PetscMPIInt rank;

  /* Used in converged criteria check */
  PetscReal reg;
  PetscReal jtwt = 0.0, max_jtwt, pre_epsilon, epsilon, jw, min_jw;
  PetscReal innerSolverTol;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  lambda = bmrm->lambda;

  /* Check Stopping Condition */
  tao->step      = 1.0;
  max_jtwt       = -BMRM_INFTY;
  min_jw         = BMRM_INFTY;
  innerSolverTol = 1.0;
  epsilon        = 0.0;

  if (rank == 0) {
    PetscCall(init_df_solver(&df));
    grad_list.next = NULL;
    tail_glist     = &grad_list;
  }

  df.tol      = 1e-6;
  tao->reason = TAO_CONTINUE_ITERATING;

  /*-----------------Algorithm Begins------------------------*/
  /* make the scatter */
  PetscCall(VecScatterCreateToZero(W, &bmrm->scatter, &bmrm->local_w));
  PetscCall(VecAssemblyBegin(bmrm->local_w));
  PetscCall(VecAssemblyEnd(bmrm->local_w));

  /* NOTE: In application pass the sub-gradient of Remp(W) */
  PetscCall(TaoComputeObjectiveAndGradient(tao, W, &f, G));
  PetscCall(TaoLogConvergenceHistory(tao, f, 1.0, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, 1.0, 0.0, tao->step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);

    /* compute bt = Remp(Wt-1) - <Wt-1, At> */
    PetscCall(VecDot(W, G, &bt));
    bt = f - bt;

    /* First gather the gradient to the rank-0 node */
    PetscCall(VecScatterBegin(bmrm->scatter, G, bmrm->local_w, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(bmrm->scatter, G, bmrm->local_w, INSERT_VALUES, SCATTER_FORWARD));

    /* Bring up the inner solver */
    if (rank == 0) {
      PetscCall(ensure_df_space(tao->niter + 1, &df));
      PetscCall(make_grad_node(bmrm->local_w, &pgrad));
      tail_glist->next = pgrad;
      tail_glist       = pgrad;

      df.a[tao->niter] = 1.0;
      df.f[tao->niter] = -bt;
      df.u[tao->niter] = 1.0;
      df.l[tao->niter] = 0.0;

      /* set up the Q */
      pgrad = grad_list.next;
      for (i = 0; i <= tao->niter; i++) {
        PetscCheck(pgrad, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Assert that there are at least tao->niter+1 pgrad available");
        PetscCall(VecDot(pgrad->V, bmrm->local_w, &reg));
        df.Q[i][tao->niter] = df.Q[tao->niter][i] = reg / lambda;
        pgrad                                     = pgrad->next;
      }

      if (tao->niter > 0) {
        df.x[tao->niter] = 0.0;
        PetscCall(solve(&df));
      } else df.x[0] = 1.0;

      /* now computing Jt*(alpha_t) which should be = Jt(wt) to check convergence */
      jtwt = 0.0;
      PetscCall(VecSet(bmrm->local_w, 0.0));
      pgrad = grad_list.next;
      for (i = 0; i <= tao->niter; i++) {
        jtwt -= df.x[i] * df.f[i];
        PetscCall(VecAXPY(bmrm->local_w, -df.x[i] / lambda, pgrad->V));
        pgrad = pgrad->next;
      }

      PetscCall(VecNorm(bmrm->local_w, NORM_2, &reg));
      reg = 0.5 * lambda * reg * reg;
      jtwt -= reg;
    } /* end if rank == 0 */

    /* scatter the new W to all nodes */
    PetscCall(VecScatterBegin(bmrm->scatter, bmrm->local_w, W, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(bmrm->scatter, bmrm->local_w, W, INSERT_VALUES, SCATTER_REVERSE));

    PetscCall(TaoComputeObjectiveAndGradient(tao, W, &f, G));

    PetscCallMPI(MPI_Bcast(&jtwt, 1, MPIU_REAL, 0, comm));
    PetscCallMPI(MPI_Bcast(&reg, 1, MPIU_REAL, 0, comm));

    jw = reg + f; /* J(w) = regularizer + Remp(w) */
    if (jw < min_jw) min_jw = jw;
    if (jtwt > max_jtwt) max_jtwt = jtwt;

    pre_epsilon = epsilon;
    epsilon     = min_jw - jtwt;

    if (rank == 0) {
      if (innerSolverTol > epsilon) innerSolverTol = epsilon;
      else if (innerSolverTol < 1e-7) innerSolverTol = 1e-7;

      /* if the annealing doesn't work well, lower the inner solver tolerance */
      if (pre_epsilon < epsilon) innerSolverTol *= 0.2;

      df.tol = innerSolverTol * 0.5;
    }

    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao, min_jw, epsilon, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, min_jw, epsilon, 0.0, tao->step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  }

  /* free all the memory */
  if (rank == 0) {
    PetscCall(destroy_grad_list(&grad_list));
    PetscCall(destroy_df_solver(&df));
  }

  PetscCall(VecDestroy(&bmrm->local_w));
  PetscCall(VecScatterDestroy(&bmrm->scatter));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------- */

static PetscErrorCode TaoSetup_BMRM(Tao tao)
{
  PetscFunctionBegin;
  /* Allocate some arrays */
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoDestroy_BMRM(Tao tao)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_BMRM(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_BMRM *bmrm = (TAO_BMRM *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "BMRM for regularized risk minimization");
  PetscCall(PetscOptionsReal("-tao_bmrm_lambda", "regulariser weight", "", 100, &bmrm->lambda, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_BMRM(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/
/*MC
  TAOBMRM - bundle method for regularized risk minimization

  Options Database Keys:
. - tao_bmrm_lambda - regulariser weight

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_BMRM(Tao tao)
{
  TAO_BMRM *bmrm;

  PetscFunctionBegin;
  tao->ops->setup          = TaoSetup_BMRM;
  tao->ops->solve          = TaoSolve_BMRM;
  tao->ops->view           = TaoView_BMRM;
  tao->ops->setfromoptions = TaoSetFromOptions_BMRM;
  tao->ops->destroy        = TaoDestroy_BMRM;

  PetscCall(PetscNew(&bmrm));
  bmrm->lambda = 1.0;
  tao->data    = (void *)bmrm;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;
  if (!tao->gatol_changed) tao->gatol = 1.0e-12;
  if (!tao->grtol_changed) tao->grtol = 1.0e-12;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode init_df_solver(TAO_DF *df)
{
  PetscInt i, n = INCRE_DIM;

  PetscFunctionBegin;
  /* default values */
  df->maxProjIter = 200;
  df->maxPGMIter  = 300000;
  df->b           = 1.0;

  /* memory space required by Dai-Fletcher */
  df->cur_num_cp = n;
  PetscCall(PetscMalloc1(n, &df->f));
  PetscCall(PetscMalloc1(n, &df->a));
  PetscCall(PetscMalloc1(n, &df->l));
  PetscCall(PetscMalloc1(n, &df->u));
  PetscCall(PetscMalloc1(n, &df->x));
  PetscCall(PetscMalloc1(n, &df->Q));

  for (i = 0; i < n; i++) PetscCall(PetscMalloc1(n, &df->Q[i]));

  PetscCall(PetscMalloc1(n, &df->g));
  PetscCall(PetscMalloc1(n, &df->y));
  PetscCall(PetscMalloc1(n, &df->tempv));
  PetscCall(PetscMalloc1(n, &df->d));
  PetscCall(PetscMalloc1(n, &df->Qd));
  PetscCall(PetscMalloc1(n, &df->t));
  PetscCall(PetscMalloc1(n, &df->xplus));
  PetscCall(PetscMalloc1(n, &df->tplus));
  PetscCall(PetscMalloc1(n, &df->sk));
  PetscCall(PetscMalloc1(n, &df->yk));

  PetscCall(PetscMalloc1(n, &df->ipt));
  PetscCall(PetscMalloc1(n, &df->ipt2));
  PetscCall(PetscMalloc1(n, &df->uv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ensure_df_space(PetscInt dim, TAO_DF *df)
{
  PetscReal *tmp, **tmp_Q;
  PetscInt   i, n, old_n;

  PetscFunctionBegin;
  df->dim = dim;
  if (dim <= df->cur_num_cp) PetscFunctionReturn(PETSC_SUCCESS);

  old_n = df->cur_num_cp;
  df->cur_num_cp += INCRE_DIM;
  n = df->cur_num_cp;

  /* memory space required by dai-fletcher */
  PetscCall(PetscMalloc1(n, &tmp));
  PetscCall(PetscArraycpy(tmp, df->f, old_n));
  PetscCall(PetscFree(df->f));
  df->f = tmp;

  PetscCall(PetscMalloc1(n, &tmp));
  PetscCall(PetscArraycpy(tmp, df->a, old_n));
  PetscCall(PetscFree(df->a));
  df->a = tmp;

  PetscCall(PetscMalloc1(n, &tmp));
  PetscCall(PetscArraycpy(tmp, df->l, old_n));
  PetscCall(PetscFree(df->l));
  df->l = tmp;

  PetscCall(PetscMalloc1(n, &tmp));
  PetscCall(PetscArraycpy(tmp, df->u, old_n));
  PetscCall(PetscFree(df->u));
  df->u = tmp;

  PetscCall(PetscMalloc1(n, &tmp));
  PetscCall(PetscArraycpy(tmp, df->x, old_n));
  PetscCall(PetscFree(df->x));
  df->x = tmp;

  PetscCall(PetscMalloc1(n, &tmp_Q));
  for (i = 0; i < n; i++) {
    PetscCall(PetscMalloc1(n, &tmp_Q[i]));
    if (i < old_n) {
      PetscCall(PetscArraycpy(tmp_Q[i], df->Q[i], old_n));
      PetscCall(PetscFree(df->Q[i]));
    }
  }

  PetscCall(PetscFree(df->Q));
  df->Q = tmp_Q;

  PetscCall(PetscFree(df->g));
  PetscCall(PetscMalloc1(n, &df->g));

  PetscCall(PetscFree(df->y));
  PetscCall(PetscMalloc1(n, &df->y));

  PetscCall(PetscFree(df->tempv));
  PetscCall(PetscMalloc1(n, &df->tempv));

  PetscCall(PetscFree(df->d));
  PetscCall(PetscMalloc1(n, &df->d));

  PetscCall(PetscFree(df->Qd));
  PetscCall(PetscMalloc1(n, &df->Qd));

  PetscCall(PetscFree(df->t));
  PetscCall(PetscMalloc1(n, &df->t));

  PetscCall(PetscFree(df->xplus));
  PetscCall(PetscMalloc1(n, &df->xplus));

  PetscCall(PetscFree(df->tplus));
  PetscCall(PetscMalloc1(n, &df->tplus));

  PetscCall(PetscFree(df->sk));
  PetscCall(PetscMalloc1(n, &df->sk));

  PetscCall(PetscFree(df->yk));
  PetscCall(PetscMalloc1(n, &df->yk));

  PetscCall(PetscFree(df->ipt));
  PetscCall(PetscMalloc1(n, &df->ipt));

  PetscCall(PetscFree(df->ipt2));
  PetscCall(PetscMalloc1(n, &df->ipt2));

  PetscCall(PetscFree(df->uv));
  PetscCall(PetscMalloc1(n, &df->uv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_df_solver(TAO_DF *df)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(PetscFree(df->f));
  PetscCall(PetscFree(df->a));
  PetscCall(PetscFree(df->l));
  PetscCall(PetscFree(df->u));
  PetscCall(PetscFree(df->x));

  for (i = 0; i < df->cur_num_cp; i++) PetscCall(PetscFree(df->Q[i]));
  PetscCall(PetscFree(df->Q));
  PetscCall(PetscFree(df->ipt));
  PetscCall(PetscFree(df->ipt2));
  PetscCall(PetscFree(df->uv));
  PetscCall(PetscFree(df->g));
  PetscCall(PetscFree(df->y));
  PetscCall(PetscFree(df->tempv));
  PetscCall(PetscFree(df->d));
  PetscCall(PetscFree(df->Qd));
  PetscCall(PetscFree(df->t));
  PetscCall(PetscFree(df->xplus));
  PetscCall(PetscFree(df->tplus));
  PetscCall(PetscFree(df->sk));
  PetscCall(PetscFree(df->yk));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Piecewise linear monotone target function for the Dai-Fletcher projector */
PetscReal phi(PetscReal *x, PetscInt n, PetscReal lambda, PetscReal *a, PetscReal b, PetscReal *c, PetscReal *l, PetscReal *u)
{
  PetscReal r = 0.0;
  PetscInt  i;

  for (i = 0; i < n; i++) {
    x[i] = -c[i] + lambda * a[i];
    if (x[i] > u[i]) x[i] = u[i];
    else if (x[i] < l[i]) x[i] = l[i];
    r += a[i] * x[i];
  }
  return r - b;
}

/** Modified Dai-Fletcher QP projector solves the problem:
 *
 *      minimise  0.5*x'*x - c'*x
 *      subj to   a'*x = b
 *                l \leq x \leq u
 *
 *  \param c The point to be projected onto feasible set
 */
PetscInt project(PetscInt n, PetscReal *a, PetscReal b, PetscReal *c, PetscReal *l, PetscReal *u, PetscReal *x, PetscReal *lam_ext, TAO_DF *df)
{
  PetscReal lambda, lambdal, lambdau, dlambda, lambda_new;
  PetscReal r, rl, ru, s;
  PetscInt  innerIter;
  PetscBool nonNegativeSlack = PETSC_FALSE;

  *lam_ext  = 0;
  lambda    = 0;
  dlambda   = 0.5;
  innerIter = 1;

  /*  \phi(x;lambda) := 0.5*x'*x + c'*x - lambda*(a'*x-b)
   *
   *  Optimality conditions for \phi:
   *
   *  1. lambda   <= 0
   *  2. r        <= 0
   *  3. r*lambda == 0
   */

  /* Bracketing Phase */
  r = phi(x, n, lambda, a, b, c, l, u);

  if (nonNegativeSlack) {
    /* inequality constraint, i.e., with \xi >= 0 constraint */
    if (r < TOL_R) return PETSC_SUCCESS;
  } else {
    /* equality constraint ,i.e., without \xi >= 0 constraint */
    if (PetscAbsReal(r) < TOL_R) return PETSC_SUCCESS;
  }

  if (r < 0.0) {
    lambdal = lambda;
    rl      = r;
    lambda  = lambda + dlambda;
    r       = phi(x, n, lambda, a, b, c, l, u);
    while (r < 0.0 && dlambda < BMRM_INFTY) {
      lambdal = lambda;
      s       = rl / r - 1.0;
      if (s < 0.1) s = 0.1;
      dlambda = dlambda + dlambda / s;
      lambda  = lambda + dlambda;
      rl      = r;
      r       = phi(x, n, lambda, a, b, c, l, u);
    }
    lambdau = lambda;
    ru      = r;
  } else {
    lambdau = lambda;
    ru      = r;
    lambda  = lambda - dlambda;
    r       = phi(x, n, lambda, a, b, c, l, u);
    while (r > 0.0 && dlambda > -BMRM_INFTY) {
      lambdau = lambda;
      s       = ru / r - 1.0;
      if (s < 0.1) s = 0.1;
      dlambda = dlambda + dlambda / s;
      lambda  = lambda - dlambda;
      ru      = r;
      r       = phi(x, n, lambda, a, b, c, l, u);
    }
    lambdal = lambda;
    rl      = r;
  }

  PetscCheck(PetscAbsReal(dlambda) <= BMRM_INFTY, PETSC_COMM_SELF, PETSC_ERR_PLIB, "L2N2_DaiFletcherPGM detected Infeasible QP problem!");

  if (ru == 0) return innerIter;

  /* Secant Phase */
  s       = 1.0 - rl / ru;
  dlambda = dlambda / s;
  lambda  = lambdau - dlambda;
  r       = phi(x, n, lambda, a, b, c, l, u);

  while (PetscAbsReal(r) > TOL_R && dlambda > TOL_LAM * (1.0 + PetscAbsReal(lambda)) && innerIter < df->maxProjIter) {
    innerIter++;
    if (r > 0.0) {
      if (s <= 2.0) {
        lambdau = lambda;
        ru      = r;
        s       = 1.0 - rl / ru;
        dlambda = (lambdau - lambdal) / s;
        lambda  = lambdau - dlambda;
      } else {
        s = ru / r - 1.0;
        if (s < 0.1) s = 0.1;
        dlambda    = (lambdau - lambda) / s;
        lambda_new = 0.75 * lambdal + 0.25 * lambda;
        if (lambda_new < (lambda - dlambda)) lambda_new = lambda - dlambda;
        lambdau = lambda;
        ru      = r;
        lambda  = lambda_new;
        s       = (lambdau - lambdal) / (lambdau - lambda);
      }
    } else {
      if (s >= 2.0) {
        lambdal = lambda;
        rl      = r;
        s       = 1.0 - rl / ru;
        dlambda = (lambdau - lambdal) / s;
        lambda  = lambdau - dlambda;
      } else {
        s = rl / r - 1.0;
        if (s < 0.1) s = 0.1;
        dlambda    = (lambda - lambdal) / s;
        lambda_new = 0.75 * lambdau + 0.25 * lambda;
        if (lambda_new > (lambda + dlambda)) lambda_new = lambda + dlambda;
        lambdal = lambda;
        rl      = r;
        lambda  = lambda_new;
        s       = (lambdau - lambdal) / (lambdau - lambda);
      }
    }
    r = phi(x, n, lambda, a, b, c, l, u);
  }

  *lam_ext = lambda;
  if (innerIter >= df->maxProjIter) PetscCall(PetscInfo(NULL, "WARNING: DaiFletcher max iterations\n"));
  return innerIter;
}

PetscErrorCode solve(TAO_DF *df)
{
  PetscInt    i, j, innerIter, it, it2, luv, info;
  PetscReal   gd, max, ak, bk, akold, bkold, lamnew, alpha, kktlam = 0.0, lam_ext;
  PetscReal   DELTAsv, ProdDELTAsv;
  PetscReal   c, *tempQ;
  PetscReal  *x = df->x, *a = df->a, b = df->b, *l = df->l, *u = df->u, tol = df->tol;
  PetscReal  *tempv = df->tempv, *y = df->y, *g = df->g, *d = df->d, *Qd = df->Qd;
  PetscReal  *xplus = df->xplus, *tplus = df->tplus, *sk = df->sk, *yk = df->yk;
  PetscReal **Q = df->Q, *f = df->f, *t = df->t;
  PetscInt    dim = df->dim, *ipt = df->ipt, *ipt2 = df->ipt2, *uv = df->uv;

  /* variables for the adaptive nonmonotone linesearch */
  PetscInt  L, llast;
  PetscReal fr, fbest, fv, fc, fv0;

  c = BMRM_INFTY;

  DELTAsv = EPS_SV;
  if (tol <= 1.0e-5 || dim <= 20) ProdDELTAsv = 0.0F;
  else ProdDELTAsv = EPS_SV;

  for (i = 0; i < dim; i++) tempv[i] = -x[i];

  lam_ext = 0.0;

  /* Project the initial solution */
  project(dim, a, b, tempv, l, u, x, &lam_ext, df);

  /* Compute gradient
     g = Q*x + f; */

  it = 0;
  for (i = 0; i < dim; i++) {
    if (PetscAbsReal(x[i]) > ProdDELTAsv) ipt[it++] = i;
  }

  PetscCall(PetscArrayzero(t, dim));
  for (i = 0; i < it; i++) {
    tempQ = Q[ipt[i]];
    for (j = 0; j < dim; j++) t[j] += (tempQ[j] * x[ipt[i]]);
  }
  for (i = 0; i < dim; i++) g[i] = t[i] + f[i];

  /* y = -(x_{k} - g_{k}) */
  for (i = 0; i < dim; i++) y[i] = g[i] - x[i];

  /* Project x_{k} - g_{k} */
  project(dim, a, b, y, l, u, tempv, &lam_ext, df);

  /* y = P(x_{k} - g_{k}) - x_{k} */
  max = ALPHA_MIN;
  for (i = 0; i < dim; i++) {
    y[i] = tempv[i] - x[i];
    if (PetscAbsReal(y[i]) > max) max = PetscAbsReal(y[i]);
  }

  if (max < tol * 1e-3) return PETSC_SUCCESS;

  alpha = 1.0 / max;

  /* fv0 = f(x_{0}). Recall t = Q x_{k}  */
  fv0 = 0.0;
  for (i = 0; i < dim; i++) fv0 += x[i] * (0.5 * t[i] + f[i]);

  /* adaptive nonmonotone linesearch */
  L     = 2;
  fr    = ALPHA_MAX;
  fbest = fv0;
  fc    = fv0;
  llast = 0;
  akold = bkold = 0.0;

  /*     Iterator begins     */
  for (innerIter = 1; innerIter <= df->maxPGMIter; innerIter++) {
    /* tempv = -(x_{k} - alpha*g_{k}) */
    for (i = 0; i < dim; i++) tempv[i] = alpha * g[i] - x[i];

    /* Project x_{k} - alpha*g_{k} */
    project(dim, a, b, tempv, l, u, y, &lam_ext, df);

    /* gd = \inner{d_{k}}{g_{k}}
        d = P(x_{k} - alpha*g_{k}) - x_{k}
    */
    gd = 0.0;
    for (i = 0; i < dim; i++) {
      d[i] = y[i] - x[i];
      gd += d[i] * g[i];
    }

    /* Gradient computation  */

    /* compute Qd = Q*d  or  Qd = Q*y - t depending on their sparsity */

    it = it2 = 0;
    for (i = 0; i < dim; i++) {
      if (PetscAbsReal(d[i]) > (ProdDELTAsv * 1.0e-2)) ipt[it++] = i;
    }
    for (i = 0; i < dim; i++) {
      if (PetscAbsReal(y[i]) > ProdDELTAsv) ipt2[it2++] = i;
    }

    PetscCall(PetscArrayzero(Qd, dim));
    /* compute Qd = Q*d */
    if (it < it2) {
      for (i = 0; i < it; i++) {
        tempQ = Q[ipt[i]];
        for (j = 0; j < dim; j++) Qd[j] += (tempQ[j] * d[ipt[i]]);
      }
    } else { /* compute Qd = Q*y-t */
      for (i = 0; i < it2; i++) {
        tempQ = Q[ipt2[i]];
        for (j = 0; j < dim; j++) Qd[j] += (tempQ[j] * y[ipt2[i]]);
      }
      for (j = 0; j < dim; j++) Qd[j] -= t[j];
    }

    /* ak = inner{d_{k}}{d_{k}} */
    ak = 0.0;
    for (i = 0; i < dim; i++) ak += d[i] * d[i];

    bk = 0.0;
    for (i = 0; i < dim; i++) bk += d[i] * Qd[i];

    if (bk > EPS * ak && gd < 0.0) lamnew = -gd / bk;
    else lamnew = 1.0;

    /* fv is computing f(x_{k} + d_{k}) */
    fv = 0.0;
    for (i = 0; i < dim; i++) {
      xplus[i] = x[i] + d[i];
      tplus[i] = t[i] + Qd[i];
      fv += xplus[i] * (0.5 * tplus[i] + f[i]);
    }

    /* fr is fref */
    if ((innerIter == 1 && fv >= fv0) || (innerIter > 1 && fv >= fr)) {
      fv = 0.0;
      for (i = 0; i < dim; i++) {
        xplus[i] = x[i] + lamnew * d[i];
        tplus[i] = t[i] + lamnew * Qd[i];
        fv += xplus[i] * (0.5 * tplus[i] + f[i]);
      }
    }

    for (i = 0; i < dim; i++) {
      sk[i] = xplus[i] - x[i];
      yk[i] = tplus[i] - t[i];
      x[i]  = xplus[i];
      t[i]  = tplus[i];
      g[i]  = t[i] + f[i];
    }

    /* update the line search control parameters */
    if (fv < fbest) {
      fbest = fv;
      fc    = fv;
      llast = 0;
    } else {
      fc = (fc > fv ? fc : fv);
      llast++;
      if (llast == L) {
        fr    = fc;
        fc    = fv;
        llast = 0;
      }
    }

    ak = bk = 0.0;
    for (i = 0; i < dim; i++) {
      ak += sk[i] * sk[i];
      bk += sk[i] * yk[i];
    }

    if (bk <= EPS * ak) alpha = ALPHA_MAX;
    else {
      if (bkold < EPS * akold) alpha = ak / bk;
      else alpha = (akold + ak) / (bkold + bk);

      if (alpha > ALPHA_MAX) alpha = ALPHA_MAX;
      else if (alpha < ALPHA_MIN) alpha = ALPHA_MIN;
    }

    akold = ak;
    bkold = bk;

    /* stopping criterion based on KKT conditions */
    /* at optimal, gradient of lagrangian w.r.t. x is zero */

    bk = 0.0;
    for (i = 0; i < dim; i++) bk += x[i] * x[i];

    if (PetscSqrtReal(ak) < tol * 10 * PetscSqrtReal(bk)) {
      it     = 0;
      luv    = 0;
      kktlam = 0.0;
      for (i = 0; i < dim; i++) {
        /* x[i] is active hence lagrange multipliers for box constraints
                are zero. The lagrange multiplier for ineq. const. is then
                defined as below
        */
        if ((x[i] > DELTAsv) && (x[i] < c - DELTAsv)) {
          ipt[it++] = i;
          kktlam    = kktlam - a[i] * g[i];
        } else uv[luv++] = i;
      }

      if (it == 0 && PetscSqrtReal(ak) < tol * 0.5 * PetscSqrtReal(bk)) return PETSC_SUCCESS;
      else {
        kktlam = kktlam / it;
        info   = 1;
        for (i = 0; i < it; i++) {
          if (PetscAbsReal(a[ipt[i]] * g[ipt[i]] + kktlam) > tol) {
            info = 0;
            break;
          }
        }
        if (info == 1) {
          for (i = 0; i < luv; i++) {
            if (x[uv[i]] <= DELTAsv) {
              /* x[i] == lower bound, hence, lagrange multiplier (say, beta) for lower bound may
                     not be zero. So, the gradient without beta is > 0
              */
              if (g[uv[i]] + kktlam * a[uv[i]] < -tol) {
                info = 0;
                break;
              }
            } else {
              /* x[i] == upper bound, hence, lagrange multiplier (say, eta) for upper bound may
                     not be zero. So, the gradient without eta is < 0
              */
              if (g[uv[i]] + kktlam * a[uv[i]] > tol) {
                info = 0;
                break;
              }
            }
          }
        }

        if (info == 1) return PETSC_SUCCESS;
      }
    }
  }
  return PETSC_SUCCESS;
}

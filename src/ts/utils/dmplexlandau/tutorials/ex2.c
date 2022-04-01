static char help[] = "Runaway electron model with Landau collision operator\n\n";

#include <petscdmplex.h>
#include <petsclandau.h>
#include <petscts.h>
#include <petscds.h>
#include <petscdmcomposite.h>

/* data for runaway electron model */
typedef struct REctx_struct {
  PetscErrorCode (*test)(TS, Vec, PetscInt, PetscReal, PetscBool,  LandauCtx *, struct REctx_struct *);
  PetscErrorCode (*impuritySrcRate)(PetscReal, PetscReal *, LandauCtx*);
  PetscErrorCode (*E)(Vec, Vec, PetscInt, PetscReal, LandauCtx*, PetscReal *);
  PetscReal     T_cold;        /* temperature of newly ionized electrons and impurity ions */
  PetscReal     ion_potential; /* ionization potential of impurity */
  PetscReal     Ne_ion;        /* effective number of electrons shed in ioization of impurity */
  PetscReal     Ez_initial;
  PetscReal     L;             /* inductance */
  Vec           X_0;
  PetscInt      imp_idx;       /* index for impurity ionizing sink */
  PetscReal     pulse_start;
  PetscReal     pulse_width;
  PetscReal     pulse_rate;
  PetscReal     current_rate;
  PetscInt      plotIdx;
  PetscInt      plotStep;
  PetscInt      idx; /* cache */
  PetscReal     j; /* cache */
  PetscReal     plotDt;
  PetscBool     plotting;
  PetscBool     use_spitzer_eta;
  PetscInt      print_period;
  PetscInt      grid_view_idx;
} REctx;

static const PetscReal kev_joul = 6.241506479963235e+15; /* 1/1000e */

#define RE_CUT 3.
/* < v, u_re * v * q > */
static void f0_j_re(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscReal n_e = PetscRealPart(u[0]);
  if (dim==2) {
    if (x[1] > RE_CUT || x[1] < -RE_CUT) { /* simply a cutoff for REs. v_|| > 3 v(T_e) */
      *f0 = n_e * 2.*PETSC_PI*x[0] * x[1] * constants[0]; /* n * r * v_|| * q */
    } else {
      *f0 = 0;
    }
  } else {
    if (x[2] > RE_CUT || x[2] < -RE_CUT) { /* simply a cutoff for REs. v_|| > 3 v(T_e) */
      *f0 = n_e                * x[2] * constants[0];
    } else {
      *f0 = 0;
    }
  }
}

/* sum < v, u*v*q > */
static void f0_jz_sum(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar q[], PetscScalar *f0)
{
  PetscInt ii;
  f0[0] = 0;
  if (dim==2) {
    for (ii=0;ii<Nf;ii++) f0[0] += u[ii] * 2.*PETSC_PI*x[0] * x[1] * q[ii]; /* n * r * v_|| * q * v_0 */
  } else {
    for (ii=0;ii<Nf;ii++) f0[0] += u[ii]                * x[2] * q[ii]; /* n * v_|| * q  * v_0 */
  }
}

/* < v, n_e > */
static void f0_n(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  if (dim==2) f0[0] = 2.*PETSC_PI*x[0]*u[ii];
  else {
    f0[0] =                        u[ii];
  }
}

/* < v, n_e v_|| > */
static void f0_vz(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  if (dim==2) f0[0] = u[ii] * 2.*PETSC_PI*x[0] * x[1]; /* n r v_|| */
  else {
    f0[0] =           u[ii] *                x[2]; /* n v_|| */
  }
}

/* < v, n_e (v-shift) > */
static void f0_ve_shift(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscReal vz = numConstants>0 ? PetscRealPart(constants[0]) : 0;
  if (dim==2) *f0 = u[0] * 2.*PETSC_PI*x[0] * PetscSqrtReal(x[0]*x[0] + (x[1]-vz)*(x[1]-vz));         /* n r v */
  else {
    *f0 =           u[0] *                PetscSqrtReal(x[0]*x[0] + x[1]*x[1] + (x[2]-vz)*(x[2]-vz)); /* n v */
  }
}

 /* CalculateE - Calculate the electric field  */
 /*  T        -- Electron temperature  */
 /*  n        -- Electron density  */
 /*  lnLambda --   */
 /*  eps0     --  */
 /*  E        -- output E, input \hat E */
static PetscReal CalculateE(PetscReal Tev, PetscReal n, PetscReal lnLambda, PetscReal eps0, PetscReal *E)
{
  PetscReal c,e,m;

  PetscFunctionBegin;
  c = 299792458.0;
  e = 1.602176e-19;
  m = 9.10938e-31;
  if (1) {
    double Ec, Ehat = *E, betath = PetscSqrtReal(2*Tev*e/(m*c*c)), j0 = Ehat * 7/(PetscSqrtReal(2)*2) * PetscPowReal(betath,3) * n * e * c;
    Ec = n*lnLambda*PetscPowReal(e,3) / (4*PETSC_PI*PetscPowReal(eps0,2)*m*c*c);
    *E = Ec;
    PetscPrintf(PETSC_COMM_WORLD, "CalculateE j0=%g Ec = %g\n",j0,Ec);
  } else {
    PetscReal Ed, vth;
    vth = PetscSqrtReal(8*Tev*e/(m*PETSC_PI));
    Ed =  n*lnLambda*PetscPowReal(e,3) / (4*PETSC_PI*PetscPowReal(eps0,2)*m*vth*vth);
    *E = Ed;
  }
  PetscFunctionReturn(0);
}

static PetscReal Spitzer(PetscReal m_e, PetscReal e, PetscReal Z, PetscReal epsilon0,  PetscReal lnLam, PetscReal kTe_joules)
{
  PetscReal Fz = (1+1.198*Z+0.222*Z*Z)/(1+2.966*Z+0.753*Z*Z), eta;
  eta = Fz*4./3.*PetscSqrtReal(2.*PETSC_PI)*Z*PetscSqrtReal(m_e)*PetscSqr(e)*lnLam*PetscPowReal(4*PETSC_PI*epsilon0,-2.)*PetscPowReal(kTe_joules,-1.5);
  return eta;
}

/*  */
static PetscErrorCode testNone(TS ts, Vec X, PetscInt stepi, PetscReal time, PetscBool islast, LandauCtx *ctx, REctx *rectx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

/*  */
static PetscErrorCode testSpitzer(TS ts, Vec X, PetscInt stepi, PetscReal time, PetscBool islast, LandauCtx *ctx, REctx *rectx)
{
  PetscInt          ii,nDMs;
  PetscDS           prob;
  static PetscReal  old_ratio = 1e10;
  TSConvergedReason reason;
  PetscReal         J,J_re,spit_eta,Te_kev=0,E,ratio,Z,n_e,v,v2;
  PetscScalar       user[2] = {0.,ctx->charges[0]}, q[LANDAU_MAX_SPECIES],tt[LANDAU_MAX_SPECIES],vz;
  PetscReal         dt;
  DM                pack, plexe = ctx->plex[0], plexi = (ctx->num_grids==1) ? NULL : ctx->plex[1];
  Vec               *XsubArray;

  PetscFunctionBeginUser;
  PetscCheck(ctx->num_species==2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "ctx->num_species %D != 2",ctx->num_species);
  PetscCall(VecGetDM(X, &pack));
  PetscCheck(pack,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no DM");
  PetscCall(DMCompositeGetNumberDM(pack,&nDMs));
  PetscCheck(nDMs == ctx->num_grids*ctx->batch_sz,PETSC_COMM_SELF, PETSC_ERR_PLIB, "nDMs != ctx->num_grids*ctx->batch_sz %D != %D",nDMs,ctx->num_grids*ctx->batch_sz);
  PetscCall(PetscMalloc(sizeof(*XsubArray)*nDMs, &XsubArray));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
  PetscCall(TSGetTimeStep(ts,&dt));
  /* get current for each grid */
  for (ii=0;ii<ctx->num_species;ii++) q[ii] = ctx->charges[ii];
  PetscCall(DMGetDS(plexe, &prob));
  PetscCall(PetscDSSetConstants(prob, 2, &q[0]));
  PetscCall(PetscDSSetObjective(prob, 0, &f0_jz_sum));
  PetscCall(DMPlexComputeIntegralFEM(plexe,XsubArray[ LAND_PACK_IDX(ctx->batch_view_idx,0) ],tt,NULL));
  J = -ctx->n_0*ctx->v_0*PetscRealPart(tt[0]);
  if (plexi) { // add first (only) ion
    PetscCall(DMGetDS(plexi, &prob));
    PetscCall(PetscDSSetConstants(prob, 1, &q[1]));
    PetscCall(PetscDSSetObjective(prob, 0, &f0_jz_sum));
    PetscCall(DMPlexComputeIntegralFEM(plexi,XsubArray[LAND_PACK_IDX(ctx->batch_view_idx,1)],tt,NULL));
    J += -ctx->n_0*ctx->v_0*PetscRealPart(tt[0]);
  }
  /* get N_e */
  PetscCall(DMGetDS(plexe, &prob));
  PetscCall(PetscDSSetConstants(prob, 1, user));
  PetscCall(PetscDSSetObjective(prob, 0, &f0_n));
  PetscCall(DMPlexComputeIntegralFEM(plexe,XsubArray[LAND_PACK_IDX(ctx->batch_view_idx,0)],tt,NULL));
  n_e = PetscRealPart(tt[0])*ctx->n_0;
  /* Z */
  Z = -ctx->charges[1]/ctx->charges[0];
  /* remove drift */
  if (0) {
    user[0] = 0; // electrons
    PetscCall(DMGetDS(plexe, &prob));
    PetscCall(PetscDSSetConstants(prob, 1, user));
    PetscCall(PetscDSSetObjective(prob, 0, &f0_vz));
    PetscCall(DMPlexComputeIntegralFEM(plexe,XsubArray[LAND_PACK_IDX(ctx->batch_view_idx,0)],tt,NULL));
    vz = ctx->n_0*PetscRealPart(tt[0])/n_e; /* non-dimensional */
  } else vz = 0;
  /* thermal velocity */
  PetscCall(DMGetDS(plexe, &prob));
  PetscCall(PetscDSSetConstants(prob, 1, &vz));
  PetscCall(PetscDSSetObjective(prob, 0, &f0_ve_shift));
  PetscCall(DMPlexComputeIntegralFEM(plexe,XsubArray[LAND_PACK_IDX(ctx->batch_view_idx,0)],tt,NULL));
  v = ctx->n_0*ctx->v_0*PetscRealPart(tt[0])/n_e;   /* remove number density to get velocity */
  v2 = PetscSqr(v);                                    /* use real space: m^2 / s^2 */
  Te_kev = (v2*ctx->masses[0]*PETSC_PI/8)*kev_joul;    /* temperature in kev */
  spit_eta = Spitzer(ctx->masses[0],-ctx->charges[0],Z,ctx->epsilon0,ctx->lnLam,Te_kev/kev_joul); /* kev --> J (kT) */
  if (0) {
    PetscCall(DMGetDS(plexe, &prob));
    PetscCall(PetscDSSetConstants(prob, 1, q));
    PetscCall(PetscDSSetObjective(prob, 0, &f0_j_re));
    PetscCall(DMPlexComputeIntegralFEM(plexe,XsubArray[LAND_PACK_IDX(ctx->batch_view_idx,0)],tt,NULL));
  } else tt[0] = 0;
  J_re = -ctx->n_0*ctx->v_0*PetscRealPart(tt[0]);
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
  PetscCall(PetscFree(XsubArray));

  if (rectx->use_spitzer_eta) {
    E = ctx->Ez = spit_eta*(rectx->j-J_re);
  } else {
    E = ctx->Ez; /* keep real E */
    rectx->j = J; /* cache */
  }

  ratio = E/J/spit_eta;
  if (stepi>10 && !rectx->use_spitzer_eta && (
        (old_ratio-ratio < 1.e-6))) {
    rectx->pulse_start = time + 0.98*dt;
    rectx->use_spitzer_eta = PETSC_TRUE;
  }
  PetscCall(TSGetConvergedReason(ts,&reason));
  PetscCall(TSGetConvergedReason(ts,&reason));
  if ((rectx->plotting) || stepi == 0 || reason || rectx->pulse_start == time + 0.98*dt) {
    PetscCall(PetscPrintf(ctx->comm, "testSpitzer: %4D) time=%11.4e n_e= %10.3e E= %10.3e J= %10.3e J_re= %10.3e %.3g%% Te_kev= %10.3e Z_eff=%g E/J to eta ratio= %g (diff=%g) %s %s spit_eta=%g\n",stepi,time,n_e/ctx->n_0,ctx->Ez,J,J_re,100*J_re/J, Te_kev,Z,ratio,old_ratio-ratio, rectx->use_spitzer_eta ? "using Spitzer eta*J E" : "constant E",rectx->pulse_start != time + 0.98*dt ? "normal" : "transition",spit_eta));
    PetscCheckFalse(rectx->pulse_start == time + 0.98*dt,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Spitzer complete ratio=%g",ratio);
  }
  old_ratio = ratio;
  PetscFunctionReturn(0);
}

static const double ppp = 2;
static void f0_0_diff_lp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  LandauCtx       *ctx = (LandauCtx *)constants;
  REctx           *rectx = (REctx*)ctx->data;
  PetscInt        ii = rectx->idx, i;
  const PetscReal kT_m = ctx->k*ctx->thermal_temps[ii]/ctx->masses[ii]; /* kT/m */
  const PetscReal n = ctx->n[ii];
  PetscReal       diff, f_maxwell, v2 = 0, theta = 2*kT_m/(ctx->v_0*ctx->v_0); /* theta = 2kT/mc^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  f_maxwell = n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta));
  diff = 2.*PETSC_PI*x[0]*(PetscRealPart(u[ii]) - f_maxwell);
  f0[0] = PetscPowReal(diff,ppp);
}
static void f0_0_maxwellian_lp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  LandauCtx       *ctx = (LandauCtx *)constants;
  REctx           *rectx = (REctx*)ctx->data;
  PetscInt        ii = rectx->idx, i;
  const PetscReal kT_m = ctx->k*ctx->thermal_temps[ii]/ctx->masses[ii]; /* kT/m */
  const PetscReal n = ctx->n[ii];
  PetscReal       f_maxwell, v2 = 0, theta = 2*kT_m/(ctx->v_0*ctx->v_0); /* theta = 2kT/mc^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  f_maxwell = 2.*PETSC_PI*x[0] * n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta));
  f0[0] = PetscPowReal(f_maxwell,ppp);
}

/*  */
static PetscErrorCode testStable(TS ts, Vec X, PetscInt stepi, PetscReal time, PetscBool islast, LandauCtx *ctx, REctx *rectx)
{
  PetscDS           prob;
  Vec               X2;
  PetscReal         ediff,idiff=0,lpm0,lpm1=1;
  PetscScalar       tt[LANDAU_MAX_SPECIES];
  DM                dm, plex = ctx->plex[0];

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(X, &dm));
  PetscCall(DMGetDS(plex, &prob));
  PetscCall(VecDuplicate(X,&X2));
  PetscCall(VecCopy(X,X2));
  if (!rectx->X_0) {
    PetscCall(VecDuplicate(X,&rectx->X_0));
    PetscCall(VecCopy(X,rectx->X_0));
  }
  PetscCall(VecAXPY(X,-1.0,rectx->X_0));
  PetscCall(PetscDSSetConstants(prob, sizeof(LandauCtx)/sizeof(PetscScalar), (PetscScalar*)ctx));
  rectx->idx = 0;
  PetscCall(PetscDSSetObjective(prob, 0, &f0_0_diff_lp));
  PetscCall(DMPlexComputeIntegralFEM(plex,X2,tt,NULL));
  ediff = PetscPowReal(PetscRealPart(tt[0]),1./ppp);
  PetscCall(PetscDSSetObjective(prob, 0, &f0_0_maxwellian_lp));
  PetscCall(DMPlexComputeIntegralFEM(plex,X2,tt,NULL));
  lpm0 = PetscPowReal(PetscRealPart(tt[0]),1./ppp);
  if (ctx->num_species>1) {
    rectx->idx = 1;
    PetscCall(PetscDSSetObjective(prob, 0, &f0_0_diff_lp));
    PetscCall(DMPlexComputeIntegralFEM(plex,X2,tt,NULL));
    idiff = PetscPowReal(PetscRealPart(tt[0]),1./ppp);
    PetscCall(PetscDSSetObjective(prob, 0, &f0_0_maxwellian_lp));
    PetscCall(DMPlexComputeIntegralFEM(plex,X2,tt,NULL));
    lpm1 = PetscPowReal(PetscRealPart(tt[0]),1./ppp);
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s %D) time=%10.3e n-%d norm electrons/max=%20.13e ions/max=%20.13e\n", "----",stepi,time,(int)ppp,ediff/lpm0,idiff/lpm1));
  /* view */
  PetscCall(VecCopy(X2,X));
  PetscCall(VecDestroy(&X2));
  if (islast) {
    PetscCall(VecDestroy(&rectx->X_0));
    rectx->X_0 = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EInduction(Vec X, Vec X_t, PetscInt step, PetscReal time, LandauCtx *ctx, PetscReal *a_E)
{
  REctx             *rectx = (REctx*)ctx->data;
  PetscInt          ii;
  DM                dm,plex;
  PetscScalar       tt[LANDAU_MAX_SPECIES], qv0[LANDAU_MAX_SPECIES];
  PetscReal         dJ_dt;
  PetscDS           prob;

  PetscFunctionBeginUser;
  for (ii=0;ii<ctx->num_species;ii++) qv0[ii] = ctx->charges[ii]*ctx->v_0;
  PetscCall(VecGetDM(X, &dm));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  /* get d current / dt */
  PetscCall(PetscDSSetConstants(prob, ctx->num_species, qv0));
  PetscCall(PetscDSSetObjective(prob, 0, &f0_jz_sum));
  PetscCheck(X_t,PETSC_COMM_SELF, PETSC_ERR_PLIB, "X_t");
  PetscCall(DMPlexComputeIntegralFEM(plex,X_t,tt,NULL));
  dJ_dt = -ctx->n_0*PetscRealPart(tt[0])/ctx->t_0;
  /* E induction */
  *a_E = -rectx->L*dJ_dt + rectx->Ez_initial;
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

static PetscErrorCode EConstant(Vec X,  Vec X_t, PetscInt step, PetscReal time, LandauCtx *ctx, PetscReal *a_E)
{
  PetscFunctionBeginUser;
  *a_E = ctx->Ez;
  PetscFunctionReturn(0);
}

static PetscErrorCode ENone(Vec X,  Vec X_t, PetscInt step, PetscReal time, LandauCtx *ctx, PetscReal *a_E)
{
  PetscFunctionBeginUser;
  *a_E = 0;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormSource - Evaluates source terms F(t).

   Input Parameters:
.  ts - the TS context
.  time -
.  X_dummmy - input vector
.  dummy - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
static PetscErrorCode FormSource(TS ts, PetscReal ftime, Vec X_dummmy, Vec F, void *dummy)
{
  PetscReal      new_imp_rate;
  LandauCtx      *ctx;
  DM             pack;
  REctx          *rectx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts,&pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  rectx = (REctx*)ctx->data;
  /* check for impurities */
  PetscCall(rectx->impuritySrcRate(ftime,&new_imp_rate,ctx));
  if (new_imp_rate != 0) {
    if (new_imp_rate != rectx->current_rate) {
      PetscInt       ii;
      PetscReal      dne_dt,dni_dt,tilda_ns[LANDAU_MAX_SPECIES],temps[LANDAU_MAX_SPECIES];
      Vec            globFarray[LANDAU_MAX_GRIDS*LANDAU_MAX_BATCH_SZ];
      rectx->current_rate = new_imp_rate;
      for (ii=1;ii<LANDAU_MAX_SPECIES;ii++) tilda_ns[ii] = 0;
      for (ii=1;ii<LANDAU_MAX_SPECIES;ii++)    temps[ii] = 1;
      dni_dt = new_imp_rate               /* *ctx->t_0 */; /* fully ionized immediately, no normalize, stay in non-dim */
      dne_dt = new_imp_rate*rectx->Ne_ion /* *ctx->t_0 */;
      tilda_ns[0] = dne_dt;        tilda_ns[rectx->imp_idx] = dni_dt;
      temps[0]    = rectx->T_cold;    temps[rectx->imp_idx] = rectx->T_cold;
      PetscCall(PetscInfo(ctx->plex[0], "\tHave new_imp_rate= %10.3e time= %10.3e de/dt= %10.3e di/dt= %10.3e ***\n",new_imp_rate,ftime,dne_dt,dni_dt));
      PetscCall(DMCompositeGetAccessArray(pack, F, ctx->num_grids*ctx->batch_sz, NULL, globFarray));
      for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
        /* add it */
        PetscCall(DMPlexLandauAddMaxwellians(ctx->plex[grid],globFarray[ LAND_PACK_IDX(0,grid) ],ftime,temps,tilda_ns,grid,0,ctx));
        PetscCall(VecViewFromOptions(globFarray[ LAND_PACK_IDX(0,grid) ],NULL,"-vec_view_sources"));
      }
      // Does DMCompositeRestoreAccessArray copy the data back? (no)
      PetscCall(DMCompositeRestoreAccessArray(pack, F, ctx->num_grids*ctx->batch_sz, NULL, globFarray));
    }
  } else {
    PetscCall(VecZeroEntries(F));
    rectx->current_rate = 0;
  }
  PetscFunctionReturn(0);
}
PetscErrorCode Monitor(TS ts, PetscInt stepi, PetscReal time, Vec X, void *actx)
{
  LandauCtx         *ctx = (LandauCtx*) actx;   /* user-defined application context */
  REctx             *rectx = (REctx*)ctx->data;
  DM                pack;
  Vec               globXArray[LANDAU_MAX_GRIDS*LANDAU_MAX_BATCH_SZ];
  TSConvergedReason reason;
  PetscFunctionBeginUser;
  PetscCall(VecGetDM(X, &pack));
  PetscCall(DMCompositeGetAccessArray(pack, X, ctx->num_grids*ctx->batch_sz, NULL, globXArray));
  if (stepi > rectx->plotStep && rectx->plotting) {
    rectx->plotting = PETSC_FALSE; /* was doing diagnostics, now done */
    rectx->plotIdx++;
  }
  /* view */
  PetscCall(TSGetConvergedReason(ts,&reason));
  if (time/rectx->plotDt >= (PetscReal)rectx->plotIdx || reason) {
    if ((reason || stepi==0 || rectx->plotIdx%rectx->print_period==0) && ctx->verbose > 0) {
      /* print norms */
      PetscCall(DMPlexLandauPrintNorms(X, stepi));
    }
    if (!rectx->plotting) { /* first step of possible backtracks */
      rectx->plotting = PETSC_TRUE;
      /* diagnostics + change E field with Sptizer (not just a monitor) */
      PetscCall(rectx->test(ts,X,stepi,time,reason ? PETSC_TRUE : PETSC_FALSE, ctx, rectx));
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "\t\t ERROR SKIP test spit ------\n");
      rectx->plotting = PETSC_TRUE;
    }
    PetscCall(PetscObjectSetName((PetscObject) globXArray[ LAND_PACK_IDX(ctx->batch_view_idx,rectx->grid_view_idx) ], rectx->grid_view_idx==0 ? "ue" : "ui"));
    /* view, overwrite step when back tracked */
    PetscCall(DMSetOutputSequenceNumber(pack, rectx->plotIdx, time*ctx->t_0));
    PetscCall(VecViewFromOptions(globXArray[ LAND_PACK_IDX(ctx->batch_view_idx, rectx->grid_view_idx) ],NULL,"-vec_view"));

    rectx->plotStep = stepi;
  } else {
    if (rectx->plotting) PetscPrintf(PETSC_COMM_WORLD," ERROR rectx->plotting=%D step %D\n",rectx->plotting,stepi);
    /* diagnostics + change E field with Sptizer (not just a monitor) - can we lag this? */
    PetscCall(rectx->test(ts,X,stepi,time,reason ? PETSC_TRUE : PETSC_FALSE, ctx, rectx));
  }
  /* parallel check that only works of all batches are identical */
  if (reason && ctx->verbose > 3) {
    PetscReal    val,rval;
    PetscMPIInt  rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      PetscInt nerrors=0;
      for (PetscInt i=0; i<ctx->batch_sz;i++) {
        PetscCall(VecNorm(globXArray[ LAND_PACK_IDX(i,grid) ],NORM_2,&val));
        if (i==0) rval = val;
        else if ((val=PetscAbs(val-rval)/rval) > 1000*PETSC_MACHINE_EPSILON) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, " [%D] Warning %D.%D) diff = %2.15e\n",rank,grid,i,val));
          nerrors++;
        }
      }
      if (nerrors) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, " ***** [%D] ERROR max %D errors\n",rank,nerrors));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%D] %D) batch consistency check OK\n",rank,grid));
      }
    }
  }
  rectx->idx = 0;
  PetscCall(DMCompositeRestoreAccessArray(pack, X, ctx->num_grids*ctx->batch_sz, NULL, globXArray));
  PetscFunctionReturn(0);
}

PetscErrorCode PreStep(TS ts)
{
  LandauCtx      *ctx;
  REctx          *rectx;
  DM             dm;
  PetscInt       stepi;
  PetscReal      time;
  Vec            X;

  PetscFunctionBeginUser;
  /* not used */
  PetscCall(TSGetDM(ts,&dm));
  PetscCall(TSGetTime(ts,&time));
  PetscCall(TSGetSolution(ts,&X));
  PetscCall(DMGetApplicationContext(dm, &ctx));
  rectx = (REctx*)ctx->data;
  PetscCall(TSGetStepNumber(ts, &stepi));
  /* update E */
  PetscCall(rectx->E(X, NULL, stepi, time, ctx, &ctx->Ez));
  PetscFunctionReturn(0);
}

/* model for source of non-ionized impurities, profile provided by model, in du/dt form in normalized units (tricky because n_0 is normalized with electrons) */
static PetscErrorCode stepSrc(PetscReal time, PetscReal *rho, LandauCtx *ctx)
{
  REctx         *rectx = (REctx*)ctx->data;

  PetscFunctionBeginUser;
  if (time >= rectx->pulse_start) *rho = rectx->pulse_rate;
  else *rho = 0.;
  PetscFunctionReturn(0);
}
static PetscErrorCode zeroSrc(PetscReal time, PetscReal *rho, LandauCtx *ctx)
{
  PetscFunctionBeginUser;
  *rho = 0.;
  PetscFunctionReturn(0);
}
static PetscErrorCode pulseSrc(PetscReal time, PetscReal *rho, LandauCtx *ctx)
{
  REctx *rectx = (REctx*)ctx->data;

  PetscFunctionBeginUser;
  PetscCheck(rectx->pulse_start != PETSC_MAX_REAL,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"'-ex2_pulse_start_time X' must be used with '-ex2_impurity_source_type pulse'");
  if (time < rectx->pulse_start || time > rectx->pulse_start + 3*rectx->pulse_width) *rho = 0;
  /* else if (0) { */
  /*   double t = time - rectx->pulse_start, start = rectx->pulse_width, stop = 2*rectx->pulse_width, cycle = 3*rectx->pulse_width, steep = 5, xi = 0.75 - (stop - start)/(2* cycle); */
  /*   *rho = rectx->pulse_rate * (cycle / (stop - start)) / (1 + PetscExpReal(steep*(PetscSinReal(2*PETSC_PI*((t - start)/cycle + xi)) - PetscSinReal(2*PETSC_PI*xi)))); */
  /* } else if (0) { */
  /*   double x = 2*(time - rectx->pulse_start)/(3*rectx->pulse_width) - 1; */
  /*   if (x==1 || x==-1) *rho = 0; */
  /*   else *rho = rectx->pulse_rate * PetscExpReal(-1/(1-x*x)); */
  /* } */
  else {
    double x = PetscSinReal((time-rectx->pulse_start)/(3*rectx->pulse_width)*2*PETSC_PI - PETSC_PI/2) + 1; /* 0:2, integrates to 1.0 */
    *rho = rectx->pulse_rate * x / (3*rectx->pulse_width);
    if (!rectx->use_spitzer_eta) rectx->use_spitzer_eta = PETSC_TRUE; /* use it next time */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessREOptions"
static PetscErrorCode ProcessREOptions(REctx *rectx, const LandauCtx *ctx, DM dm, const char prefix[])
{
  PetscErrorCode    ierr;
  PetscFunctionList plist = NULL, testlist = NULL, elist = NULL;
  char              pname[256],testname[256],ename[256];
  DM                dm_dummy;
  PetscBool         Connor_E = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(PETSC_COMM_WORLD,&dm_dummy));
  rectx->Ne_ion = 1;                 /* number of electrons given up by impurity ion */
  rectx->T_cold = .005;              /* kev */
  rectx->ion_potential = 15;         /* ev */
  rectx->L = 2;
  rectx->X_0 = NULL;
  rectx->imp_idx = ctx->num_species - 1; /* default ionized impurity as last one */
  rectx->pulse_start = PETSC_MAX_REAL;
  rectx->pulse_width = 1;
  rectx->plotStep = PETSC_MAX_INT;
  rectx->pulse_rate = 1.e-1;
  rectx->current_rate = 0;
  rectx->plotIdx = 0;
  rectx->j = 0;
  rectx->plotDt = 1.0;
  rectx->plotting = PETSC_FALSE;
  rectx->use_spitzer_eta = PETSC_FALSE;
  rectx->idx = 0;
  rectx->print_period = 10;
  rectx->grid_view_idx = 0;
  /* Register the available impurity sources */
  PetscCall(PetscFunctionListAdd(&plist,"step",&stepSrc));
  PetscCall(PetscFunctionListAdd(&plist,"none",&zeroSrc));
  PetscCall(PetscFunctionListAdd(&plist,"pulse",&pulseSrc));
  PetscCall(PetscStrcpy(pname,"none"));
  PetscCall(PetscFunctionListAdd(&testlist,"none",&testNone));
  PetscCall(PetscFunctionListAdd(&testlist,"spitzer",&testSpitzer));
  PetscCall(PetscFunctionListAdd(&testlist,"stable",&testStable));
  PetscCall(PetscStrcpy(testname,"none"));
  PetscCall(PetscFunctionListAdd(&elist,"none",&ENone));
  PetscCall(PetscFunctionListAdd(&elist,"induction",&EInduction));
  PetscCall(PetscFunctionListAdd(&elist,"constant",&EConstant));
  PetscCall(PetscStrcpy(ename,"constant"));

  ierr = PetscOptionsBegin(PETSC_COMM_SELF, prefix, "Options for Runaway/seed electron model", "none");PetscCall(ierr);
  PetscCall(PetscOptionsReal("-ex2_plot_dt", "Plotting interval", "ex2.c", rectx->plotDt, &rectx->plotDt, NULL));
  if (rectx->plotDt < 0) rectx->plotDt = 1e30;
  if (rectx->plotDt == 0) rectx->plotDt = 1e-30;
  PetscCall(PetscOptionsInt("-ex2_print_period", "Plotting interval", "ex2.c", rectx->print_period, &rectx->print_period, NULL));
  PetscCall(PetscOptionsInt("-ex2_grid_view_idx", "grid_view_idx", "ex2.c", rectx->grid_view_idx, &rectx->grid_view_idx, NULL));
  PetscCheck(rectx->grid_view_idx < ctx->num_grids,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"rectx->grid_view_idx (%D) >= ctx->num_grids (%D)",rectx->imp_idx,ctx->num_grids);
  PetscCall(PetscOptionsFList("-ex2_impurity_source_type","Name of impurity source to run","",plist,pname,pname,sizeof(pname),NULL));
  PetscCall(PetscOptionsFList("-ex2_test_type","Name of test to run","",testlist,testname,testname,sizeof(testname),NULL));
  PetscCall(PetscOptionsInt("-ex2_impurity_index", "index of sink for impurities", "none", rectx->imp_idx, &rectx->imp_idx, NULL));
  PetscCheckFalse((rectx->imp_idx >= ctx->num_species || rectx->imp_idx < 1) && ctx->num_species > 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"index of sink for impurities ions is out of range (%D), must be > 0 && < NS",rectx->imp_idx);
  PetscCall(PetscOptionsFList("-ex2_e_field_type","Electric field type","",elist,ename,ename,sizeof(ename),NULL));
  rectx->Ne_ion = -ctx->charges[rectx->imp_idx]/ctx->charges[0];
  PetscCall(PetscOptionsReal("-ex2_t_cold","Temperature of cold electron and ions after ionization in keV","none",rectx->T_cold,&rectx->T_cold, NULL));
  PetscCall(PetscOptionsReal("-ex2_pulse_start_time","Time at which pulse happens for 'pulse' source","none",rectx->pulse_start,&rectx->pulse_start, NULL));
  PetscCall(PetscOptionsReal("-ex2_pulse_width_time","Width of pulse 'pulse' source","none",rectx->pulse_width,&rectx->pulse_width, NULL));
  PetscCall(PetscOptionsReal("-ex2_pulse_rate","Number density of pulse for 'pulse' source","none",rectx->pulse_rate,&rectx->pulse_rate, NULL));
  rectx->T_cold *= 1.16e7; /* convert to Kelvin */
  PetscCall(PetscOptionsReal("-ex2_ion_potential","Potential to ionize impurity (should be array) in ev","none",rectx->ion_potential,&rectx->ion_potential, NULL));
  PetscCall(PetscOptionsReal("-ex2_inductance","Inductance E feild","none",rectx->L,&rectx->L, NULL));
  PetscCall(PetscOptionsBool("-ex2_connor_e_field_units","Scale Ex but Connor-Hastie E_c","none",Connor_E,&Connor_E, NULL));
  PetscCall(PetscInfo(dm_dummy, "Num electrons from ions=%g, T_cold=%10.3e, ion potential=%10.3e, E_z=%10.3e v_0=%10.3e\n",rectx->Ne_ion,rectx->T_cold,rectx->ion_potential,ctx->Ez,ctx->v_0));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  /* get impurity source rate function */
  PetscCall(PetscFunctionListFind(plist,pname,&rectx->impuritySrcRate));
  PetscCheck(rectx->impuritySrcRate,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"No impurity source function found '%s'",pname);
  PetscCall(PetscFunctionListFind(testlist,testname,&rectx->test));
  PetscCheck(rectx->test,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"No test found '%s'",testname);
  PetscCall(PetscFunctionListFind(elist,ename,&rectx->E));
  PetscCheck(rectx->E,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"No E field function found '%s'",ename);
  PetscCall(PetscFunctionListDestroy(&plist));
  PetscCall(PetscFunctionListDestroy(&testlist));
  PetscCall(PetscFunctionListDestroy(&elist));

  /* convert E from Connor-Hastie E_c units to real if doing Spitzer E */
  if (Connor_E) {
    PetscReal E = ctx->Ez, Tev = ctx->thermal_temps[0]*8.621738e-5, n = ctx->n_0*ctx->n[0];
    CalculateE(Tev, n, ctx->lnLam, ctx->epsilon0, &E);
    ((LandauCtx*)ctx)->Ez *= E;
  }
  PetscCall(DMDestroy(&dm_dummy));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             pack;
  Vec            X,*XsubArray;
  PetscInt       dim = 2, nDMs;
  TS             ts;
  Mat            J;
  PetscDS        prob;
  LandauCtx      *ctx;
  REctx          *rectx;
#if defined PETSC_USE_LOG
  PetscLogStage  stage;
#endif
  PetscMPIInt    rank;
#if defined(PETSC_HAVE_THREADSAFETY)
  double         starttime, endtime;
#endif
  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank) { /* turn off output stuff for duplicate runs */
    PetscCall(PetscOptionsClearValue(NULL,"-dm_view"));
    PetscCall(PetscOptionsClearValue(NULL,"-vec_view"));
    PetscCall(PetscOptionsClearValue(NULL,"-dm_view_diff"));
    PetscCall(PetscOptionsClearValue(NULL,"-vec_view_diff"));
    PetscCall(PetscOptionsClearValue(NULL,"-dm_view_sources"));
    PetscCall(PetscOptionsClearValue(NULL,"-vec_view_0"));
    PetscCall(PetscOptionsClearValue(NULL,"-dm_view_0"));
    PetscCall(PetscOptionsClearValue(NULL,"-vec_view_sources"));
    PetscCall(PetscOptionsClearValue(NULL,"-info")); /* this does not work */
  }
  PetscCall(PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL));
  /* Create a mesh */
  PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_WORLD, dim, "", &X, &J, &pack));
  PetscCall(DMCompositeGetNumberDM(pack,&nDMs));
  PetscCall(PetscMalloc(sizeof(*XsubArray)*nDMs, &XsubArray));
  PetscCall(PetscObjectSetName((PetscObject)J, "Jacobian"));
  PetscCall(PetscObjectSetName((PetscObject)X, "f"));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCall(DMSetUp(pack));
  /* context */
  PetscCall(PetscNew(&rectx));
  ctx->data = rectx;
  PetscCall(ProcessREOptions(rectx,ctx,pack,""));
  PetscCall(DMGetDS(pack, &prob));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
  PetscCall(PetscObjectSetName((PetscObject) XsubArray[ LAND_PACK_IDX(ctx->batch_view_idx, rectx->grid_view_idx) ], rectx->grid_view_idx==0 ? "ue" : "ui"));
  PetscCall(DMViewFromOptions(ctx->plex[rectx->grid_view_idx],NULL,"-dm_view"));
  PetscCall(DMViewFromOptions(ctx->plex[rectx->grid_view_idx], NULL,"-dm_view_0"));
  PetscCall(VecViewFromOptions(XsubArray[ LAND_PACK_IDX(ctx->batch_view_idx,rectx->grid_view_idx) ], NULL,"-vec_view_0")); // initial condition (monitor plots after step)
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
  PetscCall(PetscFree(XsubArray));
  PetscCall(VecViewFromOptions(X, NULL,"-vec_view_global")); // initial condition (monitor plots after step)
  PetscCall(DMSetOutputSequenceNumber(pack, 0, 0.0));
  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_SELF,&ts));
  PetscCall(TSSetDM(ts,pack));
  PetscCall(TSSetIFunction(ts,NULL,DMPlexLandauIFunction,NULL));
  PetscCall(TSSetIJacobian(ts,J,J,DMPlexLandauIJacobian,NULL));
  PetscCall(TSSetRHSFunction(ts,NULL,FormSource,NULL));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetSolution(ts,X));
  PetscCall(TSSetApplicationContext(ts, ctx));
  PetscCall(TSMonitorSet(ts,Monitor,ctx,NULL));
  PetscCall(TSSetPreStep(ts,PreStep));
  rectx->Ez_initial = ctx->Ez;       /* cache for induction caclulation - applied E field */
  if (1) { /* warm up an test just DMPlexLandauIJacobian */
    Vec           vec;
    PetscInt      nsteps;
    PetscReal     dt;
    PetscCall(PetscLogStageRegister("Warmup", &stage));
    PetscCall(PetscLogStagePush(stage));
    PetscCall(VecDuplicate(X,&vec));
    PetscCall(VecCopy(X,vec));
    PetscCall(TSGetMaxSteps(ts,&nsteps));
    PetscCall(TSGetTimeStep(ts,&dt));
    PetscCall(TSSetMaxSteps(ts,1));
    PetscCall(TSSolve(ts,X));
    PetscCall(TSSetMaxSteps(ts,nsteps));
    PetscCall(TSSetStepNumber(ts,0));
    PetscCall(TSSetTime(ts,0));
    PetscCall(TSSetTimeStep(ts,dt));
    rectx->plotIdx = 0;
    rectx->plotting = PETSC_FALSE;
    PetscCall(PetscLogStagePop());
    PetscCall(VecCopy(vec,X));
    PetscCall(VecDestroy(&vec));
  }
  /* go */
  PetscCall(PetscLogStageRegister("Solve", &stage));
  ctx->stage = 0; // lets not use this stage
#if defined(PETSC_HAVE_THREADSAFETY)
  ctx->stage = 1; // not set with thread safety
#endif
  PetscCall(TSSetSolution(ts,X));
  PetscCall(PetscLogStagePush(stage));
#if defined(PETSC_HAVE_THREADSAFETY)
  starttime = MPI_Wtime();
#endif
  PetscCall(TSSolve(ts,X));
  PetscCall(PetscLogStagePop());
#if defined(PETSC_HAVE_THREADSAFETY)
  endtime = MPI_Wtime();
  ctx->times[LANDAU_EX2_TSSOLVE] += (endtime - starttime);
#endif
  PetscCall(VecViewFromOptions(X, NULL,"-vec_view_global"));
  /* clean up */
  PetscCall(DMPlexLandauDestroyVelocitySpace(&pack));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFree(rectx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: p4est !complex double defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex2_0.out
    args: -dm_landau_num_species_grid 1,1 -dm_landau_Ez 0 -petscspace_degree 3 -petscspace_poly_tensor 1 -dm_landau_type p4est -dm_landau_ion_masses 2 -dm_landau_ion_charges 1 -dm_landau_thermal_temps 5,5 -dm_landau_n 2,2 -dm_landau_n_0 5e19 -ts_monitor -snes_rtol 1.e-10 -snes_stol 1.e-14 -snes_monitor -snes_converged_reason -snes_max_it 10 -ts_type arkimex -ts_arkimex_type 1bee -ts_max_snes_failures -1 -ts_rtol 1e-3 -ts_dt 1.e-1 -ts_max_time 1 -ts_adapt_clip .5,1.25 -ts_max_steps 2 -ts_adapt_scale_solve_failed 0.75 -ts_adapt_time_step_increase_delay 5 -ksp_type tfqmr -pc_type jacobi -dm_landau_amr_levels_max 2,2 -ex2_impurity_source_type pulse -ex2_pulse_start_time 1e-1 -ex2_pulse_width_time 10 -ex2_pulse_rate 1e-2 -ex2_t_cold .05 -ex2_plot_dt 1e-1 -dm_refine 0 -dm_landau_gpu_assembly true -dm_landau_batch_size 2
    test:
      suffix: cpu
      args: -dm_landau_device_type cpu -ksp_pc_side right
    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -ksp_pc_side right
    test:
      suffix: cuda
      requires: cuda
      args: -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda -mat_cusparse_use_cpu_solve -ksp_pc_side right
    test:
      suffix: kokkos_batch
      requires: kokkos_kernels
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -ksp_type preonly -pc_type bjkokkos -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi
    test:
      suffix: kokkos_batch_coo
      requires: kokkos_kernels
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -ksp_type preonly -pc_type bjkokkos -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi -dm_landau_coo_assembly

TEST*/

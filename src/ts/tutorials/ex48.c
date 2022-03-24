static char help[] = "Evolution of magnetic islands.\n\
The aim of this model is to self-consistently study the interaction between the tearing mode and small scale drift-wave turbulence.\n\n\n";

/*F
This is a three field model for the density $\tilde n$, vorticity $\tilde\Omega$, and magnetic flux $\tilde\psi$, using auxiliary variables potential $\tilde\phi$ and current $j_z$.
\begin{equation}
  \begin{aligned}
    \partial_t \tilde n       &= \left\{ \tilde n, \tilde\phi \right\} + \beta \left\{ j_z, \tilde\psi \right\} + \left\{ \ln n_0, \tilde\phi \right\} + \mu \nabla^2_\perp \tilde n \\
  \partial_t \tilde\Omega   &= \left\{ \tilde\Omega, \tilde\phi \right\} + \beta \left\{ j_z, \tilde\psi \right\} + \mu \nabla^2_\perp \tilde\Omega \\
  \partial_t \tilde\psi     &= \left\{ \psi_0 + \tilde\psi, \tilde\phi - \tilde n \right\} - \left\{ \ln n_0, \tilde\psi \right\} + \frac{\eta}{\beta} \nabla^2_\perp \tilde\psi \\
  \nabla^2_\perp\tilde\phi        &= \tilde\Omega \\
  j_z  &= -\nabla^2_\perp  \left(\tilde\psi + \psi_0  \right)\\
  \end{aligned}
\end{equation}
F*/

#include <petscdmplex.h>
#include <petscts.h>
#include <petscds.h>

typedef struct {
  PetscInt       debug;             /* The debugging level */
  PetscBool      plotRef;           /* Plot the reference fields */
  PetscReal      lower[3], upper[3];
  /* Problem definition */
  PetscErrorCode (**initialFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscReal      mu, eta, beta;
  PetscReal      a,b,Jo,Jop,m,ke,kx,ky,DeltaPrime,eps;
  /* solver */
  PetscBool      implicit;
} AppCtx;

static AppCtx *s_ctx;

static PetscScalar poissonBracket(PetscInt dim, const PetscScalar df[], const PetscScalar dg[])
{
  PetscScalar ret = df[0]*dg[1] - df[1]*dg[0];
  return ret;
}

enum field_idx {DENSITY,OMEGA,PSI,PHI,JZ};

static void f0_n(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar *pnDer   = &u_x[uOff_x[DENSITY]];
  const PetscScalar *ppsiDer = &u_x[uOff_x[PSI]];
  const PetscScalar *pphiDer = &u_x[uOff_x[PHI]];
  const PetscScalar *jzDer   = &u_x[uOff_x[JZ]];
  const PetscScalar *logRefDenDer = &a_x[aOff_x[DENSITY]];
  f0[0] += - poissonBracket(dim,pnDer, pphiDer) - s_ctx->beta*poissonBracket(dim,jzDer, ppsiDer) - poissonBracket(dim,logRefDenDer, pphiDer);
  if (u_t) f0[0] += u_t[DENSITY];
}

static void f1_n(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *pnDer = &u_x[uOff_x[DENSITY]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = -s_ctx->mu*pnDer[d];
}

static void f0_Omega(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar *pOmegaDer = &u_x[uOff_x[OMEGA]];
  const PetscScalar *ppsiDer   = &u_x[uOff_x[PSI]];
  const PetscScalar *pphiDer   = &u_x[uOff_x[PHI]];
  const PetscScalar *jzDer     = &u_x[uOff_x[JZ]];

  f0[0] += - poissonBracket(dim,pOmegaDer, pphiDer) - s_ctx->beta*poissonBracket(dim,jzDer, ppsiDer);
  if (u_t) f0[0] += u_t[OMEGA];
}

static void f1_Omega(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *pOmegaDer = &u_x[uOff_x[OMEGA]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = -s_ctx->mu*pOmegaDer[d];
}

static void f0_psi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar *pnDer     = &u_x[uOff_x[DENSITY]];
  const PetscScalar *ppsiDer   = &u_x[uOff_x[PSI]];
  const PetscScalar *pphiDer   = &u_x[uOff_x[PHI]];
  const PetscScalar *refPsiDer = &a_x[aOff_x[PSI]];
  const PetscScalar *logRefDenDer= &a_x[aOff_x[DENSITY]];
  PetscScalar       psiDer[3];
  PetscScalar       phi_n_Der[3];
  PetscInt          d;
  if (dim < 2) {MPI_Abort(MPI_COMM_WORLD,1); return;} /* this is needed so that the clang static analyzer does not generate a warning about variables used by not set */
  for (d = 0; d < dim; ++d) {
    psiDer[d]    = refPsiDer[d] + ppsiDer[d];
    phi_n_Der[d] = pphiDer[d]   - pnDer[d];
  }
  f0[0] = - poissonBracket(dim,psiDer, phi_n_Der) + poissonBracket(dim,logRefDenDer, ppsiDer);
  if (u_t) f0[0] += u_t[PSI];
}

static void f1_psi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *ppsi = &u_x[uOff_x[PSI]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = -(s_ctx->eta/s_ctx->beta)*ppsi[d];
}

static void f0_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -u[uOff[OMEGA]];
}

static void f1_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *pphi = &u_x[uOff_x[PHI]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = pphi[d];
}

static void f0_jz(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[uOff[JZ]];
}

static void f1_jz(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *ppsi = &u_x[uOff_x[PSI]];
  const PetscScalar *refPsiDer = &a_x[aOff_x[PSI]]; /* aOff_x[PSI] == 2*PSI */
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = ppsi[d] + refPsiDer[d];
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug    = 1;
  options->plotRef  = PETSC_FALSE;
  options->implicit = PETSC_FALSE;
  options->mu       = 0;
  options->eta      = 0;
  options->beta     = 1;
  options->a        = 1;
  options->b        = PETSC_PI;
  options->Jop      = 0;
  options->m        = 1;
  options->eps      = 1.e-6;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-debug", "The debugging level", "ex48.c", options->debug, &options->debug, NULL));
  CHKERRQ(PetscOptionsBool("-plot_ref", "Plot the reference fields", "ex48.c", options->plotRef, &options->plotRef, NULL));
  CHKERRQ(PetscOptionsReal("-mu", "mu", "ex48.c", options->mu, &options->mu, NULL));
  CHKERRQ(PetscOptionsReal("-eta", "eta", "ex48.c", options->eta, &options->eta, NULL));
  CHKERRQ(PetscOptionsReal("-beta", "beta", "ex48.c", options->beta, &options->beta, NULL));
  CHKERRQ(PetscOptionsReal("-Jop", "Jop", "ex48.c", options->Jop, &options->Jop, NULL));
  CHKERRQ(PetscOptionsReal("-m", "m", "ex48.c", options->m, &options->m, NULL));
  CHKERRQ(PetscOptionsReal("-eps", "eps", "ex48.c", options->eps, &options->eps, NULL));
  CHKERRQ(PetscOptionsBool("-implicit", "Use implicit time integrator", "ex48.c", options->implicit, &options->implicit, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  options->ke = PetscSqrtScalar(options->Jop);
  if (options->Jop==0.0) {
    options->Jo = 1.0/PetscPowScalar(options->a,2);
  } else {
    options->Jo = options->Jop*PetscCosReal(options->ke*options->a)/(1.0-PetscCosReal(options->ke*options->a));
  }
  options->ky = PETSC_PI*options->m/options->b;
  if (PetscPowReal(options->ky, 2) < options->Jop) {
    options->kx = PetscSqrtScalar(options->Jop-PetscPowScalar(options->ky,2));
    options->DeltaPrime = -2.0*options->kx*options->a*PetscCosReal(options->kx*options->a)/PetscSinReal(options->kx*options->a);
  } else if (PetscPowReal(options->ky, 2) > options->Jop) {
    options->kx = PetscSqrtScalar(PetscPowScalar(options->ky,2)-options->Jop);
    options->DeltaPrime = -2.0*options->kx*options->a*PetscCoshReal(options->kx*options->a)/PetscSinhReal(options->kx*options->a);
  } else { /*they're equal (or there's a NaN), lim(x*cot(x))_x->0=1*/
    options->kx = 0;
    options->DeltaPrime = -2.0;
  }
  CHKERRQ(PetscPrintf(comm, "DeltaPrime=%g\n",options->DeltaPrime));

  PetscFunctionReturn(0);
}

static void f_n(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  const PetscScalar *pn = &u[uOff[DENSITY]];
  *f0 = *pn;
}

static PetscErrorCode PostStep(TS ts)
{
  DM        dm;
  AppCtx   *ctx;
  PetscInt  stepi,num;
  Vec       X;

  PetscFunctionBegin;
  CHKERRQ(TSGetApplicationContext(ts, &ctx));
  if (ctx->debug<1) PetscFunctionReturn(0);
  CHKERRQ(TSGetSolution(ts, &X));
  CHKERRQ(VecGetDM(X, &dm));
  CHKERRQ(TSGetStepNumber(ts, &stepi));
  CHKERRQ(DMGetOutputSequenceNumber(dm, &num, NULL));
  if (num < 0) CHKERRQ(DMSetOutputSequenceNumber(dm, 0, 0.0));
  CHKERRQ(PetscObjectSetName((PetscObject) X, "u"));
  CHKERRQ(VecViewFromOptions(X, NULL, "-vec_view"));
  /* print integrals */
  {
    PetscDS          prob;
    DM               plex;
    PetscScalar den, tt[5];
    CHKERRQ(DMConvert(dm, DMPLEX, &plex));
    CHKERRQ(DMGetDS(plex, &prob));
    CHKERRQ(PetscDSSetObjective(prob, 0, &f_n));
    CHKERRQ(DMPlexComputeIntegralFEM(plex,X,tt,ctx));
    den = tt[0];
    CHKERRQ(DMDestroy(&plex));
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dm), "%D) total perturbed mass = %g\n", stepi, (double) PetscRealPart(den)));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));

  CHKERRQ(DMGetBoundingBox(*dm, ctx->lower, ctx->upper));
  ctx->a = (ctx->upper[0] - ctx->lower[0])/2.0;
  ctx->b = (ctx->upper[1] - ctx->lower[1])/2.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode log_n_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *lctx = (AppCtx*)ctx;
  u[0] = 2.*lctx->a + x[0];
  return 0;
}

static PetscErrorCode Omega_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode psi_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *lctx = (AppCtx*)ctx;
  /* This sets up a symmetrix By flux aroound the mid point in x, which represents a current density flux along z.  The stability
     is analytically known and reported in ProcessOptions. */
  if (lctx->ke!=0.0) {
    u[0] = (PetscCosReal(lctx->ke*(x[0]-lctx->a))-PetscCosReal(lctx->ke*lctx->a))/(1.0-PetscCosReal(lctx->ke*lctx->a));
  } else {
    u[0] = 1.0-PetscPowScalar((x[0]-lctx->a)/lctx->a,2);
  }
  return 0;
}

static PetscErrorCode initialSolution_n(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode initialSolution_Omega(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode initialSolution_psi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx*)a_ctx;
  PetscScalar r = ctx->eps*(PetscScalar) (rand()) / (PetscScalar) (RAND_MAX);
  if (x[0] == ctx->lower[0] || x[0] == ctx->upper[0]) r = 0;
  u[0] = r;
  /* PetscPrintf(PETSC_COMM_WORLD, "rand psi %lf\n",u[0]); */
  return 0;
}

static PetscErrorCode initialSolution_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode initialSolution_jz(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *ctx)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSSetResidual(ds, 0, f0_n,     f1_n));
  CHKERRQ(PetscDSSetResidual(ds, 1, f0_Omega, f1_Omega));
  CHKERRQ(PetscDSSetResidual(ds, 2, f0_psi,   f1_psi));
  CHKERRQ(PetscDSSetResidual(ds, 3, f0_phi,   f1_phi));
  CHKERRQ(PetscDSSetResidual(ds, 4, f0_jz,    f1_jz));
  ctx->initialFuncs[0] = initialSolution_n;
  ctx->initialFuncs[1] = initialSolution_Omega;
  ctx->initialFuncs[2] = initialSolution_psi;
  ctx->initialFuncs[3] = initialSolution_phi;
  ctx->initialFuncs[4] = initialSolution_jz;
  for (PetscInt f = 0; f < 5; ++f) {
    CHKERRQ(PetscDSSetImplicit(ds, f, ctx->implicit));
    CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, f, 0, NULL, (void (*)(void)) ctx->initialFuncs[f], NULL, ctx, NULL));
  }
  CHKERRQ(PetscDSSetContext(ds, 0, ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupEquilibriumFields(DM dm, DM dmAux, AppCtx *ctx)
{
  PetscErrorCode (*eqFuncs[3])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *) = {log_n_0, Omega_0, psi_0};
  Vec            eq;
  AppCtx *ctxarr[3];

  ctxarr[0] = ctxarr[1] = ctxarr[2] = ctx; /* each variable could have a different context */
  PetscFunctionBegin;
  CHKERRQ(DMCreateLocalVector(dmAux, &eq));
  CHKERRQ(DMProjectFunctionLocal(dmAux, 0.0, eqFuncs, (void **)ctxarr, INSERT_ALL_VALUES, eq));
  CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, eq));
  if (ctx->plotRef) {  /* plot reference functions */
    PetscViewer       viewer = NULL;
    PetscBool         isHDF5,isVTK;
    char              buf[256];
    Vec               global;
    PetscInt          dim;

    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(DMCreateGlobalVector(dmAux,&global));
    CHKERRQ(VecSet(global,.0)); /* BCs! */
    CHKERRQ(DMLocalToGlobalBegin(dmAux,eq,INSERT_VALUES,global));
    CHKERRQ(DMLocalToGlobalEnd(dmAux,eq,INSERT_VALUES,global));
    CHKERRQ(PetscViewerCreate(PetscObjectComm((PetscObject)dmAux),&viewer));
#ifdef PETSC_HAVE_HDF5
    CHKERRQ(PetscViewerSetType(viewer,PETSCVIEWERHDF5));
#else
    CHKERRQ(PetscViewerSetType(viewer,PETSCVIEWERVTK));
#endif
    CHKERRQ(PetscViewerSetFromOptions(viewer));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&isHDF5));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&isVTK));
    if (isHDF5) {
      CHKERRQ(PetscSNPrintf(buf, 256, "uEquilibrium-%dD.h5", dim));
    } else if (isVTK) {
      CHKERRQ(PetscSNPrintf(buf, 256, "uEquilibrium-%dD.vtu", dim));
      CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_VTK_VTU));
    }
    CHKERRQ(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
    CHKERRQ(PetscViewerFileSetName(viewer,buf));
    if (isHDF5) CHKERRQ(DMView(dmAux,viewer));
    /* view equilibrium fields, this will overwrite fine grids with coarse grids! */
    CHKERRQ(PetscObjectSetName((PetscObject) global, "u0"));
    CHKERRQ(VecView(global,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(VecDestroy(&global));
  }
  CHKERRQ(VecDestroy(&eq));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscInt NfAux, PetscFE feAux[], AppCtx *user)
{
  DM             dmAux, coordDM;
  PetscInt       f;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  CHKERRQ(DMGetCoordinateDM(dm, &coordDM));
  if (!feAux) PetscFunctionReturn(0);
  CHKERRQ(DMClone(dm, &dmAux));
  CHKERRQ(DMSetCoordinateDM(dmAux, coordDM));
  for (f = 0; f < NfAux; ++f) CHKERRQ(DMSetField(dmAux, f, NULL, (PetscObject) feAux[f]));
  CHKERRQ(DMCreateDS(dmAux));
  CHKERRQ(SetupEquilibriumFields(dm, dmAux, user));
  CHKERRQ(DMDestroy(&dmAux));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  DM              cdm = dm;
  PetscFE         fe[5], feAux[3];
  PetscInt        dim, Nf = 5, NfAux = 3, f;
  PetscBool       simplex;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  /* Create finite element */
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &fe[0]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[0], "density"));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &fe[1]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[1], "vorticity"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[1]));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &fe[2]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[2], "flux"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[2]));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &fe[3]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[3], "potential"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[3]));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &fe[4]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[4], "current"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[4]));

  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &feAux[0]));
  CHKERRQ(PetscObjectSetName((PetscObject) feAux[0], "n_0"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], feAux[0]));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &feAux[1]));
  CHKERRQ(PetscObjectSetName((PetscObject) feAux[1], "vorticity_0"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], feAux[1]));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &feAux[2]));
  CHKERRQ(PetscObjectSetName((PetscObject) feAux[2], "flux_0"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], feAux[2]));
  /* Set discretization and boundary conditions for each mesh */
  for (f = 0; f < Nf; ++f) CHKERRQ(DMSetField(dm, f, NULL, (PetscObject) fe[f]));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(SetupProblem(dm, ctx));
  while (cdm) {
    CHKERRQ(SetupAuxDM(dm, NfAux, feAux, ctx));
    CHKERRQ(DMCopyDisc(dm, cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  for (f = 0; f < Nf; ++f) CHKERRQ(PetscFEDestroy(&fe[f]));
  for (f = 0; f < NfAux; ++f) CHKERRQ(PetscFEDestroy(&feAux[f]));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  TS             ts;
  Vec            u, r;
  AppCtx         ctx;
  PetscReal      t       = 0.0;
  PetscReal      L2error = 0.0;
  AppCtx        *ctxarr[5];

  ctxarr[0] = ctxarr[1] = ctxarr[2] = ctxarr[3] = ctxarr[4] = &ctx; /* each variable could have a different context */
  s_ctx = &ctx;
  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  /* create mesh and problem */
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  CHKERRQ(DMSetApplicationContext(dm, &ctx));
  CHKERRQ(PetscMalloc1(5, &ctx.initialFuncs));
  CHKERRQ(SetupDiscretization(dm, &ctx));
  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "u"));
  CHKERRQ(VecDuplicate(u, &r));
  CHKERRQ(PetscObjectSetName((PetscObject) r, "r"));
  /* create TS */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD, &ts));
  CHKERRQ(TSSetDM(ts, dm));
  CHKERRQ(TSSetApplicationContext(ts, &ctx));
  CHKERRQ(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx));
  if (ctx.implicit) {
    CHKERRQ(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx));
    CHKERRQ(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx));
  } else {
    CHKERRQ(DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, &ctx));
  }
  CHKERRQ(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetPostStep(ts, PostStep));
  /* make solution & solve */
  CHKERRQ(DMProjectFunction(dm, t, ctx.initialFuncs, (void **)ctxarr, INSERT_ALL_VALUES, u));
  CHKERRQ(TSSetSolution(ts,u));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(PostStep(ts)); /* print the initial state */
  CHKERRQ(TSSolve(ts, u));
  CHKERRQ(TSGetTime(ts, &t));
  CHKERRQ(DMComputeL2Diff(dm, t, ctx.initialFuncs, (void **)ctxarr, u, &L2error));
  if (L2error < 1.0e-11) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n"));
  else                   CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", L2error));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFree(ctx.initialFuncs));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -debug 1 -dm_refine 1 -dm_plex_simplex 0 -dm_plex_box_faces 3,3 -dm_plex_box_bd periodic,none -dm_plex_box_upper 2.0,6.283185307179586 \
          -ts_max_steps 1 -ts_max_time 10. -ts_dt 1.0
  test:
    # Remapping with periodicity is broken
    suffix: 1
    args: -debug 1 -dm_plex_shape cylinder -dm_plex_dim 3 -dm_refine 1 -dm_refine_remap 0 -dm_plex_cylinder_bd periodic -dm_plex_boundary_label marker \
           -ts_max_steps 1 -ts_max_time 10. -ts_dt 1.0

TEST*/

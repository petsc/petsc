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
#include <assert.h>

typedef struct {
  PetscInt       debug;             /* The debugging level */
  PetscBool      plotRef;           /* Plot the reference fields */
  /* Domain and mesh definition */
  PetscInt       dim;               /* The topological mesh dimension */
  char           filename[2048];    /* The optional ExodusII file */
  PetscBool      cell_simplex;           /* Simplicial mesh */
  DMBoundaryType boundary_types[3];
  PetscInt       cells[3];
  PetscInt       refine;
  /* geometry  */
  PetscReal      domain_lo[3], domain_hi[3];
  DMBoundaryType periodicity[3];              /* The domain periodicity */
  PetscReal      b0[3]; /* not used */
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
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscInt       ii, bd;
  PetscFunctionBeginUser;
  options->debug               = 1;
  options->plotRef             = PETSC_FALSE;
  options->dim                 = 2;
  options->filename[0]         = '\0';
  options->cell_simplex        = PETSC_FALSE;
  options->implicit            = PETSC_FALSE;
  options->refine              = 2;
  options->domain_lo[0]  = 0.0;
  options->domain_lo[1]  = 0.0;
  options->domain_lo[2]  = 0.0;
  options->domain_hi[0]  = 2.0;
  options->domain_hi[1]  = 2.0*PETSC_PI;
  options->domain_hi[2]  = 2.0;
  options->periodicity[0]    = DM_BOUNDARY_NONE;
  options->periodicity[1]    = DM_BOUNDARY_NONE;
  options->periodicity[2]    = DM_BOUNDARY_NONE;
  options->mu   = 0;
  options->eta  = 0;
  options->beta = 1;
  options->a = 1;
  options->b = PETSC_PI;
  options->Jop = 0;
  options->m = 1;
  options->eps = 1.e-6;

  for (ii = 0; ii < options->dim; ++ii) options->cells[ii] = 4;
  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex48.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-plot_ref", "Plot the reference fields", "ex48.c", options->plotRef, &options->plotRef, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex48.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  if (options->dim < 2 || options->dim > 3) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Dim %D must be 2 or 3",options->dim);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_refine", "Hack to get refinement level for cylinder", "ex48.c", options->refine, &options->refine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu", "mu", "ex48.c", options->mu, &options->mu, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eta", "eta", "ex48.c", options->eta, &options->eta, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-beta", "beta", "ex48.c", options->beta, &options->beta, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Jop", "Jop", "ex48.c", options->Jop, &options->Jop, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-m", "m", "ex48.c", options->m, &options->m, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps", "eps", "ex48.c", options->eps, &options->eps, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Exodus.II filename to read", "ex48.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Simplicial (true) or tensor (false) mesh", "ex48.c", options->cell_simplex, &options->cell_simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit", "Use implicit time integrator", "ex48.c", options->implicit, &options->implicit, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_hi", "Domain size", "ex48.c", options->domain_hi, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_lo", "Domain size", "ex48.c", options->domain_lo, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  bd = options->periodicity[0];
  ierr = PetscOptionsEList("-x_periodicity", "The x-boundary periodicity", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[0]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[0] = (DMBoundaryType) bd;
  bd = options->periodicity[1];
  ierr = PetscOptionsEList("-y_periodicity", "The y-boundary periodicity", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[1]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[1] = (DMBoundaryType) bd;
  bd = options->periodicity[2];
  ierr = PetscOptionsEList("-z_periodicity", "The z-boundary periodicity", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[2]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[2] = (DMBoundaryType) bd;
  ii = options->dim;
  ierr = PetscOptionsIntArray("-cells", "Number of cells in each dimension", "ex48.c", options->cells, &ii, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  options->a = (options->domain_hi[0]-options->domain_lo[0])/2.0;
  options->b = (options->domain_hi[1]-options->domain_lo[1])/2.0;
  for (ii = 0; ii < options->dim; ++ii) {
    if (options->domain_hi[ii] <= options->domain_lo[ii]) SETERRQ3(comm,PETSC_ERR_ARG_WRONG,"Domain %D lo=%g hi=%g",ii,options->domain_lo[ii],options->domain_hi[ii]);
  }
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
  ierr = PetscPrintf(comm, "DeltaPrime=%g\n",options->DeltaPrime);CHKERRQ(ierr);

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
  PetscErrorCode    ierr;
  DM                dm;
  AppCtx            *ctx;
  PetscInt          stepi,num;
  Vec               X;
  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts, &ctx);CHKERRQ(ierr); assert(ctx);
  if (ctx->debug<1) PetscFunctionReturn(0);
  ierr = TSGetSolution(ts, &X);CHKERRQ(ierr);
  ierr = VecGetDM(X, &dm);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &stepi);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &num, NULL);CHKERRQ(ierr);
  if (num < 0) {ierr = DMSetOutputSequenceNumber(dm, 0, 0.0);CHKERRQ(ierr);}
  ierr = PetscObjectSetName((PetscObject) X, "u");CHKERRQ(ierr);
  ierr = VecViewFromOptions(X, NULL, "-vec_view");CHKERRQ(ierr);
  /* print integrals */
  {
    PetscDS          prob;
    DM               plex;
    PetscScalar den, tt[5];
    ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f_n);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,ctx);CHKERRQ(ierr);
    den = tt[0];
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
    PetscPrintf(PetscObjectComm((PetscObject)dm), "%D) total perturbed mass = %g\n", stepi, (double) PetscRealPart(den));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DM             plex;
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscInt       dim      = ctx->dim;
  const char    *filename = ctx->filename;
  size_t         len;
  PetscMPIInt    numProcs;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRMPI(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    PetscInt        d;

    /* create DM */
    if (ctx->cell_simplex && dim == 3) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Cannot mesh a cylinder with simplices");
    if (dim==2) {
      PetscInt refineRatio, totCells = 1;
      if (ctx->cell_simplex) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Cannot mesh 2D with simplices");
      refineRatio = PetscMax((PetscInt) (PetscPowReal(numProcs, 1.0/dim) + 0.1) - 1, 1);
      for (d = 0; d < dim; ++d) {
        if (ctx->cells[d] < refineRatio) ctx->cells[d] = refineRatio;
        if (ctx->periodicity[d]==DM_BOUNDARY_PERIODIC && ctx->cells[d]*refineRatio <= 2) refineRatio = 2;
      }
      for (d = 0; d < dim; ++d) {
        ctx->cells[d] *= refineRatio;
        totCells *= ctx->cells[d];
      }
      if (totCells % numProcs) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Total cells %D not divisible by processes %D", totCells, numProcs);
      ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, ctx->cells, ctx->domain_lo, ctx->domain_hi, ctx->periodicity, PETSC_TRUE, dm);CHKERRQ(ierr);
    } else {
      if (ctx->periodicity[0]==DM_BOUNDARY_PERIODIC || ctx->periodicity[1]==DM_BOUNDARY_PERIODIC) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Cannot do periodic in x or y in a cylinder");
      /* we stole dm_refine so clear it */
      ierr = PetscOptionsClearValue(NULL,"-dm_refine");CHKERRQ(ierr);
      ierr = DMPlexCreateHexCylinderMesh(comm, ctx->refine, ctx->periodicity[2], dm);CHKERRQ(ierr);
    }
  }
  {
    DM distributedMesh = NULL;
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  {
    PetscBool hasLabel;
    ierr = DMHasLabel(*dm, "marker", &hasLabel);CHKERRQ(ierr);
    if (!hasLabel) {ierr = CreateBCLabel(*dm, "marker");CHKERRQ(ierr);}
  }
  {
    char      convType[256];
    PetscBool flg;
    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex48",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      DM dmConv;
      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  PetscFunctionReturn(0);
}

static PetscErrorCode log_n_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *lctx = (AppCtx*)ctx;
  assert(ctx);
  u[0] = (lctx->domain_hi-lctx->domain_lo)+x[0];
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
  assert(ctx);
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
  assert(ctx);
  if (x[0] == ctx->domain_lo[0] || x[0] == ctx->domain_hi[0]) r = 0;
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
  PetscDS        prob;
  const PetscInt id = 1;
  PetscErrorCode ierr, f;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_n,     f1_n);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_Omega, f1_Omega);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 2, f0_psi,   f1_psi);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 3, f0_phi,   f1_phi);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 4, f0_jz,    f1_jz);CHKERRQ(ierr);
  ctx->initialFuncs[0] = initialSolution_n;
  ctx->initialFuncs[1] = initialSolution_Omega;
  ctx->initialFuncs[2] = initialSolution_psi;
  ctx->initialFuncs[3] = initialSolution_phi;
  ctx->initialFuncs[4] = initialSolution_jz;
  for (f = 0; f < 5; ++f) {
    ierr = PetscDSSetImplicit(prob, f, ctx->implicit);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", f, 0, NULL, (void (*)(void)) ctx->initialFuncs[f], NULL, 1, &id, ctx);CHKERRQ(ierr);
  }
  ierr = PetscDSSetContext(prob, 0, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupEquilibriumFields(DM dm, DM dmAux, AppCtx *ctx)
{
  PetscErrorCode (*eqFuncs[3])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *) = {log_n_0, Omega_0, psi_0};
  Vec            eq;
  PetscErrorCode ierr;
  AppCtx *ctxarr[3];

  ctxarr[0] = ctxarr[1] = ctxarr[2] = ctx; /* each variable could have a different context */
  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dmAux, &eq);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, eqFuncs, (void **)ctxarr, INSERT_ALL_VALUES, eq);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) eq);CHKERRQ(ierr);
  if (ctx->plotRef) {  /* plot reference functions */
    PetscViewer       viewer = NULL;
    PetscBool         isHDF5,isVTK;
    char              buf[256];
    Vec               global;
    ierr = DMCreateGlobalVector(dmAux,&global);CHKERRQ(ierr);
    ierr = VecSet(global,.0);CHKERRQ(ierr); /* BCs! */
    ierr = DMLocalToGlobalBegin(dmAux,eq,INSERT_VALUES,global);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dmAux,eq,INSERT_VALUES,global);CHKERRQ(ierr);
    ierr = PetscViewerCreate(PetscObjectComm((PetscObject)dmAux),&viewer);CHKERRQ(ierr);
#ifdef PETSC_HAVE_HDF5
    ierr = PetscViewerSetType(viewer,PETSCVIEWERHDF5);CHKERRQ(ierr);
#else
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
#endif
    ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&isHDF5);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&isVTK);CHKERRQ(ierr);
    if (isHDF5) {
      ierr = PetscSNPrintf(buf, 256, "uEquilibrium-%dD.h5", ctx->dim);CHKERRQ(ierr);
    } else if (isVTK) {
      ierr = PetscSNPrintf(buf, 256, "uEquilibrium-%dD.vtu", ctx->dim);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    }
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,buf);CHKERRQ(ierr);
    if (isHDF5) {ierr = DMView(dmAux,viewer);CHKERRQ(ierr);}
    /* view equilibrium fields, this will overwrite fine grids with coarse grids! */
    ierr = PetscObjectSetName((PetscObject) global, "u0");CHKERRQ(ierr);
    ierr = VecView(global,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&global);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&eq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscInt NfAux, PetscFE feAux[], AppCtx *user)
{
  DM             dmAux, coordDM;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
  if (!feAux) PetscFunctionReturn(0);
  ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
  for (f = 0; f < NfAux; ++f) {ierr = DMSetField(dmAux, f, NULL, (PetscObject) feAux[f]);CHKERRQ(ierr);}
  ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
  ierr = SetupEquilibriumFields(dm, dmAux, user);CHKERRQ(ierr);
  ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  DM              cdm = dm;
  const PetscInt  dim = ctx->dim;
  PetscFE         fe[5], feAux[3];
  PetscInt        Nf = 5, NfAux = 3, f;
  PetscBool       cell_simplex = ctx->cell_simplex;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "density");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "vorticity");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[1]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[2]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[2], "flux");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[2]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[3]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[3], "potential");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[3]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[4]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[4], "current");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[4]);CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &feAux[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feAux[0], "n_0");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], feAux[0]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &feAux[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feAux[1], "vorticity_0");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], feAux[1]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &feAux[2]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feAux[2], "flux_0");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], feAux[2]);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  for (f = 0; f < Nf; ++f) {ierr = DMSetField(dm, f, NULL, (PetscObject) fe[f]);CHKERRQ(ierr);}
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, ctx);CHKERRQ(ierr);
  while (cdm) {
    ierr = SetupAuxDM(dm, NfAux, feAux, ctx);CHKERRQ(ierr);
    {
      PetscBool hasLabel;

      ierr = DMHasLabel(cdm, "marker", &hasLabel);CHKERRQ(ierr);
      if (!hasLabel) {ierr = CreateBCLabel(cdm, "marker");CHKERRQ(ierr);}
    }
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&fe[f]);CHKERRQ(ierr);}
  for (f = 0; f < NfAux; ++f) {ierr = PetscFEDestroy(&feAux[f]);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;
  AppCtx        *ctxarr[5];

  ctxarr[0] = ctxarr[1] = ctxarr[2] = ctxarr[3] = ctxarr[4] = &ctx; /* each variable could have a different context */
  s_ctx = &ctx;
  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  /* create mesh and problem */
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = PetscMalloc1(5, &ctx.initialFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &ctx);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "u");CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "r");CHKERRQ(ierr);
  /* create TS */
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx);CHKERRQ(ierr);
  if (ctx.implicit) {
    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx);CHKERRQ(ierr);
  } else {
    ierr = DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, &ctx);CHKERRQ(ierr);
  }
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts, PostStep);CHKERRQ(ierr);
  /* make solution & solve */
  ierr = DMProjectFunction(dm, t, ctx.initialFuncs, (void **)ctxarr, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PostStep(ts);CHKERRQ(ierr); /* print the initial state */
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, t, ctx.initialFuncs, (void **)ctxarr, u, &L2error);CHKERRQ(ierr);
  if (L2error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
  else                   {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", L2error);CHKERRQ(ierr);}
#if 0
  {
    PetscReal res = 0.0;
    /* Check discretization error */
    ierr = VecViewFromOptions(u, NULL, "-initial_guess_view");CHKERRQ(ierr);
    ierr = DMComputeL2Diff(dm, 0.0, ctx.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
    else                 {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);}
    /* Check residual */
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecViewFromOptions(r, NULL, "-initial_residual_view");CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Mat A;
      Vec b;

      ierr = SNESGetJacobian(snes, &A, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = SNESComputeJacobian(snes, u, A, A);CHKERRQ(ierr);
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
      ierr = MatMult(A, u, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n");CHKERRQ(ierr);
      ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
      ierr = VecViewFromOptions(r, NULL, "-linear_residual_view");CHKERRQ(ierr);
      ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", res);CHKERRQ(ierr);
    }
  }
#endif
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(ctx.initialFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -debug 1 -dim 2 -dm_refine 1 -x_periodicity PERIODIC -ts_max_steps 1 -ts_max_time 10. -ts_dt 1.0
  test:
    suffix: 1
    args: -debug 1 -dim 3 -dm_refine 1 -z_periodicity PERIODIC -ts_max_steps 1 -ts_max_time 10. -ts_dt 1.0 -domain_lo -2,-1,-1 -domain_hi 2,1,1

TEST*/

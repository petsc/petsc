
#include <petsc/private/petscimpl.h>
#include <petscdm.h>     /*I "petscdm.h" I*/
#include <petscdmplex.h> /*I "petscdmplex.h" I*/
#include <petscksp.h>    /*I "petscksp.h" I*/

typedef struct _projectConstraintsCtx
{
  DM  dm;
  Vec mask;
}
projectConstraintsCtx;

PetscErrorCode MatMult_GlobalToLocalNormal(Mat CtC, Vec x, Vec y)
{
  DM                    dm;
  Vec                   local, mask;
  projectConstraintsCtx *ctx;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(CtC,&ctx);CHKERRQ(ierr);
  dm   = ctx->dm;
  mask = ctx->mask;
  ierr = DMGetLocalVector(dm,&local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x,INSERT_VALUES,local);CHKERRQ(ierr);
  if (mask) {ierr = VecPointwiseMult(local,mask,local);CHKERRQ(ierr);}
  ierr = VecSet(y,0.);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,local,ADD_VALUES,y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,local,ADD_VALUES,y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalSolve_project1 (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscInt f;

  PetscFunctionBegin;
  for (f = 0; f < Nf; f++) {
    u[f] = 1.;
  }
  PetscFunctionReturn(0);
}

/*@
  DMGlobalToLocalSolve - Solve for the global vector that is mapped to a given local vector by DMGlobalToLocalBegin()/DMGlobalToLocalEnd() with mode
  = INSERT_VALUES.  It is assumed that the sum of all the local vector sizes is greater than or equal to the global vector size, so the solution is
  a least-squares solution.  It is also assumed that DMLocalToGlobalBegin()/DMLocalToGlobalEnd() with mode = ADD_VALUES is the adjoint of the
  global-to-local map, so that the least-squares solution may be found by the normal equations.

  collective

  Input Parameters:
+ dm - The DM object
. x - The local vector
- y - The global vector: the input value of globalVec is used as an initial guess

  Output Parameters:
. y - The least-squares solution

  Level: advanced

  Note: If the DM is of type DMPLEX, then y is the solution of L' * D * L * y = L' * D * x, where D is a diagonal mask that is 1 for every point in
  the union of the closures of the local cells and 0 otherwise.  This difference is only relevant if there are anchor points that are not in the
  closure of any local cell (see DMPlexGetAnchors()/DMPlexSetAnchors()).

.seealso: DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMLocalToGlobalEnd(), DMPlexGetAnchors(), DMPlexSetAnchors()
@*/
PetscErrorCode DMGlobalToLocalSolve(DM dm, Vec x, Vec y)
{
  Mat                   CtC;
  PetscInt              n, N, cStart, cEnd, c;
  PetscBool             isPlex;
  KSP                   ksp;
  PC                    pc;
  Vec                   global, mask=NULL;
  projectConstraintsCtx ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (isPlex) {
    /* mark points in the closure */
    ierr = DMCreateLocalVector(dm,&mask);CHKERRQ(ierr);
    ierr = VecSet(mask,0.0);CHKERRQ(ierr);
    ierr = DMPlexGetInteriorCellStratum(dm,&cStart,&cEnd);CHKERRQ(ierr);
    if (cEnd > cStart) {
      PetscScalar *ones;
      PetscInt numValues, i;

      ierr = DMPlexVecGetClosure(dm,NULL,mask,cStart,&numValues,NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(numValues,&ones);CHKERRQ(ierr);
      for (i = 0; i < numValues; i++) {
        ones[i] = 1.;
      }
      for (c = cStart; c < cEnd; c++) {
        ierr = DMPlexVecSetClosure(dm,NULL,mask,c,ones,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = PetscFree(ones);CHKERRQ(ierr);
    }
  }
  else {
    PetscBool hasMask;

    ierr = DMHasNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &hasMask);CHKERRQ(ierr);
    if (!hasMask) {
      PetscErrorCode (**func) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
      void            **ctx;
      PetscInt          Nf, f;

      ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
      ierr = PetscMalloc2(Nf, &func, Nf, &ctx);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        func[f] = DMGlobalToLocalSolve_project1;
        ctx[f]  = NULL;
      }
      ierr = DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask);CHKERRQ(ierr);
      ierr = DMProjectFunctionLocal(dm,0.0,func,ctx,INSERT_ALL_VALUES,mask);CHKERRQ(ierr);
      ierr = DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask);CHKERRQ(ierr);
      ierr = PetscFree2(func, ctx);CHKERRQ(ierr);
    }
    ierr = DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask);CHKERRQ(ierr);
  }
  ctx.dm   = dm;
  ctx.mask = mask;
  ierr = VecGetSize(y,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)dm),&CtC);CHKERRQ(ierr);
  ierr = MatSetSizes(CtC,n,n,N,N);CHKERRQ(ierr);
  ierr = MatSetType(CtC,MATSHELL);CHKERRQ(ierr);
  ierr = MatSetUp(CtC);CHKERRQ(ierr);
  ierr = MatShellSetContext(CtC,&ctx);CHKERRQ(ierr);
  ierr = MatShellSetOperation(CtC,MATOP_MULT,(void(*)(void))MatMult_GlobalToLocalNormal);CHKERRQ(ierr);
  ierr = KSPCreate(PetscObjectComm((PetscObject)dm),&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,CtC,CtC);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&global);CHKERRQ(ierr);
  ierr = VecSet(global,0.);CHKERRQ(ierr);
  if (mask) {ierr = VecPointwiseMult(x,mask,x);CHKERRQ(ierr);}
  ierr = DMLocalToGlobalBegin(dm,x,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,x,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,global,y);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&global);CHKERRQ(ierr);
  /* clean up */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&CtC);CHKERRQ(ierr);
  if (isPlex) {
    ierr = VecDestroy(&mask);CHKERRQ(ierr);
  }
  else {
    ierr = DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*@C
  DMProjectField - This projects the given function of the input fields into the function space provided, putting the coefficients in a global vector.

  Collective on DM

  Input Parameters:
+ dm      - The DM
. time    - The time
. U       - The input field vector
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. X       - The output vector

   Calling sequence of func:
$    func(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);

+  dim          - The spatial dimension
.  Nf           - The number of input fields
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of each field in u[]
.  uOff_x       - The offset of each field in u_x[]
.  u            - The field values at this point in space
.  u_t          - The field time derivative at this point in space (or NULL)
.  u_x          - The field derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note: There are three different DMs that potentially interact in this function. The output DM, dm, specifies the layout of the values calculates by funcs.
  The input DM, attached to U, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary DM, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

.seealso: DMProjectFieldLocal(), DMProjectFieldLabelLocal(), DMProjectFunction(), DMComputeL2Diff()
@*/
PetscErrorCode DMProjectField(DM dm, PetscReal time, Vec U,
                              void (**funcs)(PetscInt, PetscInt, PetscInt,
                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                             PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                              InsertMode mode, Vec X)
{
  Vec            localX, localU;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  /* We currently check whether locU == locX to see if we need to apply BC */
  if (U != X) {ierr = DMGetLocalVector(dm, &localU);CHKERRQ(ierr);}
  else        {localU = localX;}
  ierr = DMGlobalToLocalBegin(dm, U, INSERT_VALUES, localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, U, INSERT_VALUES, localU);CHKERRQ(ierr);
  ierr = DMProjectFieldLocal(dm, time, localU, funcs, mode, localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  if (mode == INSERT_VALUES || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES) {
    Mat cMat;

    ierr = DMGetDefaultConstraints(dm, NULL, &cMat);CHKERRQ(ierr);
    if (cMat) {
      ierr = DMGlobalToLocalSolve(dm, localX, X);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  if (U != X) {ierr = DMRestoreLocalVector(dm, &localU);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

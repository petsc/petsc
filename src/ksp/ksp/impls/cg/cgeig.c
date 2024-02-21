/*
   Code for calculating extreme eigenvalues via the Lanczos method
   running with CG. Note this only works for symmetric real and Hermitian
   matrices (not complex matrices that are symmetric).
*/
#include <../src/ksp/ksp/impls/cg/cgimpl.h>
#include <../include/petscblaslapack.h>

PetscErrorCode KSPComputeEigenvalues_CG(KSP ksp, PetscInt nmax, PetscReal *r, PetscReal *c, PetscInt *neig)
{
  KSP_CG      *cgP = (KSP_CG *)ksp->data;
  PetscScalar *d, *e;
  PetscReal   *ee;
  PetscInt     n = ksp->its;
  PetscBLASInt bn, lierr = 0, ldz = 1;

  PetscFunctionBegin;
  PetscCheck(nmax >= n, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_SIZ, "Not enough room in work space r and c for eigenvalues");
  *neig = n;

  PetscCall(PetscArrayzero(c, nmax));
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  d  = cgP->d;
  e  = cgP->e;
  ee = cgP->ee;

  /* copy tridiagonal matrix to work space */
  for (PetscInt j = 0; j < n; j++) {
    r[j]  = PetscRealPart(d[j]);
    ee[j] = PetscRealPart(e[j + 1]);
  }

  PetscCall(PetscBLASIntCast(n, &bn));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKREALstev", LAPACKREALstev_("N", &bn, r, ee, NULL, &ldz, NULL, &lierr));
  PetscCheck(!lierr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "xSTEV error");
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscSortReal(n, r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP ksp, PetscReal *emax, PetscReal *emin)
{
  KSP_CG      *cgP = (KSP_CG *)ksp->data;
  PetscScalar *d, *e;
  PetscReal   *dd, *ee;
  PetscInt     n = ksp->its;
  PetscBLASInt bn, lierr = 0, ldz = 1;

  PetscFunctionBegin;
  if (!n) {
    *emax = *emin = 1.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  d  = cgP->d;
  e  = cgP->e;
  dd = cgP->dd;
  ee = cgP->ee;

  /* copy tridiagonal matrix to work space */
  for (PetscInt j = 0; j < n; j++) {
    dd[j] = PetscRealPart(d[j]);
    ee[j] = PetscRealPart(e[j + 1]);
  }

  PetscCall(PetscBLASIntCast(n, &bn));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKREALstev", LAPACKREALstev_("N", &bn, dd, ee, NULL, &ldz, NULL, &lierr));
  PetscCheck(!lierr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "xSTEV error");
  PetscCall(PetscFPTrapPop());
  *emin = dd[0];
  *emax = dd[n - 1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

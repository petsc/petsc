

#include "slesimpl.h"

/*@
    SLESCreate - Creates a linear equation solver context

  Output Parameter:
.   sles - the create context
@*/
int SLESCreate(outsles)
SLES *outsles;
{
  int ierr;
  SLES sles;
  *outsles = 0;
  CREATEHEADER(sles,_SLES);
  if (ierr = KSPCreate(&sles->ksp)) return ierr;
  if (ierr = PCCreate(&sles->pc)) return ierr;
  sles->cookie = SLES_COOKIE;
  *outsles = sles;
  return 0;
}

/*@
    SLESGetKSP - returns the Krylov Space context for a sles solver.

  Input Parameter:
.   sles - the solver context

  Output Paramter:
.   ksp - the Krylov space context
@*/
int SLESGetKSP(sles,ksp)
SLES sles;
KSP  *ksp;
{
  VALIDHEADER(sles,SLES_COOKIE);
  *ksp = sles->ksp;
  return 0;
}
/*@
    SLESGetPC - returns the preconditioner context for a sles solver.

  Input Parameter:
.   sles - the solver context

  Output Paramter:
.  pc - the preconditioner context
@*/
int SLESGetPC(sles,pc)
SLES sles;
PC   *pc;
{
  VALIDHEADER(sles,SLES_COOKIE);
  *pc = sles->pc;
  return 0;
}

#include "mat/matimpl.h"
/*@
    SLESSetMat - sets the matrix to use for the sles solver.

  Input Parameters:
.   sles - the sles context
.   mat - the matrix to use
@*/
int SLESSetMat(sles,mat)
SLES sles;
Mat  mat;
{
  VALIDHEADER(sles,SLES_COOKIE);
  VALIDHEADER(mat,MAT_COOKIE);
  sles->mat = mat;
  return 0;
}

From bsmith@mcs.anl.gov Wed Jun  2 15:24:57 1999
Status: RO
X-Status: 
Received: (from daemon@localhost) by antares.mcs.anl.gov (8.6.10/8.6.10)
	id PAA11839 for petsc-maint-dist; Wed, 2 Jun 1999 15:24:57 -0500
Received: from newman.cs.purdue.edu (0@newman.cs.purdue.edu [128.10.2.6]) by antares.mcs.anl.gov (8.6.10/8.6.10)  with ESMTP
        id PAA11834 for <petsc-maint@mcs.anl.gov>; Wed, 2 Jun 1999 15:24:51 -0500
Received: from khan.cs.purdue.edu (0@khan.cs.purdue.edu [128.10.8.35])
        by newman.cs.purdue.edu (8.8.7/8.8.7/PURDUE_CS-2.0) with ESMTP id PAA20176
        for <petsc-maint@mcs.anl.gov>; Wed, 2 Jun 1999 15:24:48 -0500 (EST)
Received: from localhost (604@localhost [127.0.0.1])
        by khan.cs.purdue.edu (8.8.7/8.8.7/PURDUE_CS-2.0) with SMTP id PAA15635
        for <petsc-maint@mcs.anl.gov>; Wed, 2 Jun 1999 15:24:47 -0500 (EST)
Message-Id: <199906022024.PAA15635@khan.cs.purdue.edu>
X-Authentication-Warning: khan.cs.purdue.edu: 604@localhost [127.0.0.1] didn't use HELO protocol
To: petsc-maint@mcs.anl.gov
Subject: [PETSC #2602] Line Search Params
Date: Wed, 02 Jun 1999 15:24:45 -0500
From: knepley@cs.purdue.edu (Matthew Gregg Knepley)
Cc: petsc-maint@mcs.anl.gov

        I did not see a way to replicate the cubic line search included
in Petsc without breaking the encapsulation, so I wrote some routines
to retrieve the line search parameters and put them in ls.c. Here is the
code (and tell me if I missed some other way to do it).

        Thanks,

                Matt

----------ls.c (changes)
#undef __FUNC__  
#define __FUNC__ "SNESSetLineSeachParams"
/*@C
   SNESSetLineSearchCheck - Sets the parameters associated with the line search
   routine in the Newton-based method SNES_EQ_LS.

   Collective on SNES

   Input Parameters:
+  snes    - The nonlinear context obtained from SNESCreate()
.  alpha   - The scalar such that x_{n+1} . x_{n+1} <= x_n . x_n - alpha |x_n . J . x_n|
.  maxstep - The maximum norm of the update vector
-  steptol - The minimum norm fraction of the original step after scaling

   Level: intermediate

   Note:
   Negative values are ignored.

.keywords: SNES, nonlinear, set, line search params

.seealso: SNESGetLineSearchParams(), SNESSetLineSearch()
@*/
int SNESSetLineSearchParams(SNES snes, double alpha, double maxstep, double steptol)
{
  SNES_LS *ls = (SNES_LS *) snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_COOKIE);
  if (alpha   >= 0.0)
    ls->alpha   = alpha;
  if (maxstep >= 0.0)
    ls->maxstep = maxstep;
  if (steptol >= 0.0)
    ls->steptol = steptol;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESGetLineSeachParams"
/*@C
   SNESGetLineSearchCheck - Gets the parameters associated with the line search
   routine in the Newton-based method SNES_EQ_LS.

   Collective on SNES

   Input Parameters:
+  snes    - The nonlinear context obtained from SNESCreate()
.  alpha   - The scalar such that x_{n+1} . x_{n+1} <= x_n . x_n - alpha |x_n . J . x_n|
.  maxstep - The maximum norm of the update vector
-  steptol - The minimum norm fraction of the original step after scaling

   Level: intermediate

   Note:
   The argument PETSC_NULL is ignored.

.keywords: SNES, nonlinear, set, line search params

.seealso: SNESSetLineSearchParams()
@*/
int SNESGetLineSearchParams(SNES snes, double *alpha, double *maxstep, double *steptol)
{
  SNES_LS *ls = (SNES_LS *) snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_COOKIE);
  if (alpha   != PETSC_NULL) {
    PetscValidPointer(alpha);
    *alpha   = ls->alpha;
  }
  if (maxstep != PETSC_NULL) {
    PetscValidPointer(maxstep);
    *maxstep = ls->maxstep;
  }
  if (steptol != PETSC_NULL) {
    PetscValidPointer(steptol);
    *steptol = ls->steptol;
  }
  PetscFunctionReturn(0);
}



/*
       This file contains an implementaion of Tony Chan's 
   transpose free QMR.
*/

#include <math.h>
#include <stdio.h>
#include "petsc.h"
#include "kspimpl.h"
#include "tcqmrp.h"

static int KSPiTCQMRSolve(KSP itP,int *its )
{
double      rnorm0, rnorm;                      /* residual values */
Scalar      theta, ep, cl1, sl1, cl, sl, sprod, tau_n1, f; 
Scalar      deltmp, rho, beta, eptmp, ta, s, c, tau_n, delta;
Scalar      dp11,dp2, rhom1, alpha,tmp, zero = 0.0;
int         it, cerr;
double      dp1,Gamma;

it = 0;
KSPResidual(itP,x,u,v, r, v0, b );
VecNorm(r,&rnorm0);                            /*  rnorm0 = ||r|| */

VecSet(&zero,um1);
VecCopy(r,u);
rnorm = rnorm0;
tmp = 1.0/rnorm; VecScale(&tmp, u);
VecSet(&zero,vm1);
VecCopy(u,v);
VecCopy(u,v0);
VecSet(&zero,pvec1);
VecSet(&zero,pvec2);
VecSet(&zero,p);
theta = 0.0; 
ep    = 0.0; 
cl1   = 0.0; 
sl1   = 0.0; 
cl    = 0.0; 
sl    = 0.0;
sprod = 1.0; 
tau_n1= rnorm0;
f     = 1.0; 
Gamma = 1.0; 
rhom1 = 1.0;

/*
 CALCULATE SQUARED LANCZOS  vectors
 */
while ( !CONVERGED(itP,rnorm,it)) {     
    if (itP->usr_monitor) {
        (*itP->usr_monitor)( itP, it, rnorm,itP->monP );
	}
    MATOP(itP, u, y, vtmp );                       /* y = A*u */
    VecDot( v0, y, &dp11 );
    VecDot( v0, u, &dp2 );
    alpha = dp11 / dp2;                          /* alpha = v0'*y/v0'*u */
    deltmp = alpha;
    VecCopy(y,z);     
    tmp = -alpha; VecAXPY(&tmp,u,z);             /* z = y - alpha u */
    VecDot( v0, u, &rho );
    beta   = rho / (f*rhom1);
    rhom1  = rho;
    VecCopy(z,utmp);                               /* up1 = (A-alpha*I)*
					       (z-2*beta*p) + f*beta*
					     beta*um1 */
    tmp = -2.0*beta;VecAXPY(&tmp,p,utmp);
    MATOP(itP,utmp,up1,vtmp);
    tmp = -alpha; VecAXPY(&tmp,utmp,up1);
    tmp = f*beta*beta; VecAXPY(&tmp,um1,up1);
    VecNorm(up1,&dp1);
    f     = 1.0 / dp1;
    VecScale(&f,up1);
    tmp = -beta; VecAYPX(&tmp,z,p);                          /* p = f*(z-beta*p) */
    VecScale(&f,p);
    VecCopy(u,um1);
    VecCopy(up1,u);
    beta  = beta/Gamma;
    eptmp = beta;
    MATOP(itP,v,vp1,vtmp);
    tmp = -alpha; VecAXPY(&tmp,v,vp1);
    tmp = -beta;VecAXPY(&tmp,vm1,vp1);
    VecNorm(vp1,&Gamma);
    tmp = 1.0/Gamma; VecScale(&tmp,vp1);
    VecCopy(v,vm1);
    VecCopy(vp1,v);

/*
     SOLVE  Ax = b
 */
/* Apply the last two Given's (Gl-1 and Gl) rotations to (beta,alpha,Gamma) */
    if (it > 1) {
	theta =  sl1*beta;
	eptmp = -cl1*beta;
	}
    if (it > 0) {
	ep     = -cl*eptmp + sl*alpha;
	deltmp = -sl*eptmp - cl*alpha;
	}
#if defined(PETSC_COMPLEX)
    if (abs(Gamma) > abs(deltmp)) {
#else    
    if (fabs(Gamma) > fabs(deltmp)) {
#endif
	ta = -deltmp / Gamma;
	s = 1.0 / sqrt(1.0 + ta*ta);
	c = s*ta;
	}
    else {
	ta = -Gamma/deltmp;
	c = 1.0 / sqrt(1.0 + ta*ta);
	s = c*ta;
	}

    delta  = -c*deltmp + s*Gamma;
    tau_n  = -c*tau_n1; tau_n1 = -s*tau_n1;
    VecCopy(vm1,pvec);
    tmp = -theta; VecAXPY(&tmp,pvec2,pvec);
    tmp = -ep; VecAXPY(&tmp,pvec1,pvec);
    tmp = 1.0/delta; VecScale(&tmp,pvec);
    VecAXPY(&tau_n,pvec,x);
    cl1 = cl; sl1 = sl; cl = c; sl = s;     

    VecCopy(pvec1,pvec2);
    VecCopy(pvec,pvec1);

    /* Compute the upper bound on the residual norm r (See QMR paper p. 13) */
#if defined(PETSC_COMPLEX)
    sprod = sprod*abs(s);
    rnorm = rnorm0 * sqrt((double)it+2.0) * real(sprod);     
#else
    sprod = sprod*fabs(s);
    rnorm = rnorm0 * sqrt((double)it+2.0) * sprod;     
#endif
    it++; if (it > itP->max_it) {break;}
    }

/* Get floating point work */
itP->nmatop += (it * 3);
itP->nvectors += (it) * 34;

/* Need to undo preconditioning here  */
KSPUnwindPre(  itP, x, vtmp );

*its = RCONV(itP,it); return 0;
}

static int KSPiTCQMRSetUp(KSP  itP )
{
  int ierr;
  if (ierr = KSPCheckDef( itP )) return ierr;
  if (ierr = KSPiDefaultGetWork(itP,TCQMR_VECS)) return ierr;
  return 0;
}

int KSPiTCQMRCreate(KSP itP)
{
itP->MethodPrivate = (void *) 0;
itP->method        = KSPTCQMR;
itP->BuildSolution = KSPDefaultBuildSolution;
itP->BuildResidual = KSPDefaultBuildResidual;
itP->setup         = KSPiTCQMRSetUp;
itP->solver        = KSPiTCQMRSolve;
itP->adjustwork    = KSPiDefaultAdjustWork;
itP->destroy       = KSPiDefaultDestroy;
return 0;
}

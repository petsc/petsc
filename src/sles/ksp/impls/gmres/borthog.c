#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: borthog.c,v 1.43 1998/01/06 20:09:17 bsmith Exp bsmith $";
#endif
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "src/ksp/impls/gmres/gmresp.h"
#include <math.h>

/*
    This is the basic orthogonalization routine using modified Gram-Schmidt.
 */
#undef __FUNC__  
#define __FUNC__ "KSPGMRESModifiedGramSchmidtOrthogonalization"
int KSPGMRESModifiedGramSchmidtOrthogonalization( KSP ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j;
  Scalar    *hh, *hes, tmp;

  PetscFunctionBegin;
  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* update Hessenberg matrix and do Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);
  for (j=0; j<=it; j++) {
    /* ( vv(it+1), vv(j) ) */
    VecDot( VEC_VV(it+1), VEC_VV(j), hh );
    *hes++   = *hh;
    /* vv(it+1) <- vv(it+1) - hh[it+1][j] vv(j) */
    tmp = - (*hh++);  VecAXPY(&tmp , VEC_VV(j), VEC_VV(it+1) );
  }
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  PetscFunctionReturn(0);
}

/* This is from
         J.W. Daniel, W.B. Gragg, L. Kaufman, and G.W. Stewart
         Reorthogonalization and Stable Algorithms for Updating the Gram-Schmidt QR Factorization
         Mathematics of Computation, Vol. 30, 136, 1976, pp. 772-795
  This version uses iterative refinement of UNMODIFIED Gram-Schmidt.  
  It can give better performance when running in a parallel 
  environment and in some cases even in a sequential environment (because
  MAXPY has more data reuse).

  Care is taken to accumulate the updated HH/HES values.
 */
int KSPGMRESDGKSOrthogonalization(KSP  ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int        j, m, ncnt;
  Scalar    *hh, *hes, shh[100], *lhh;
  double     omega, theta, alpha, beta, mgamma, delta, epsilon;
  double     sqrit, delta0, delta1, delta3, delta4, delta5, delta6;
  double     rho0, rho1, dnorm;
  int        ierr;

  /* the next line is wrong! */
  rho1 = 0.0;

  PLogEventBegin(KSP_GMRESOrthogonalization, ksp, 0, 0, 0);
  /* Don't allocate small arrays */
  if (it < 100)
    lhh = shh;
  else {
    lhh = (Scalar *) PetscMalloc((it+1) * sizeof(Scalar)); CHKPTRQ(lhh);
  }

  /* Setup parameters -- Should be able to select a scheme */
  ierr = VecNorm(VEC_VV(it+1), NORM_2, &rho0); CHKERRQ(ierr);
  if (1)
  {
    ierr = VecGetSize(VEC_VV(0), &m); CHKERRQ(ierr);
    epsilon = 0.0; /* Assume that we began perfectly orthogonal, we should really update this at each step */
    sqrit   = sqrt(it);
    delta0  = 1.0e-16;
    delta1  = delta0 * pow(1.0 + 1.5*delta0, (double)it);
    /* delta2  = delta0*delta0 * (1.0 + delta0) * pow(1.0 + 1.5*delta0*delta0, (double)it); */
    delta3  = delta0*delta0 * (1.0 + delta0) * pow(1.0 + 1.5*delta0*delta0, (double)m);
    delta4  = delta0 + (m + sqrit) * delta3*delta3 * (1 + 0.5*(3.0*it + 5.0*sqrit)*delta1) * (1.0 + epsilon);
    delta5  = delta0 * (1.0 + delta0) * pow(1.0 + 1.5*delta0, (double)it);
    delta6  = delta0 * pow(1.0 + 1.5*delta0,(double)m) * (1.0 + 0.5*(3.0*it + 5.0*sqrit)*delta1);
    delta   = PetscMax(delta4, PetscMax(delta5, delta6));
    mgamma  = sqrt(1.0 + epsilon);
    beta    = epsilon + 1.5*(sqrit + 1.0)*(sqrit + 1.0)*mgamma*mgamma*delta;
    alpha   = 1.5*mgamma*delta;
    theta   = (sqrt(2.0 - alpha*alpha) + alpha)/(1.0 + alpha*alpha);
    omega   = beta / alpha;
  }
  else
  {
    omega = 0.0;
    theta = sqrt(2.0);
  }

  /* update Hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* Clear hh and hes since we will accumulate values into them */
  for(j = 0; j <= it; j++)
  {
    hh[j]  = 0.0;
    hes[j] = 0.0;
  }

  ncnt = 0;
  do {
    if (ncnt != 0)
      /* Update \norm{v^{k-1}} */
      rho0 = rho1;

    /* 
         This is really a matrix-vector product, with the matrix stored
         as pointer to rows 

    */
    VecMDot(it+1, VEC_VV(it+1), &(VEC_VV(0)), lhh); /* <v,vnew> */

    /*
         This is really a matrix vector product: 
         [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it].
    */
    for(j = 0; j <= it; j++)
      lhh[j] = -lhh[j];
    VecMAXPY(it+1, lhh, VEC_VV(it+1), &VEC_VV(0));
    for(j = 0; j <= it; j++)
    {
      hh[j]  -= lhh[j];     /* hh  +=  <v,vnew> */
      hes[j] += lhh[j];     /* hes += -<v,vnew> */
    }

    /* We want to check that
         \norm{v^{k-1}} + \omega \norm{Q^T v^{k-1}} < \theta \norm{v^k}
       where
       m          : dimension of the vectors
       n          : number of previous vectors = it
       v          : New vector to be made orthogonal to the columns of Q
       Q          : [v_0 ... v_it] matrix with orthogonal columns
       \omega     : \beta / \alpha, but 0 is not unreasonable if \alpha \approx \beta
       \theta     : {\sqrt{2 - \alpha^2} + \alpha \over 1 - \alpha^2} \approx \sqrt{2} for small \alpha
       \alpha     : {3\over2} \mgamma \delta
       \beta      : \epsilon + {3\over2} (\sqrt{n} + 1)^2 \mgamma^2 \delta
       \mgamma    : \sqrt{1 + \epsilon}
       \epsilon   : \norm{Q^T Q - I}_F, we could use any norm here, but Frobenius is easy to calculate
       \delta     : \max{\delta_6, \delta_5, \delta_4}
       \delta_6   : \delta_0 (1 + {3\over2} \delta_0)^m (1 + {3n + 5\sqrt{n} \over 2} \delta_1)
       \delta_5   : \delta_0 (1 + \delta_0) (1 + {3\over2} \delta_0)^n
       \delta_4   : \delta_0 + (m + \sqrt{n}) \delta^2_3 (1 + {3n + 5\sqrt{n} \over 2} \delta_1) (1 + \epsilon)
       \delta^2_3 : \delta^2_0 (1 + \delta_0) (1 + {3\over2} \delta^2_0)^m
       \delta^2_2 : \delta^2_0 (1 + \delta_0) (1 + {3\over2} \delta^2_0)^n
       \delta_1   : \delta_0 (1 + {3\over2} \delta_0)^n
       \delta_0   : Machine precision
    */

    /* Note that dnorm = (norm(d))**2 */
    dnorm = 0.0;
#if defined(USE_PETSC_COMPLEX)
    for (j = 0; j <= it; j++) dnorm += real(lhh[j] * conj(lhh[j]));
#else
    for (j = 0; j <= it; j++) dnorm += lhh[j] * lhh[j];
#endif
    dnorm = sqrt(dnorm);

    /* Get \norm{v^k} */
    ierr = VecNorm(VEC_VV(it+1), NORM_2, &rho1); CHKERRQ(ierr);

    /* Continue until either we have only small corrections or we've done
         as much work as a full orthogonalization (in terms of Mdots) */
    ncnt++;
  } while ((rho0 + omega*dnorm > theta*rho1) && (ncnt < it));

  PLogInfo(ksp,"KSPGMRESDGKSOrthogonalization:Iterative refinement of orthogonalization took %d iterations\n", ncnt);

  if (it >= 100) PetscFree(lhh);
  PLogEventEnd(KSP_GMRESOrthogonalization, ksp, 0, 0, 0);
  PetscFunctionReturn(0);
}



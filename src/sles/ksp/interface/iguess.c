#ifndef lint
static char vcid[] = "$Id: iguess.c,v 1.3 1994/08/19 02:06:38 bsmith Exp $";
#endif

#include "kspimpl.h"
/* 
  This code inplements Paul Fischer's initial guess code for situations where
  a linear system is solved repeatedly 
 */

typedef struct {
    int      curl,     /* Current number of basis vectors */
             maxl;     /* Maximum number of basis vectors */
    double   *alpha;   /* */
    Vec      *xtilde,  /* Saved x vectors */
             *btilde;  /* Saved b vectors */
} KSPIGUESS;

int KSPGuessCreate( itctx, maxl,ITG )
KSP   itctx;
int   maxl;
void  **ITG;
{
  KSPIGUESS *itg;
  *ITG = 0;
  VALIDHEADER(itctx,KSP_COOKIE);
  itg  = NEW(KSPIGUESS);    
  CHKPTR(itg);
  itg->curl = 0;
  itg->maxl = maxl;
  itg->alpha = (double *)MALLOC( maxl * sizeof(double) );  CHKPTR(itg->alpha);
  VecGetVecs(itctx->vec_rhs,maxl,&itg->xtilde);
  VecGetVecs(itctx->vec_rhs,maxl,&itg->btilde);
  *ITG = (void *)itg;
  return 0;
}

int KSPGuessDestroy( itctx, itg )
KSP   itctx;
KSPIGUESS *itg;
{
  VALIDHEADER(itctx,KSP_COOKIE);
  FREE( itg->alpha );
  VecFreeVecs( itg->btilde, itg->maxl );
  VecFreeVecs( itg->xtilde, itg->maxl );
  FREE( itg );
  return 0;
}

int KSPGuessFormB( itctx, itg, b )
KSP       itctx;
KSPIGUESS *itg;
Vec       b;
{
  int i;
  double tmp;
  VALIDHEADER(itctx,KSP_COOKIE);
  for (i=1; i<=itg->curl; i++) {
    VecDot(itg->btilde[i-1],b,&(itg->alpha[i-1]));
    tmp = -itg->alpha[i-1];
    VecAXPY(&tmp,itg->btilde[i-1],b);
  }
  return 0;
}

int KSPGuessFormX( itctx, itg, x )
KSP   itctx;
KSPIGUESS *itg;
Vec x;
{
  int i;
  VALIDHEADER(itctx,KSP_COOKIE);
  VecCopy(x,itg->xtilde[itg->curl]);
  for (i=1; i<=itg->curl; i++) {
    VecAXPY(&itg->alpha[i-1],itg->xtilde[i-1],x);
  }
  return 0;
}

int  KSPGuessUpdate( itctx, x, itg )
KSP    itctx;
Vec       *x;
KSPIGUESS *itg;
{
  double normax, tmp, norm;
  int    curl = itg->curl, i;
  VALIDHEADER(itctx,KSP_COOKIE);
  if (curl == itg->maxl) {
    (*itctx->amult)( itctx->amultP, x, itg->btilde[0] );
    VecNorm(itg->btilde[0],&normax);
    tmp = 1.0/normax; VecScale(&tmp,itg->btilde[0]);
    /* VCOPY(itctx->vc,x,itg->xtilde[0]); */
    VecScale(&tmp,itg->xtilde[0]);
  }
  else {
    (*itctx->amult)( itctx->amultP, itg->xtilde[curl], itg->btilde[curl] );
    for (i=1; i<=curl; i++) 
      VecDot(itg->btilde[curl],itg->btilde[i-1],itg->alpha+i-1);
    for (i=1; i<=curl; i++) {
      tmp = -itg->alpha[i-1];
      VecAXPY(&tmp,itg->btilde[i-1],itg->btilde[curl]);
      VecAXPY(&itg->alpha[i-1],itg->xtilde[i-1],itg->xtilde[curl]);
    }
    VecNorm(itg->btilde[curl],&norm);
    tmp = 1.0/norm; VecScale(&tmp,itg->btilde[curl]);
    VecNorm(itg->xtilde[curl],&norm);
    VecScale(&tmp,itg->xtilde[curl]);
    itg->curl++;
  }
  return 0;
}

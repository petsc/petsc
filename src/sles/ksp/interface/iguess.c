#ifndef lint
static char vcid[] = "$Id: iguess.c,v 1.6 1995/04/12 23:51:06 curfman Exp bsmith $";
#endif

#include "kspimpl.h"  /*I "ksp.h" I*/
/* 
  This code inplements Paul Fischer's initial guess code for situations where
  a linear system is solved repeatedly 
 */

typedef struct {
    int      curl,     /* Current number of basis vectors */
             maxl;     /* Maximum number of basis vectors */
    Scalar   *alpha;   /* */
    Vec      *xtilde,  /* Saved x vectors */
             *btilde;  /* Saved b vectors */
} KSPIGUESS;

int KSPGuessCreate(KSP itctx,int  maxl,void **ITG )
{
  KSPIGUESS *itg;
  *ITG = 0;
  VALIDHEADER(itctx,KSP_COOKIE);
  itg  = NEW(KSPIGUESS);    
  CHKPTR(itg);
  itg->curl = 0;
  itg->maxl = maxl;
  itg->alpha = (Scalar *)MALLOC( maxl * sizeof(Scalar) );  CHKPTR(itg->alpha);
  VecGetVecs(itctx->vec_rhs,maxl,&itg->xtilde);
  PLogObjectParents(itctx,maxl,itg->xtilde);
  VecGetVecs(itctx->vec_rhs,maxl,&itg->btilde);
  PLogObjectParents(itctx,maxl,itg->btilde);
  *ITG = (void *)itg;
  return 0;
}

int KSPGuessDestroy( KSP itctx, KSPIGUESS *itg )
{
  VALIDHEADER(itctx,KSP_COOKIE);
  FREE( itg->alpha );
  VecFreeVecs( itg->btilde, itg->maxl );
  VecFreeVecs( itg->xtilde, itg->maxl );
  FREE( itg );
  return 0;
}

int KSPGuessFormB( KSP itctx, KSPIGUESS *itg, Vec b )
{
  int i;
  Scalar tmp;
  VALIDHEADER(itctx,KSP_COOKIE);
  for (i=1; i<=itg->curl; i++) {
    VecDot(itg->btilde[i-1],b,&(itg->alpha[i-1]));
    tmp = -itg->alpha[i-1];
    VecAXPY(&tmp,itg->btilde[i-1],b);
  }
  return 0;
}

int KSPGuessFormX( KSP itctx, KSPIGUESS *itg, Vec x )
{
  int i;
  VALIDHEADER(itctx,KSP_COOKIE);
  VecCopy(x,itg->xtilde[itg->curl]);
  for (i=1; i<=itg->curl; i++) {
    VecAXPY(&itg->alpha[i-1],itg->xtilde[i-1],x);
  }
  return 0;
}

int  KSPGuessUpdate( KSP itctx, Vec x, KSPIGUESS *itg )
{
  double       normax, norm;
  Scalar       tmp;
  MatStructure pflag;
  int          curl = itg->curl, i;
  Mat          Amat, Pmat;

  VALIDHEADER(itctx,KSP_COOKIE);
  PCGetOperators(itctx->B,&Amat,&Pmat,&pflag);
  if (curl == itg->maxl) {
    MatMult(Amat,x,itg->btilde[0] );
    VecNorm(itg->btilde[0],&normax);
    tmp = 1.0/normax; VecScale(&tmp,itg->btilde[0]);
    /* VCOPY(itctx->vc,x,itg->xtilde[0]); */
    VecScale(&tmp,itg->xtilde[0]);
  }
  else {
    MatMult( Amat, itg->xtilde[curl], itg->btilde[curl] );
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

#ifndef __TAO_IPM_H
#define __TAO_IPM_H
#include "tao-private/taosolver_impl.h"

/* 
 Context for Interior-Point Method
*/

typedef struct {
  PetscInt mi,me,n;
  PetscScalar sig;
  PetscScalar muaff;
  TaoLineSearch lag_ls;
  Vec lag;  /* [x; yi; lamdae; lamdai] */
  Vec work; 
  Vec lamdai;
  Vec lamdae;
  Vec yi;
  PetscScalar kkt_f; /* d'*x + (1/2)*x'*H*x; */
  Vec rd;            /* H*x + d + Ae'*lamdae - Ai'*lamdai */
  Vec rpe; /* residual  Ae*x - be */
  Vec rpi; /*           Ai*x - yi - bi */
  Vec complementarity; /* yi.*lamdai */
  PetscScalar phi;
  MatStructure Hflag; /*flag for nonzero change in hessian */
  MatStructure Aiflag,Aeflag;
} TAO_IPM;

#endif /* ifndef __TAO_IPM_H */

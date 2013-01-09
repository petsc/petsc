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
  Vec work, dx; 
  Vec Xold,Gold;
  Vec lamdai, dlamdai;
  Vec lamdae, dlamdae;
  Vec yi, dyi;
  Vec Yaff,Laff,dYaff, dLaff;
  Vec Zero_mi,Zero_me,Ninf_n,Inf_mi,Inf_me,Inf_n;
  Vec biglb,bigub;
  PetscScalar kkt_f; /* d'*x + (1/2)*x'*H*x; */
  Vec rd;            /* H*x + d + Ae'*lamdae - Ai'*lamdai */
  Vec rpe; /* residual  Ae*x - be */
  Vec rpi; /*           Ai*x - yi - bi */
  Vec complementarity; /* yi.*lamdai */
  PetscScalar phi;
  MatStructure Hflag; /*flag for nonzero change in hessian */
  MatStructure Aiflag,Aeflag;
  Mat L; /* diag(lamdai) */
  Mat Y; /* diag(yi) */
  Mat minusI;
  Mat mAi_T, Ae_T;
  Mat K; /* [ H , 0,   Ae',-Ai']; 
	    [Ae , 0,   0  , 0];
            [Ai ,-Imi, 0 ,  0];  
            [ 0 , L ,  0 ,  Y ];  */

  Vec bigrhs; /* [x; lamdae; yi; lamdai] */
  Vec bigstep; /* [dx; dyi; dlamdae; dlamdai] */
} TAO_IPM;

#endif /* ifndef __TAO_IPM_H */

#ifndef __TAO_IPM_H
#define __TAO_IPM_H
#include "tao-private/taosolver_impl.h"

/* 
 Context for Interior-Point Method
*/

typedef struct {
  PetscInt mi,me,n,nxb,nib,nb,nslack;
  PetscInt nuser_inequalities;
  PetscInt nxlb,nxub,niub,nilb;
  PetscScalar sig,mu,taumin,dec;
  PetscScalar muaff;
  TaoLineSearch lag_ls;
  Vec work, dx, rhs_x,save_x; 
  Vec lamdai, dlamdai, rhs_lamdai, save_lamdai;
  Vec lamdae, dlamdae, rhs_lamdae, save_lamdae;
  Vec s,ds,rhs_s,save_s;
  Vec ci, di;
  Vec Zero_nb, One_nb,Inf_nb,worknb;
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
  Mat Ai;  /* JacI (lb)
	      -JacI (ub)
	      I (xlb)
	      -I (xub) */
  Mat K; /* [ H , 0,   Ae',-Ai']; 
	    [Ae , 0,   0  , 0];
            [Ai ,-Imi, 0 ,  0];  
            [ 0 , L ,  0 ,  Y ];  */

  Vec bigrhs; /* rhs [x; lamdae; yi; lamdai] */
  Vec bigstep; /* [dx; dyi; dlamdae; dlamdai] */
  PetscBool monitorkkt;
  PetscScalar alpha1,alpha2;
  PetscScalar pushs,pushnu;
  IS isxl,isxu,isil,isiu;
} TAO_IPM;

#endif /* ifndef __TAO_IPM_H */

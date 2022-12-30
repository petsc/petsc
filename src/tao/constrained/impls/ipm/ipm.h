#ifndef __TAO_IPM_H
#define __TAO_IPM_H
#include <petsc/private/taoimpl.h>

/*
 Context for Interior-Point Method
*/

typedef struct {
  PetscInt      mi, me, n, nxb, nib, nb, nslack;
  PetscInt      nuser_inequalities;
  PetscInt      nxlb, nxub, niub, nilb;
  PetscScalar   sig, mu, taumin, dec;
  PetscScalar   muaff;
  TaoLineSearch lag_ls;
  Vec           work, rhs_x, save_x;
  Vec           lambdai, dlambdai, rhs_lambdai, save_lambdai;
  Vec           lambdae, dlambdae, rhs_lambdae, save_lambdae;
  Vec           s, ds, rhs_s, save_s;
  Vec           ci;
  Vec           Zero_nb, One_nb, Inf_nb;
  PetscScalar   kkt_f;           /* d'*x + (1/2)*x'*H*x; */
  Vec           rd;              /* H*x + d + Ae'*lambdae - Ai'*lambdai */
  Vec           rpe;             /* residual  Ae*x - be */
  Vec           rpi;             /*           Ai*x - yi - bi */
  Vec           complementarity; /* yi.*lambdai */
  PetscScalar   phi;
  Mat           L;  /* diag(lambdai) */
  Mat           Y;  /* diag(yi) */
  Mat           Ai; /* JacI (lb)
              -JacI (ub)
              I (xlb)
              -I (xub) */
  Mat           K;  /* [ H , 0,   Ae',-Ai'];
            [Ae , 0,   0  , 0];
            [Ai ,-Imi, 0 ,  0];
            [ 0 , L ,  0 ,  Y ];  */

  Vec         bigrhs;  /* rhs [x; lambdae; yi; lambdai] */
  Vec         bigstep; /* [dx; dyi; dlambdae; dlambdai] */
  PetscBool   monitorkkt;
  PetscScalar alpha1, alpha2;
  PetscScalar pushs, pushnu;
  IS          isxl, isxu, isil, isiu;
  VecScatter  ci_scat, xl_scat, xu_scat;
  VecScatter  step1, step2, step3, step4;
  VecScatter  rhs1, rhs2, rhs3, rhs4;
} TAO_IPM;

#endif /* ifndef __TAO_IPM_H */

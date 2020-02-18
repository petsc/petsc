#if !defined(TAOADMM_H)
#define TAOADMM_H
#include <petsc/private/taoimpl.h>

typedef struct _TaoADMMOps *TaoADMMOps;

struct _TaoADMMOps {
  PetscErrorCode (*misfitobjgrad)(Tao, Vec, PetscReal*, Vec, void*);
  PetscErrorCode (*misfithess)(Tao, Vec, Mat, Mat,  void*);
  PetscErrorCode (*misfitjac)(Tao, Vec, Mat, Mat,  void*);
  PetscErrorCode (*regobjgrad)(Tao, Vec, PetscReal*, Vec, void*);
  PetscErrorCode (*reghess)(Tao, Vec, Mat, Mat,  void*);
  PetscErrorCode (*regjac)(Tao, Vec, Mat, Mat,  void*);
};

typedef struct {
  PETSCHEADER(struct _TaoADMMOps);
  Tao                    subsolverX, subsolverZ, parent;
  Vec                    residual,y,yold,y0,yhat,yhatold,constraint;
  Vec                    z,zold,Ax,Bz,Axold,Bzold,Bz0;
  Vec                    workLeft,workJacobianRight,workJacobianRight2; /*Ax,Bz,y,constraint are workJacobianRight sized. workLeft is solution sized */
  Mat                    Hx,Hxpre,Hz,Hzpre,ATA,BTB,JA,JApre,JB,JBpre;
  void*                  regobjgradP;
  void*                  reghessP;
  void*                  regjacobianP;
  void*                  misfitobjgradP;
  void*                  misfithessP;
  void*                  misfitjacobianP;
  PetscReal              gamma,last_misfit_val,last_reg_val,l1epsilon;
  PetscReal              lambda,mu,muold,orthval,mueps,tol,mumin;
  PetscReal              Bzdiffnorm,dualres,resnorm,const_norm;
  PetscReal              gatol_admm,catol_admm;
  PetscInt               T;                                             /* adaptive iteration cutoff */
  PetscBool              xJI, zJI;                                      /* Bool to check whether A,B Jacobians are NULL-set identity */
  PetscBool              Hxchange, Hzchange;                            /* Bool to check whether Hx,Hz change wrt to x and z */
  PetscBool              Hxbool, Hzbool;                                /* Bool to make sure Hessian gets updated only once for Hchange False case */
  TaoADMMUpdateType      update;                                        /* update policy for mu */
  TaoADMMRegularizerType regswitch;                                     /* regularization policy */
} TAO_ADMM;

#endif

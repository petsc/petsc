#define PETSCMAT_DLL
/*
    ADIC based nonlinear operator object that can be used with FAS

    This does not really belong in the matrix directories but since it 
    was cloned off of Mat_DAAD I'm leaving it here until I have a better place

*/
#include "petscsys.h"
#include "petscda.h"

EXTERN_C_BEGIN
#include "adic/ad_utils.h"
EXTERN_C_END

#include "../src/dm/da/daimpl.h"
#include "../src/mat/blockinvert.h"

struct NLF_DAAD {
  DA         da;
  void       *ctx;
  Vec        residual;
  int        newton_its;
};

/*
      Solves the one dimensional equation using Newton's method 
*/
#undef __FUNCT__  
#define __FUNCT__ "NLFNewton_DAAD"
PetscErrorCode NLFNewton_DAAD(NLF A,DALocalInfo *info,MatStencil *stencil,void *ad_vu,PetscScalar *ad_vustart,int nI,int gI,PetscScalar residual)
{
  PetscErrorCode ierr;
  PetscInt       cnt = A->newton_its;
  PetscScalar    ad_f[2],J,f;

  PetscFunctionBegin;
  ad_vustart[1+2*gI] = 1.0;
  do {
    /* compute the function and Jacobian */        
    ierr = (*A->da->adicmf_lfi)(info,stencil,ad_vu,ad_f,A->ctx);CHKERRQ(ierr);
    J    = -ad_f[1];
    f    = -ad_f[0] + residual;
    if (f != f) SETERRQ(1,"nan");
    ad_vustart[2*gI] =  ad_vustart[2*gI] - f/J;
  } while (--cnt > 0 && PetscAbsScalar(f) > 1.e-14);

  ad_vustart[1+2*gI] = 0.0;
  PetscFunctionReturn(0);
}

/*
      Solves the four dimensional equation using Newton's method 
*/
#undef __FUNCT__  
#define __FUNCT__ "NLFNewton_DAAD4"
PetscErrorCode NLFNewton_DAAD4(NLF A,DALocalInfo *info,MatStencil *stencil,void *ad_vu,PetscScalar *ad_vustart,int nI,int gI,PetscScalar *residual)
{
  PetscErrorCode ierr;
  PetscInt       cnt = A->newton_its;
  PetscScalar    ad_f[20], J[16],f[4], res, dd[5];

  PetscFunctionBegin;

  /* This sets the identity as the seed matrix for ADIC */   
    CHKMEMQ;
  ad_vustart[1+5*gI   ] = 1.0;
    CHKMEMQ;
  ad_vustart[2+5*gI+5 ] = 1.0;
    CHKMEMQ;
  ad_vustart[3+5*gI+10] = 1.0;
    CHKMEMQ;
  ad_vustart[4+5*gI+15] = 1.0;
    CHKMEMQ;

  do {
    /* compute the function and Jacobian */        
    CHKMEMQ;
    ierr = (*A->da->adicmf_lfib)(info,stencil,ad_vu,ad_f,A->ctx);CHKERRQ(ierr);
       CHKMEMQ;
    /* copy ADIC formated Jacobian into regular C array */
    J[0] = ad_f[1] ; J[1] = ad_f[2] ; J[2] = ad_f[3] ; J[3] = ad_f[4] ;
    J[4] = ad_f[6] ; J[5] = ad_f[7] ; J[6] = ad_f[8] ; J[7] = ad_f[9] ;
    J[8] = ad_f[11]; J[9] = ad_f[12]; J[10]= ad_f[13]; J[11]= ad_f[14];
    J[12]= ad_f[16]; J[13]= ad_f[17]; J[14]= ad_f[18]; J[15]= ad_f[19];
    CHKMEMQ;
    f[0]    = -ad_f[0]   + residual[0];
    f[1]    = -ad_f[5]   + residual[1];
    f[2]    = -ad_f[10]  + residual[2];
    f[3]    = -ad_f[15]  + residual[3];

    /* solve Jacobian * dd = ff */

    /* could use PETSc kernel code to solve system with pivoting */

    /* could put code in here to compute the solution directly using ADIC data structures instead of copying first */
    dd[0]=J[0]*(J[5]*(J[10]*J[15]-J[11]*J[14])-J[6]*(J[9]*J[15]-J[11]*J[13])+J[7]*(J[9]*J[14]-J[10]*J[13]))-
          J[1]*(J[4]*(J[10]*J[15]-J[11]*J[14])-J[6]*(J[8]*J[15]-J[11]*J[12])+J[7]*(J[8]*J[14]-J[10]*J[12]))+
          J[2]*(J[4]*(J[ 9]*J[15]-J[11]*J[13])-J[5]*(J[8]*J[15]-J[11]*J[12])+J[7]*(J[8]*J[13]-J[ 9]*J[12]))-
          J[3]*(J[4]*(J[ 9]*J[14]-J[10]*J[13])-J[5]*(J[8]*J[14]-J[10]*J[12])+J[6]*(J[8]*J[13]-J[ 9]*J[12]));

    dd[1]=(f[0]*(J[5]*(J[10]*J[15]-J[11]*J[14])-J[6]*(J[9]*J[15]-J[11]*J[13])+J[7]*(J[9]*J[14]-J[10]*J[13]))-
          J[1]*(f[1]*(J[10]*J[15]-J[11]*J[14])-J[6]*(f[2]*J[15]-J[11]*f[ 3])+J[7]*(f[2]*J[14]-J[10]*f[ 3]))+
          J[2]*(f[1]*(J[ 9]*J[15]-J[11]*J[13])-J[5]*(f[2]*J[15]-J[11]*f[ 3])+J[7]*(f[2]*J[13]-J[ 9]*f[ 3]))-
	   J[3]*(f[1]*(J[ 9]*J[14]-J[10]*J[13])-J[5]*(f[2]*J[14]-J[10]*f[ 3])+J[6]*(f[2]*J[13]-J[ 9]*f[ 3])))/dd[0];

    dd[2]=(J[0]*(f[1]*(J[10]*J[15]-J[11]*J[14])-J[6]*(f[2]*J[15]-J[11]*f[ 3])+J[7]*(f[2]*J[14]-J[10]*f[ 3]))-
          f[0]*(J[4]*(J[10]*J[15]-J[11]*J[14])-J[6]*(J[8]*J[15]-J[11]*J[12])+J[7]*(J[8]*J[14]-J[10]*J[12]))+
          J[2]*(J[4]*(f[ 2]*J[15]-J[11]*f[ 3])-f[1]*(J[8]*J[15]-J[11]*J[12])+J[7]*(J[8]*f[ 3]-f[ 2]*J[12]))-
	  J[3]*(J[4]*(f[ 2]*J[14]-J[10]*f[ 3])-f[2]*(J[8]*J[14]-J[10]*J[12])+J[6]*(J[8]*f[ 3]-f[ 2]*J[12])))/dd[0];

    dd[3]=(J[0]*(J[5]*(f[ 2]*J[15]-J[11]*f[ 3])-f[1]*(J[9]*J[15]-J[11]*J[13])+J[7]*(J[9]*f[ 3]-f[ 2]*J[13]))-
          J[1]*(J[4]*(f[ 2]*J[15]-J[11]*f[ 3])-f[1]*(J[8]*J[15]-J[11]*J[12])+J[7]*(J[8]*f[ 3]-f[ 2]*J[12]))+
          f[0]*(J[4]*(J[ 9]*J[15]-J[11]*J[13])-J[5]*(J[8]*J[15]-J[11]*J[12])+J[7]*(J[8]*J[13]-J[ 9]*J[12]))-
	   J[3]*(J[4]*(J[ 9]*f[ 3]-f[ 2]*J[13])-J[5]*(J[8]*f[ 3]-f[ 2]*J[12])+f[1]*(J[8]*J[13]-J[ 9]*J[12])))/dd[0];

    dd[4]=(J[0]*(J[5]*(J[10]*f[ 3]-f[ 2]*J[14])-J[6]*(J[9]*f[ 3]-f[ 2]*J[13])+f[1]*(J[9]*J[14]-J[10]*J[13]))-
          J[1]*(J[4]*(J[10]*f[ 3]-f[ 2]*J[14])-J[6]*(J[8]*f[ 3]-f[ 2]*J[12])+f[1]*(J[8]*J[14]-J[10]*J[12]))+
          J[2]*(J[4]*(J[ 9]*f[ 3]-f[ 2]*J[13])-J[5]*(J[8]*f[ 3]-f[ 2]*J[12])+f[1]*(J[8]*J[13]-J[ 9]*J[12]))-
	  f[0]*(J[4]*(J[ 9]*J[14]-J[10]*J[13])-J[5]*(J[8]*J[14]-J[10]*J[12])+J[6]*(J[8]*J[13]-J[ 9]*J[12])))/dd[0];
    CHKMEMQ;
    /* copy solution back into ADIC data structure */
    ad_vustart[5*(gI+0)] += dd[1];
    ad_vustart[5*(gI+1)] += dd[2];
    ad_vustart[5*(gI+2)] += dd[3];
    ad_vustart[5*(gI+3)] += dd[4];
    CHKMEMQ;
    res =  f[0]*f[0]; 
    res += f[1]*f[1]; 
    res += f[2]*f[2]; 
    res += f[3]*f[3]; 
    res =  sqrt(res);
  
  } while (--cnt > 0 && res > 1.e-14);

  /* zero out this part of the seed matrix that was set initially */
  ad_vustart[1+5*gI   ] = 0.0;
  ad_vustart[2+5*gI+5 ] = 0.0;
  ad_vustart[3+5*gI+10] = 0.0;
  ad_vustart[4+5*gI+15] = 0.0;
    CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "NLFNewton_DAAD9"
PetscErrorCode NLFNewton_DAAD9(NLF A,DALocalInfo *info,MatStencil *stencil,void *ad_vu,PetscScalar *ad_vustart,int nI,int gI,PetscScalar *residual)
{
  PetscErrorCode ierr;
  PetscInt       cnt = A->newton_its;
  PetscScalar    ad_f[100], J[81],f[9], res;
  PetscInt       i,j,ngI[9];             
  PetscFunctionBegin;
   
  // the order of the nodes
   /*
         (6)      (7)         (8)
      i-1,j+1 --- i,j+1 --- i+1,j+1
        |         |           |
	|         |           |
      i-1,j   --- i,j  --- i+1,j
        |(3)      |(4)        |(5)
	|         |           |
      i-1,j-1 --- i,j-1--- i+1,j-1
       (0)       (1)         (2)
  */
  
  // the order of the derivative for the center nodes
   /*
         (7)      (8)         (9)
      i-1,j+1 --- i,j+1 --- i+1,j+1
        |         |           |
	|         |           |
      i-1,j   --- i,j  --- i+1,j
        |(4)      |(5)        |(6)
	|         |           |
      i-1,j-1 --- i,j-1--- i+1,j-1
       (1)       (2)         (3)
  */
  if( (*stencil).i==0 || (*stencil).i==1||(*stencil).i==(*info).gxs+(*info).gxm-1 || (*stencil).i==(*info).gxs+(*info).gxm-2  || (*stencil).j==0 ||  (*stencil).j==1 ||(*stencil).j==(*info).gys+(*info).gym-1 || (*stencil).j==(*info).gys+(*info).gym -2) {

  ad_vustart[1+10*gI] = 1.0;
 
  do {
    /* compute the function and Jacobian */        
    ierr = (*A->da->adicmf_lfi)(info,stencil,ad_vu,ad_f,A->ctx);CHKERRQ(ierr);
    J[0]    = -ad_f[1];
    f[0]    = -ad_f[0] + residual[gI];
    ad_vustart[10*gI] =  ad_vustart[10*gI] - f[0]/J[0];
  } while (--cnt > 0 && PetscAbsScalar(f[0]) > 1.e-14);

  ad_vustart[1+10*gI] = 0.0;
  PetscFunctionReturn(0);


  }
  
  ngI[0]  =  ((*stencil).i -1 - (*info).gxs)*(*info).dof + ((*stencil).j -1 - (*info).gys)*(*info).dof*(*info).gxm + ((*stencil).k - (*info).gzs)*(*info).dof*(*info).gxm*(*info).gym;  
  ngI[1]  =  ngI[0] + 1;
  ngI[2]  =  ngI[1] + 1;
  ngI[3]  =  gI     - 1;
  ngI[4]  =  gI        ;
  ngI[5]  =  gI     + 1;
  ngI[6]  =  ((*stencil).i -1 - (*info).gxs)*(*info).dof + ((*stencil).j +1 - (*info).gys)*(*info).dof*(*info).gxm + ((*stencil).k - (*info).gzs)*(*info).dof*(*info).gxm*(*info).gym;  
  ngI[7]  =  ngI[6] + 1;
  ngI[8]  =  ngI[7] + 1;
 

  for(j=0 ; j<9; j++){
    ad_vustart[ngI[j]*10+j+1] = 1.0;
  }
  
  do{
    /* compute the function and the Jacobian */
    
    ierr = (*A->da->adicmf_lfi)(info,stencil,ad_vu,ad_f,A->ctx);CHKERRQ(ierr);
 
    for(i=0; i<9; i++){
      for(j=0; j<9; j++){
        J[i*9+j] = -ad_f[i*10+j+1];
      }
      f[i]= -ad_f[i*10] + residual[ngI[i]];
    }

    Kernel_A_gets_inverse_A_9(J,0.0);
 
    res =0 ;
    for(i=0; i<9; i++){
      for(j=0;j<9;j++){
        ad_vustart[10*ngI[i]]= ad_vustart[10*ngI[i]] - J[i*9 +j]*f[j];
      }
      res = res + f[i]*f[i];
    }
    res = sqrt(res); 
  } while(--cnt>0 && res>1.e-14);

  for(j=0; j<9; j++){
    ad_vustart[10*ngI[j]+j+1]=0.0;
  }

  PetscFunctionReturn(0);
}

/*
        Nonlinear relax on all the equations with an initial guess in x
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFMatSOR_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFRelax_DAAD(NLF A,MatSORType flag,int its,Vec xx)
{
  PetscErrorCode ierr;
  PetscInt       j,gtdof,nI,gI;
  PetscScalar    *avu,*av,*ad_vustart,*residual;
  Vec            localxx;
  DALocalInfo    info;
  MatStencil     stencil;
  void*          *ad_vu;

  PetscFunctionBegin;
  if (its <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D positive",its);

  ierr = DAGetLocalVector(A->da,&localxx);CHKERRQ(ierr);
  /* get space for derivative object.  */
  ierr = DAGetAdicMFArray(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = VecGetArray(A->residual,&residual);CHKERRQ(ierr);


  /* tell ADIC we will be computing one dimensional Jacobians */
  PetscADResetIndep();
  ierr = PetscADIncrementTotalGradSize(1);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = DAGetLocalInfo(A->da,&info);CHKERRQ(ierr);
  while (its--) {

    /* get initial solution properly ghosted */
    ierr = DAGlobalToLocalBegin(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);

    /* copy input vector into derivative object */
    ierr = VecGetArray(localxx,&avu);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      ad_vustart[2*j]   = avu[j];
      ad_vustart[2*j+1] = 0.0;
    }
     
    ierr = VecRestoreArray(localxx,&avu);CHKERRQ(ierr);

    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      nI = 0;
      for (stencil.k = info.zs; stencil.k<info.zs+info.zm; stencil.k++) {
        for (stencil.j = info.ys; stencil.j<info.ys+info.ym; stencil.j++) {
          for (stencil.i = info.xs; stencil.i<info.xs+info.xm; stencil.i++) {
            for (stencil.c = 0; stencil.c<info.dof; stencil.c++) {
              gI   = stencil.c + (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
              ierr = NLFNewton_DAAD(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual[nI]);CHKERRQ(ierr);
              nI++;
            }
          }
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      nI = info.dof*info.xm*info.ym*info.zm - 1;
      for (stencil.k = info.zs+info.zm-1; stencil.k>=info.zs; stencil.k--) {
        for (stencil.j = info.ys+info.ym-1; stencil.j>=info.ys; stencil.j--) {
          for (stencil.i = info.xs+info.xm-1; stencil.i>=info.xs; stencil.i--) {
            for (stencil.c = info.dof-1; stencil.c>=0; stencil.c--) {
              gI   = stencil.c + (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
              ierr = NLFNewton_DAAD(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual[nI]);CHKERRQ(ierr);
              nI--;
            }
          }
        }
      }
    }

    /* copy solution back into ghosted vector from derivative object */
    ierr = VecGetArray(localxx,&av);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      av[j] = ad_vustart[2*j];
    }
    ierr = VecRestoreArray(localxx,&av);CHKERRQ(ierr);
    /* stick relaxed solution back into global solution */
    ierr = DALocalToGlobal(A->da,localxx,INSERT_VALUES,xx);CHKERRQ(ierr);
  }


  ierr = VecRestoreArray(A->residual,&residual);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(A->da,&localxx);CHKERRQ(ierr);
  ierr = DARestoreAdicMFArray(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSOR_DAAD4"
PetscErrorCode PETSCMAT_DLLEXPORT NLFRelax_DAAD4(NLF A,MatSORType flag,int its,Vec xx)
{
  PetscErrorCode ierr;
  PetscInt       j,gtdof,nI,gI;
  PetscScalar    *avu,*av,*ad_vustart,*residual;
  Vec            localxx;
  DALocalInfo    info;
  MatStencil     stencil;
  void*          *ad_vu;

  PetscFunctionBegin;
  if (its <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D positive",its);
  
  ierr = DAGetLocalVector(A->da,&localxx);CHKERRQ(ierr);
  /* get space for derivative object.  */
  ierr = DAGetAdicMFArray4(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = VecGetArray(A->residual,&residual);CHKERRQ(ierr);


  /* tell ADIC we will be computing four dimensional Jacobians */
  PetscADResetIndep();
  ierr = PetscADIncrementTotalGradSize(4);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = DAGetLocalInfo(A->da,&info);CHKERRQ(ierr);
  while (its--) {

    /* get initial solution properly ghosted */
    ierr = DAGlobalToLocalBegin(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);

    /* copy input vector into derivative object */
    ierr = VecGetArray(localxx,&avu);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      ad_vustart[5*j  ]   = avu[j];
      ad_vustart[5*j+1]   = 0.0;
      ad_vustart[5*j+2]   = 0.0;
      ad_vustart[5*j+3]   = 0.0;
      ad_vustart[5*j+4]   = 0.0;
    }
    
    ierr = VecRestoreArray(localxx,&avu);CHKERRQ(ierr);

    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      nI = 0;
      for (stencil.k = info.zs; stencil.k<info.zs+info.zm; stencil.k++) {
        for (stencil.j = info.ys; stencil.j<info.ys+info.ym; stencil.j++) {
          for (stencil.i = info.xs; stencil.i<info.xs+info.xm; stencil.i++) {
	    CHKMEMQ;
            gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;  

            ierr = NLFNewton_DAAD4(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual+nI);CHKERRQ(ierr);
            nI=nI+4;  
	    CHKMEMQ;
          }
        }
      }
    } 
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      nI = info.dof*info.xm*info.ym*info.zm - 4;
 
      for (stencil.k = info.zs+info.zm-1; stencil.k>=info.zs; stencil.k--) {
        for (stencil.j = info.ys+info.ym-1; stencil.j>=info.ys; stencil.j--) {
          for (stencil.i = info.xs+info.xm-1; stencil.i>=info.xs; stencil.i--) {
            gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
            ierr = NLFNewton_DAAD4(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual+nI);CHKERRQ(ierr);
            nI=nI-4;          
          }
        }
      }
    }
   
    /* copy solution back into ghosted vector from derivative object */
    ierr = VecGetArray(localxx,&av);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      av[j] = ad_vustart[5*j];
    }
    ierr = VecRestoreArray(localxx,&av);CHKERRQ(ierr);
    /* stick relaxed solution back into global solution */
    ierr = DALocalToGlobal(A->da,localxx,INSERT_VALUES,xx);CHKERRQ(ierr);
  }


  ierr = VecRestoreArray(A->residual,&residual);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(A->da,&localxx);CHKERRQ(ierr);
  ierr = DARestoreAdicMFArray(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_DAAD9"
PetscErrorCode PETSCMAT_DLLEXPORT NLFRelax_DAAD9(NLF A,MatSORType flag,int its,Vec xx)
{
  PetscErrorCode ierr;
  PetscInt       j,gtdof,nI,gI;
  PetscScalar    *avu,*av,*ad_vustart,*residual;
  Vec            localxx;
  DALocalInfo    info;
  MatStencil     stencil;
  void*          *ad_vu;

  PetscFunctionBegin;
  if (its <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D positive",its);
  
  ierr = DAGetLocalVector(A->da,&localxx);CHKERRQ(ierr);
  /* get space for derivative object.  */
  ierr = DAGetAdicMFArray9(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = VecGetArray(A->residual,&residual);CHKERRQ(ierr);


  /* tell ADIC we will be computing nine dimensional Jacobians */
  PetscADResetIndep();
  ierr = PetscADIncrementTotalGradSize(9);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = DAGetLocalInfo(A->da,&info);CHKERRQ(ierr);
  while (its--) {

    /* get initial solution properly ghosted */
    ierr = DAGlobalToLocalBegin(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);

    /* copy input vector into derivative object */
    ierr = VecGetArray(localxx,&avu);CHKERRQ(ierr);
     for (j=0; j<gtdof; j++) {
      ad_vustart[10*j  ]   = avu[j];
      ad_vustart[10*j+1]   = 0.0;
      ad_vustart[10*j+2]   = 0.0;
      ad_vustart[10*j+3]   = 0.0;
      ad_vustart[10*j+4]   = 0.0;
      ad_vustart[10*j+5]   = 0.0;
      ad_vustart[10*j+6]   = 0.0;
      ad_vustart[10*j+7]   = 0.0;
      ad_vustart[10*j+8]   = 0.0;
      ad_vustart[10*j+9]   = 0.0;
    }
    
    ierr = VecRestoreArray(localxx,&avu);CHKERRQ(ierr);

    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      nI = 0;
      for (stencil.k = info.zs; stencil.k<info.zs+info.zm; stencil.k++) {
        for (stencil.j = info.ys; stencil.j<info.ys+info.ym; stencil.j++) {
          for (stencil.i = info.xs; stencil.i<info.xs+info.xm; stencil.i++) {
	    CHKMEMQ;
              gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;  
 ierr = NLFNewton_DAAD9(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual);CHKERRQ(ierr);
 nI=nI+1;  
	    CHKMEMQ;
          }
        }
      }
    } 
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      nI = info.dof*info.xm*info.ym*info.zm - 4;
 
      for (stencil.k = info.zs+info.zm-1; stencil.k>=info.zs; stencil.k--) {
        for (stencil.j = info.ys+info.ym-1; stencil.j>=info.ys; stencil.j--) {
          for (stencil.i = info.xs+info.xm-1; stencil.i>=info.xs; stencil.i--) {
              gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
	      ierr = NLFNewton_DAAD9(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual);CHKERRQ(ierr);
    nI=nI-1;          
          }
        }
      }
    }
   
    /* copy solution back into ghosted vector from derivative object */
    ierr = VecGetArray(localxx,&av);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      av[j] = ad_vustart[10*j];
    }
    ierr = VecRestoreArray(localxx,&av);CHKERRQ(ierr);
    /* stick relaxed solution back into global solution */
    ierr = DALocalToGlobal(A->da,localxx,INSERT_VALUES,xx);CHKERRQ(ierr);
  }


  ierr = VecRestoreArray(A->residual,&residual);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(A->da,&localxx);CHKERRQ(ierr);
  ierr = DARestoreAdicMFArray(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
        Point-block nonlinear relax on all the equations with an initial guess in xx using 
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFMatSOR_DAADb"
PetscErrorCode PETSCMAT_DLLEXPORT NLFRelax_DAADb(NLF A,MatSORType flag,int its,Vec xx)
{
  PetscErrorCode ierr;
  PetscInt       i,j,gtdof,nI,gI, bs = A->da->w, bs1 = bs + 1;
  PetscScalar    *avu,*av,*ad_vustart,*residual;
  Vec            localxx;
  DALocalInfo    info;
  MatStencil     stencil;
  void*          *ad_vu;
  PetscErrorCode (*NLFNewton_DAADb)(NLF,DALocalInfo*,MatStencil*,void*,PetscScalar*,int,int,PetscScalar*);

  PetscFunctionBegin;
  if (its <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D positive",its);
  if (bs == 4) {
    NLFNewton_DAADb       = NLFNewton_DAAD4;
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Point block nonlinear relaxation currently not for this block size",bs);
  }

  ierr = DAGetLocalVector(A->da,&localxx);CHKERRQ(ierr);
  /* get space for derivative object.  */
  ierr = DAGetAdicMFArrayb(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = VecGetArray(A->residual,&residual);CHKERRQ(ierr);


  /* tell ADIC we will be computing bs dimensional Jacobians */
  PetscADResetIndep();
  ierr = PetscADIncrementTotalGradSize(bs);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = DAGetLocalInfo(A->da,&info);CHKERRQ(ierr);
  while (its--) {

    /* get initial solution properly ghosted */
    ierr = DAGlobalToLocalBegin(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);

    /* copy input vector into derivative object */
    ierr = VecGetArray(localxx,&avu);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      ad_vustart[bs1*j]   = avu[j];
      for (i=0; i<bs; i++) {
         ad_vustart[bs1*j+1+i] = 0.0;
      }
    }
    ierr = VecRestoreArray(localxx,&avu);CHKERRQ(ierr);

    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      nI = 0;
      for (stencil.k = info.zs; stencil.k<info.zs+info.zm; stencil.k++) {
        for (stencil.j = info.ys; stencil.j<info.ys+info.ym; stencil.j++) {
          for (stencil.i = info.xs; stencil.i<info.xs+info.xm; stencil.i++) {
            gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
            ierr = (*NLFNewton_DAADb)(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual+nI);CHKERRQ(ierr);
            nI += bs;
          }
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      nI = info.dof*info.xm*info.ym*info.zm - bs;
      for (stencil.k = info.zs+info.zm-1; stencil.k>=info.zs; stencil.k--) {
        for (stencil.j = info.ys+info.ym-1; stencil.j>=info.ys; stencil.j--) {
          for (stencil.i = info.xs+info.xm-1; stencil.i>=info.xs; stencil.i--) {
            gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
            ierr = (*NLFNewton_DAADb)(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual+nI);CHKERRQ(ierr);
            nI -= bs;
          }
        }
      }
    }

    /* copy solution back into ghosted vector from derivative object */
    ierr = VecGetArray(localxx,&av);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      av[j] = ad_vustart[bs1*j];
    }
    ierr = VecRestoreArray(localxx,&av);CHKERRQ(ierr);
    /* stick relaxed solution back into global solution */
    ierr = DALocalToGlobal(A->da,localxx,INSERT_VALUES,xx);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(A->residual,&residual);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(A->da,&localxx);CHKERRQ(ierr);
  ierr = DARestoreAdicMFArray(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "NLFDestroy_DAAD"
PetscErrorCode NLFDestroy_DAAD(NLF A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DADestroy(A->da);CHKERRQ(ierr);
  ierr = PetscFree(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFDAADSetDA_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFDAADSetDA_DAAD(NLF A,DA da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)da);CHKERRQ(ierr);
  if (A->da) {ierr = DADestroy(A->da);CHKERRQ(ierr);}
  A->da = da;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFDAADSetNewtonIterations_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFDAADSetNewtonIterations_DAAD(NLF A,int its)
{
  PetscFunctionBegin;
  A->newton_its = its;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFDAADSetResidual_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFDAADSetResidual_DAAD(NLF A,Vec residual)
{
  PetscFunctionBegin;
  A->residual = residual;
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFDAADSetCtx_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFDAADSetCtx_DAAD(NLF A,void *ctx)
{
  PetscFunctionBegin;
  A->ctx = ctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFCreate_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFCreate_DAAD(NLF *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr    = PetscNew(struct NLF_DAAD,A);CHKERRQ(ierr);
  (*A)->da         = 0;
  (*A)->ctx        = 0;
  (*A)->newton_its = 2;
  PetscFunctionReturn(0);
}
EXTERN_C_END




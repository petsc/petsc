#define PETSCMAT_DLL
/*
    ADIC based nonlinear operator object that can be used with FAS

    This does not really belong in the matrix directories but since it 
    was cloned off of Mat_DAAD I'm leaving it here until I have a better place

*/
#include "petsc.h"
#include "petscda.h"
#include "petscsys.h"

EXTERN_C_BEGIN
#include "adic/ad_utils.h"
EXTERN_C_END


#include "src/dm/da/daimpl.h"



struct NLF_DAAD {
  DA         da;
  void       *ctx;
  Vec        residual;
  int        newton_its;
};

#undef __FUNCT__  
#define __FUNCT__ "kernel_A_gets_inverse_A_4"
PetscErrorCode kernel_A_gets_inverse_A_4(MatScalar *a)
{
    PetscInt   i__2,i__3,kp1,j,k,l,ll,i,ipvt[4],kb,k3;
    PetscInt   k4,j3;
    MatScalar  *aa,*ax,*ay,work[16],stmp;
    MatReal    tmp,max;

/*     gaussian elimination with partial pivoting */
    PetscFunctionBegin;
    /* Parameter adjustments */
    a       -= 5;

    for (k = 1; k <= 3; ++k) {
        kp1 = k + 1;
        k3  = 4*k;
        k4  = k3 + k;
/*        find l = pivot index */

        i__2 = 4 - k;
        aa = &a[k4];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for (ll=1; ll<i__2; ll++) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l       += k - 1;
        ipvt[k-1] = l;

        if (a[l + k3] == 0.0) {
	  SETERRQ1(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",k-1);
        }

/*           interchange if necessary */

        if (l != k) {
          stmp      = a[l + k3];
          a[l + k3] = a[k4];
          a[k4]     = stmp;
        }

/*           compute multipliers */

        stmp = -1. / a[k4];
        i__2 = 4 - k;
        aa = &a[1 + k4]; 
        for (ll=0; ll<i__2; ll++) {
          aa[ll] *= stmp;
        }

/*           row elimination with column indexing */

        ax = &a[k4+1]; 
        for (j = kp1; j <= 4; ++j) {
            j3   = 4*j;
            stmp = a[l + j3];
            if (l != k) {
              a[l + j3] = a[k + j3];
              a[k + j3] = stmp;
            }

            i__3 = 4 - k;
            ay = &a[1+k+j3];
            for (ll=0; ll<i__3; ll++) {
              ay[ll] += stmp*ax[ll];
            }
        }
    }
    ipvt[3] = 4;
    if (a[20] == 0.0) {
      SETERRQ1(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",3);
    }

    /*
         Now form the inverse 
    */
 
   /*     compute inverse(u) */

    for (k = 1; k <= 4; ++k) {
        k3    = 4*k;
        k4    = k3 + k;
        a[k4] = 1.0 / a[k4];
        stmp  = -a[k4];
        i__2  = k - 1;
        aa    = &a[k3 + 1]; 
        for (ll=0; ll<i__2; ll++) aa[ll] *= stmp;
        kp1 = k + 1;
        if (4 < kp1) continue;
        ax = aa;
        for (j = kp1; j <= 4; ++j) {
            j3        = 4*j;
            stmp      = a[k + j3];
            a[k + j3] = 0.0;
            ay        = &a[j3 + 1];
            for (ll=0; ll<k; ll++) {
              ay[ll] += stmp*ax[ll];
            }
        }
    }

   /*    form inverse(u)*inverse(l) */

    for (kb = 1; kb <= 3; ++kb) {
        k   = 4 - kb;
        k3  = 4*k;
        kp1 = k + 1;
        aa  = a + k3;
        for (i = kp1; i <= 4; ++i) {
            work[i-1] = aa[i];
            aa[i]   = 0.0;
        }
        for (j = kp1; j <= 4; ++j) {
            stmp  = work[j-1];
            ax    = &a[4*j + 1];
            ay    = &a[k3 + 1];
            ay[0] += stmp*ax[0];
            ay[1] += stmp*ax[1];
            ay[2] += stmp*ax[2];
            ay[3] += stmp*ax[3];
        }
        l = ipvt[k-1];
        if (l != k) {
            ax = &a[k3 + 1]; 
            ay = &a[4*l + 1];
            stmp = ax[0]; ax[0] = ay[0]; ay[0] = stmp;
            stmp = ax[1]; ax[1] = ay[1]; ay[1] = stmp;
            stmp = ax[2]; ax[2] = ay[2]; ay[2] = stmp;
            stmp = ax[3]; ax[3] = ay[3]; ay[3] = stmp;
        }
    }

    PetscFunctionReturn(0);
}


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
    ad_vustart[2*gI] =  ad_vustart[2*gI] - f/J;
  } while (--cnt > 0 && PetscAbsScalar(f) > 1.e-14);

  ad_vustart[1+2*gI] = 0.0;
  PetscFunctionReturn(0);
}

/*
      Solves the four dimensionals equation using Newton's method 
*/
#undef __FUNCT__  
#define __FUNCT__ "NLFNewton_DAAD4"
PetscErrorCode NLFNewton_DAAD4(NLF A,DALocalInfo *info,MatStencil *stencil,void *ad_vu,PetscScalar *ad_vustart,int nI,int gI,PetscScalar *residual)
//PetscErrorCode NLFNewton_DAAD4(NLF A,DALocalInfo *info,MatStencil *stencil,void *ad_vu,PetscScalar *ad_vustart,int nI,int gI,PetscScalar res1,PetscScalar res2,PetscScalar res3, PetscScalar res4)
{
  PetscErrorCode ierr;
  PetscInt       cnt = A->newton_its;
  PetscScalar    ad_f[20], J[16],f[4], res, dd[5];
  PetscInt       i,j;             
  PetscFunctionBegin;
   
  ad_vustart[1+5*gI   ] = 1.0;
  ad_vustart[2+5*gI+5 ] = 1.0;
  ad_vustart[3+5*gI+10] = 1.0;
  ad_vustart[4+5*gI+15] = 1.0;

  do {
    /* compute the function and Jacobian */        
    ierr = (*A->da->adicmf_lfi)(info,stencil,ad_vu,ad_f,A->ctx);CHKERRQ(ierr);
   
    J[0]= -ad_f[1] ; J[1] = -ad_f[2] ; J[2]= -ad_f[3] ; J[3]= -ad_f[4] ;
    J[4]= -ad_f[6] ; J[5] = -ad_f[7] ; J[6]= -ad_f[8] ; J[7]= -ad_f[9] ;
    J[8]= -ad_f[11]; J[9] = -ad_f[12]; J[10]= -ad_f[13]; J[11]= -ad_f[14];
    J[12]= -ad_f[16]; J[13]= -ad_f[17]; J[14]= -ad_f[18]; J[15]= -ad_f[19];
    //  ierr = PetscPrintf(PETSC_COMM_WORLD,"gI=%d,J31=%g,J32=%g,J41=%g,J42=%g\n",gI,J[8]/J[10],J[9]/J[10],J[12]/J[15],J[13]/J[15]);CHKERRQ(ierr);
    f[0]    = -ad_f[0]   + residual[0];
    f[1]    = -ad_f[5]   + residual[1];
    f[2]    = -ad_f[10]  + residual[2];
    f[3]    = -ad_f[15]  + residual[3];
   
    
   
    /*  kernel_A_gets_inverse_A_4(J);   
res=0;   
 
    for (i=0;i<4;i++){
      for(j=0;j<4;j++) {
	ad_vustart[5*(gI+i)] = ad_vustart[5*(gI+i)] - J[i*4+j]*f[j];
      }
      res= res+f[i]*f[i]; 
  
      }*/
     
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



    /* dd[0]=J[5]* J[12]* J[2]* J[11] - J[5]* J[8]* J[2]* J[15] + J[9]* J[4]* J[2]* J[15] + J[10]* J[13]* J[4]* J[3]

     + J[8]* J[1]* J[6]* J[15] + J[8]* J[2]* J[13]* J[7] + J[10]* J[12]* J[1]* J[7] + J[9]* J[6]* J[12]* J[3]

     - J[13]* J[6] *J[8]* J[3] - J[10]* J[4] *J[1] *J[15] - J[14]* J[9]* J[4]* J[3] - J[12]* J[1] *J[6]* J[11]

     - J[12]* J[2]* J[9]* J[7] + J[14]* J[4]* J[1]* J[11] - J[14]* J[8]* J[1]* J[7] - J[13]* J[4]* J[2]* J[11]

      + J[5]* J[14]* J[8]* J[3] - J[5]* J[10]* J[12]* J[3] + J[0]* J[5]* J[10]* J[15] - J[0] *J[5] *J[14]* J[11]

      - J[0]* J[9]* J[6]* J[15] - J[0]* J[10]* J[13]* J[7] + J[0]* J[14]* J[9]* J[7] + J[0]* J[13]* J[6]* J[11];
     


    dd[4] =( J[9] *J[4] *J[2] *f[3] + J[8] *J[1] *J[6] *f[3] + J[10] *J[13] *J[4] *f[0] + J[10] *J[12] *J[1] *f[1] 
          - J[10] *J[4] *J[1] *f[3] + J[9] *J[6] *J[12] *f[0] + J[8] *J[2] *J[13] *f[1] - J[13] *J[6] *J[8] *f[0] 
          - J[12] *J[1] *J[6] *f[2] - J[12] *J[2] *J[9] *f[1] + J[14] *J[4] *J[1] *f[2] - J[14] *J[8] *J[1] *f[1]
          - J[14] *J[9] *J[4] *f[0] - J[13] *J[4] *J[2] *f[2] - J[5] *J[8] *J[2] *f[3] + J[5] *J[14] *J[8] *f[0]
          - J[5] *J[10] *J[12] *f[0] + J[5] *J[12] *J[2] *f[2] + J[0] *J[5] *J[10] *f[3] - J[0] *J[5] *J[14] *f[2]
	     - J[0] *J[10] *J[13] *f[1] + J[0]*J[14] *J[9] *f[1] + J[0] *J[13] *J[6] *f[2] - J[0] *J[9] *J[6] *f[3])/dd[0];

  

 dd[3] = -
    (J[0] *J[5] *J[11] *f[3] - J[0] *J[5] *f[2] *J[15] - J[0] *J[11] *J[13] *f[1] + J[0] *J[9] *f[1] *J[15] - J[0] *J[9] *J[7] *f[3]

     + J[0] *f[2] *J[13] *J[7] + J[5] *f[2] *J[12] *J[3] - J[5] *J[8] *J[3] *f[3] - J[5] *J[11] *J[12] *f[0]

     + J[5] *J[8] *f[0] *J[15] - J[4] *J[1] *J[11] *f[3] + J[8] *J[1]*J[7] *f[3] + J[11] *J[12] *J[1] *f[1]

     + J[9] *J[7] *J[12] *f[0] - f[2] *J[12] *J[1] *J[7] + J[8] *J[3] *J[13] *f[1] + J[9] *J[4] *J[3] *f[3]

     + J[11] *J[13] *J[4] *f[0] - J[9] *f[1] *J[12] *J[3] + J[4] *J[1] *f[2] *J[15] - f[2] *J[13] *J[4] *J[3]

     - J[8] *f[0] *J[13] *J[7] - J[9] *J[4] *f[0] *J[15] - J[8] *J[1] *f[1] *J[15])/dd[0];
 
dd[2] = (-J[0] *J[6] *f[2] *J[15]

     + J[0] *J[6] *J[11] *f[3] - J[0] *J[7] *J[10] *f[3] + J[0] *J[7] *J[14] *f[2] + J[0] *f[1] *J[10] *J[15]

     - J[0] *f[1] *J[14] *J[11] - J[4] *J[2] *J[11] *f[3] + J[4] *J[2] *f[2] *J[15] + J[6] *f[2] *J[12] *J[3]

     - J[6] *J[11] *J[12] *f[0] - J[6] *J[8] *J[3] *f[3] + J[4] *J[3] *J[10] *f[3] - J[4] *J[3] *J[14] *f[2]

     + J[6] *J[8] *f[0] *J[15] + J[7] *J[8] *J[2] *f[3] - J[4] *f[0] *J[10] *J[15] + J[4] *f[0] *J[14] *J[11]

     - f[1] *J[8] *J[2] *J[15] - J[7] *J[14] *J[8] *f[0] + J[7] *J[10] *J[12] *f[0] - J[7] *J[12] *J[2] *f[2]

     + f[1] *J[14] *J[8] *J[3] + 
	       f[1] *J[12] *J[2] *J[11] - f[1] *J[10] *J[12] *J[3])/dd[0];
 dd[1] = - (

    -f[0] *J[5] *J[10] *J[15] + f[0] *J[5] *J[14] *J[11] - f[0] *J[14] *J[9] *J[7] + f[0] *J[9] *J[6] *J[15]

     - f[0] *J[13] *J[6] *J[11] + f[0] *J[10] *J[13] *J[7] - J[1] *J[6] *f[2] *J[15] + J[1] *J[6] *J[11] *f[3]

     - J[1] *J[7] *J[10] *f[3] + J[1] *J[7] *J[14] *f[2] + J[1] *f[1] *J[10] *J[15] - J[1] *f[1] *J[14] *J[11]

     - J[2] *J[5] *J[11] *f[3] + J[2] *J[5] *f[2] *J[15] + J[2] *J[11] *J[13] *f[1] - J[2] *J[9] *f[1] *J[15]

     + J[2] *J[9] *J[7] *f[3] - J[2] *f[2] *J[13] *J[7] + J[3] *J[5] *J[10] *f[3] - J[3] *J[5] *J[14] *f[2]

    - J[3] *J[10] *J[13] *f[1] + J[3] *J[14] *J[9] *f[1] + J[3] *J[13] *J[6] *f[2] - J[3] *J[9] *J[6] *f[3])/dd[0];

    */


    res=0;   
 
    for (i=0;i<4;i++){
      ad_vustart[5*(gI+i)] = ad_vustart[5*(gI+i)] - dd[i+1];
    
      res= res+f[i]*f[i]; 
      //         ierr = PetscPrintf(PETSC_COMM_WORLD,"gI=%d,ad_f[%d]=%g,u[%d]=%g\n",gI,i,ad_f[5*i],i,ad_vustart[5*(gI+i)]);CHKERRQ(ierr);
 
      } 
    res = sqrt(res);
  
  } while (--cnt > 0 && res > 1.e-14);
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"use gI6 gI=%d\n",gI);CHKERRQ(ierr);
  ad_vustart[1+5*gI   ] = 0.0;
  ad_vustart[2+5*gI+5 ] = 0.0;
  ad_vustart[3+5*gI+10] = 0.0;
  ad_vustart[4+5*gI+15] = 0.0;
  
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"finish one newton\n");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*
        Nonlinear relax on all the equations with an initial guess in x
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFMatRelax_DAAD"
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

/*
        Nonlinear relax on all the equations with an initial guess in xx
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRelax_DAAD4"
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
    // ierr = PetscPrintf(PETSC_COMM_WORLD,"gtdof=%d residual=%d\n",gtdof, info.dof*info.xm*info.ym*info.zm - 1);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      ad_vustart[5*j  ]   = avu[j];
      // ierr = PetscPrintf(PETSC_COMM_WORLD,"initial[%d]=%g\n",j,ad_vustart[5*j]);
      ad_vustart[5*j+1]   = 0.0;
      ad_vustart[5*j+2]   = 0.0;
      ad_vustart[5*j+3]   = 0.0;
      ad_vustart[5*j+4]   = 0.0;
    }
    
    ierr = VecRestoreArray(localxx,&avu);CHKERRQ(ierr);

    // ierr = PetscPrintf(PETSC_COMM_WORLD,"info.zs=%d,info.zs+info.zm=%d\n",info.zs,info.zm);
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"info.ys=%d,info.zs+info.ym=%d\n",info.ys,info.ym);
    // ierr = PetscPrintf(PETSC_COMM_WORLD,"info.xs=%d,info.zs+info.xm=%d\n",info.xs,info.xm);
 if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      nI = 0;
      for (stencil.k = info.zs; stencil.k<info.zs+info.zm; stencil.k++) {
        for (stencil.j = info.ys; stencil.j<info.ys+info.ym; stencil.j++) {
          for (stencil.i = info.xs; stencil.i<info.xs+info.xm; stencil.i++) {
	    CHKMEMQ;
              gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;  
	      // ierr = PetscPrintf(PETSC_COMM_WORLD,"gI=%d, stencil.k=%d,stencil.j=%d,stencil.i=%d,nI=%d,residual0=%g,residual1=%g,residual2=%g,residual3=%g\n",gI, stencil.k,stencil.j,stencil.i,nI,residual[nI],residual[nI+1],residual[nI+2],residual[nI+3]);CHKERRQ(ierr);
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
	      //  ierr = PetscPrintf(PETSC_COMM_WORLD,"gI=%d, stencil.k=%d,stencil.j=%d,stencil.i=%d,nI=%d,residual0=%g,residual1=%g,residual2=%g,residual3=%g\n",gI, stencil.k,stencil.j,stencil.i,nI,residual[nI],residual[nI+1],residual[nI+2],residual[nI+3]);CHKERRQ(ierr);    
	          ierr = NLFNewton_DAAD4(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual+nI);CHKERRQ(ierr);
		  //ierr = NLFNewton_DAAD4(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual[nI-4],residual[nI-3],residual[nI-1],residual[nI]);CHKERRQ(ierr);
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

/*
        Nonlinear relax on all the equations with an initial guess in xx
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFMatRelax_DAADb"
PetscErrorCode PETSCMAT_DLLEXPORT NLFRelax_DAADb(NLF A,MatSORType flag,int its,Vec xx)
{
  PetscErrorCode ierr;
  PetscInt       j,gtdof,nI,gI, bs = A->da->w;
  PetscScalar    *avu,*av,*ad_vustart,*residual;
  Vec            localxx;
  DALocalInfo    info;
  MatStencil     stencil;
  void*          *ad_vu;
  PetscErrorCode (*NLFNewton_DAADb)(NLF,DALocalInfo*,MatStencil*,void*,PetscScalar*,int,int,PetscScalar);
  PetscErrorCode (*DAGetAdicMFArrayb)(DA,PetscTruth,void**,void**,PetscInt*);
  PetscErrorCode (*DARestoreAdicMFArrayb)(DA,PetscTruth,void**,void**,PetscInt*);

  PetscFunctionBegin;
  if (its <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D positive",its);

  ierr = DAGetLocalVector(A->da,&localxx);CHKERRQ(ierr);
  /* get space for derivative object.  */
  ierr = (*DAGetAdicMFArrayb)(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = VecGetArray(A->residual,&residual);CHKERRQ(ierr);


  /* tell ADIC we will be computing one dimensional Jacobians */
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
      ad_vustart[2*j]   = avu[j];
      ad_vustart[2*j+1] = 0.0;
    }
    ierr = VecRestoreArray(localxx,&avu);CHKERRQ(ierr);

    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      nI = 0;
      for (stencil.k = info.zs; stencil.k<info.zs+info.zm; stencil.k++) {
        for (stencil.j = info.ys; stencil.j<info.ys+info.ym; stencil.j++) {
          for (stencil.i = info.xs; stencil.i<info.xs+info.xm; stencil.i++) {
            gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
            ierr = (*NLFNewton_DAADb)(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual[nI]);CHKERRQ(ierr);
            nI++;
          }
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      nI = info.dof*info.xm*info.ym*info.zm - 1;
      for (stencil.k = info.zs+info.zm-1; stencil.k>=info.zs; stencil.k--) {
        for (stencil.j = info.ys+info.ym-1; stencil.j>=info.ys; stencil.j--) {
          for (stencil.i = info.xs+info.xm-1; stencil.i>=info.xs; stencil.i--) {
            gI   = (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
            ierr = (*NLFNewton_DAADb)(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual[nI]);CHKERRQ(ierr);
            nI--;
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
  ierr = (*DARestoreAdicMFArrayb)(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
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
  A->da = da;
  ierr = PetscObjectReference((PetscObject)da);CHKERRQ(ierr);
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




/* $Id: petscfunc.h,v 1.4 2001/09/07 20:13:00 bsmith Exp $ */
#ifndef PETSCFUNC_H
#define PETSCFUNC_H

#include "petscksp.h"

/*..KSP monitoring routines..*/ 
extern int KSPMonitorWriteConvHist(KSP ksp,int n,double rnorm,void* ctx);
extern int KSPMonitorAmg(KSP ksp,int n,double rnorm,void* ctx);
extern int KSPMonitorWriteResVecs(KSP ksp,int n,double rnorm,void* ctx); 

/*..KSP convergence criteria routines..*/ 
typedef struct{ 
    double BNRM2;
    int    NUMNODES; 
}CONVHIST; 

extern int ConvhistCtxCreate(CONVHIST **convhist);
extern int ConvhistCtxDestroy(CONVHIST *convhist);
extern int MyConvTest(KSP ksp,int n, double rnorm, KSPConvergedReason *reason, 
                      void* ctx); 

/*..Functions defined for block preconditioners..*/ 
extern int ReorderSubmatrices(PC pc,int nsub,IS *row,IS *col,
                              Mat *submat,void *dummy);
extern int PrintSubMatrices(PC pc,int nsub,IS *row,IS *col,
                            Mat *submat,void *dummy);
extern int ViewSubMatrices(PC pc,int nsub,IS *row,IS *col,
                           Mat *submat,void *dummy);
extern int KSPMonitorWriteConvHistOnFem(KSP ksp,int n,double rnorm,void* ctx);

/*..Viewing and printing matrices and vectors..*/
extern int MyMatView(Mat mat,void *dummy);
extern int PrintMatrix(Mat mat, char* path, char* base);
extern int PrintVector(Vec vec, char* path, char* base);

/*..From the MatCreateFcts collection..*/ 
extern int MatMatMult(Mat Fact1, Mat Fact2, Mat* Prod); 
extern int MatSubstract(Mat Term1, Mat Term2, Mat* Diff); 

/*..Structure used in the interface to RAMG..*/
typedef struct{
    PetscTruth        arraysset; /* indicates the arrays have already been allocated
                                    they will be deleted in the next setup call */
    double            *A; 
    int               *IA; 
    int               *JA;
    double            *U_APPROX; 
    double            *RHS;
    int               *IG;    
    struct RAMG_PARAM *PARAM;    
} RamgShellPC; 

/*..interface to RAMG..*/ 
extern int RamgShellPCCreate(RamgShellPC **shell); 
extern int RamgShellPCSetUp(RamgShellPC *shell, Mat pmat);
extern int RamgShellPCApply(void *ctx, Vec r, Vec z); 
extern int RamgShellPCDestroy(RamgShellPC *shell); 
extern int RamgGetParam(Mat A,struct RAMG_PARAM *ramg_param);

/*..Structure used in the interface to SAMG..*/ 
typedef struct{
    double *A; 
    int    *IA; 
    int    *JA; 
    struct SAMG_PARAM *PARAM; 
    int    LEVELS;           /* Number of levels created */   
} SamgShellPC; 

/*..Interface to SAMG..*/ 
extern int SamgShellPCCreate(SamgShellPC **shell); 
extern int SamgShellPCSetUp(SamgShellPC *shell, Mat pmat);
extern int SamgShellPCApply(void *ctx, Vec r, Vec z); 
extern int SamgShellPCDestroy(SamgShellPC *shell); 
extern int SamgGetParam(struct SAMG_PARAM *samg_param);

/*..Multigrid structure for PETSc..*/ 

/*....Maximum number of levels to be used in SAMG....*/ 
#define MAX_LEVELS 25 

typedef struct{
  /*..Implementation notes
    1 - The menber A is not stored on level 1 (the finest level in SAMG 
        ordering) to avoid unnecessary memory useage. 
  */ 
  KSP ksp_pre;  
  KSP ksp_post;
  Mat  A, B, C;
  Mat  Interp; 
  Vec  x, b, upd_b, r, y, b_y, r_y; 
  int  size; /*..Number of variables on level..*/ 
  /*  int  debug; */
} GridCtx; 

/*..Level 2 routine to get coarser level matrices..*/ 
extern int SamgGetCoarseMat(int level, int ia_shift, int ja_shift, 
                            Mat* coarsemat, void* ctx);
/*..Level 2 routine to get interpolation operators..*/ 
extern int SamgGetInterpolation(int level, int iw_shift, int jw_shift,
                                Mat* interpolation, void* ctx) ;

/*..Parse SAMG hierarchy to PETSc..*/ 
extern int SamgGetGrid(int levels, int numnodes, int numnonzero, 
                       GridCtx* grid, void* ctx); 
/*..Check parsing..*/ 
extern int SamgCheckGalerkin(int levels, Mat A, GridCtx* grid, 
                      void* ctx);

#endif//PETSCFUNC_H


#ifndef PETSCFUNC_H
#define PETSCFUNC_H

#include "petscksp.h"

/*..KSP monitoring routines..*/ 
EXTERN PetscErrorCode KSPMonitorWriteConvHist(KSP ksp,int n,double rnorm,void* ctx);
EXTERN PetscErrorCode KSPMonitorAmg(KSP ksp,int n,double rnorm,void* ctx);
EXTERN PetscErrorCode KSPMonitorWriteResVecs(KSP ksp,int n,double rnorm,void* ctx); 

/*..KSP convergence criteria routines..*/ 
typedef struct{ 
    double BNRM2;
    int    NUMNODES; 
}CONVHIST; 

EXTERN PetscErrorCode ConvhistCtxCreate(CONVHIST **convhist);
EXTERN PetscErrorCode ConvhistCtxDestroy(CONVHIST *convhist);
EXTERN PetscErrorCode MyConvTest(KSP ksp,int n, double rnorm, KSPConvergedReason *reason, 
                      void* ctx); 

/*..Functions defined for block preconditioners..*/ 
EXTERN PetscErrorCode ReorderSubmatrices(PC pc,int nsub,IS *row,IS *col,
                              Mat *submat,void *dummy);
EXTERN PetscErrorCode PrintSubMatrices(PC pc,int nsub,IS *row,IS *col,
                            Mat *submat,void *dummy);
EXTERN PetscErrorCode ViewSubMatrices(PC pc,int nsub,IS *row,IS *col,
                           Mat *submat,void *dummy);
EXTERN PetscErrorCode KSPMonitorWriteConvHistOnFem(KSP ksp,int n,double rnorm,void* ctx);

/*..Viewing and printing matrices and vectors..*/
EXTERN PetscErrorCode MyMatView(Mat mat,void *dummy);
EXTERN PetscErrorCode PrintMatrix(Mat mat, char* path, char* base);
EXTERN PetscErrorCode PrintVector(Vec vec, char* path, char* base);

/*..From the MatCreateFcts collection..*/ 
EXTERN PetscErrorCode MatSubstract(Mat Term1, Mat Term2, Mat* Diff); 

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
EXTERN PetscErrorCode RamgShellPCCreate(RamgShellPC **shell); 
EXTERN PetscErrorCode RamgShellPCSetUp(RamgShellPC *shell, Mat pmat);
EXTERN PetscErrorCode RamgShellPCApply(void *ctx, Vec r, Vec z); 
EXTERN PetscErrorCode RamgShellPCDestroy(RamgShellPC *shell); 
EXTERN PetscErrorCode RamgGetParam(Mat A,struct RAMG_PARAM *ramg_param);

/*..Structure used in the interface to SAMG..*/ 
typedef struct{
    double *A; 
    int    *IA; 
    int    *JA; 
    struct SAMG_PARAM *PARAM; 
    int    LEVELS;           /* Number of levels created */   
} SamgShellPC; 

/*..Interface to SAMG..*/ 
EXTERN PetscErrorCode SamgShellPCCreate(SamgShellPC **shell); 
EXTERN PetscErrorCode SamgShellPCSetUp(SamgShellPC *shell, Mat pmat);
EXTERN PetscErrorCode SamgShellPCApply(void *ctx, Vec r, Vec z); 
EXTERN PetscErrorCode SamgShellPCDestroy(SamgShellPC *shell); 
EXTERN PetscErrorCode SamgGetParam(struct SAMG_PARAM *samg_param);

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
EXTERN PetscErrorCode SamgGetCoarseMat(int level, int ia_shift, int ja_shift, 
                            Mat* coarsemat, void* ctx);
/*..Level 2 routine to get interpolation operators..*/ 
EXTERN PetscErrorCode SamgGetInterpolation(int level, int iw_shift, int jw_shift,
                                Mat* interpolation, void* ctx) ;

/*..Parse SAMG hierarchy to PETSc..*/ 
EXTERN PetscErrorCode SamgGetGrid(int levels, int numnodes, int numnonzero, 
                       GridCtx* grid, void* ctx); 
/*..Check parsing..*/ 
EXTERN PetscErrorCode SamgCheckGalerkin(int levels, Mat A, GridCtx* grid, 
                      void* ctx);

#endif//PETSCFUNC_H

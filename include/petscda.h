/* $Id: petscda.h,v 1.77 2001/09/11 16:34:35 bsmith Exp $ */

/*
      Regular array object, for easy parallelism of simple grid 
   problems on regular distributed arrays.
*/
#if !defined(__PETSCDA_H)
#define __PETSCDA_H
#include "petscvec.h"
#include "petscao.h"

/*S
     DA - Abstract PETSc object that manages distributed field data for a single structured grid

   Level: beginner

  Concepts: distributed array

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), VecScatter
S*/
typedef struct _p_DA* DA;

/*E
    DAStencilType - Determines if the stencil extends only along the coordinate directions, or also
      to the northest, northwest etc

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA
E*/
typedef enum { DA_STENCIL_STAR,DA_STENCIL_BOX } DAStencilType;

/*E
    DAPeriodicType - Is the domain periodic in one or more directions

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA
E*/
typedef enum { DA_NONPERIODIC,DA_XPERIODIC,DA_YPERIODIC,DA_XYPERIODIC,
               DA_XYZPERIODIC,DA_XZPERIODIC,DA_YZPERIODIC,DA_ZPERIODIC} 
               DAPeriodicType;

/*E
    DAInterpolationType - Defines the type of interpolation that will be returned by 
       DAGetInterpolation.

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA, DAGetInterpolation(), DASetInterpolationType()
E*/
typedef enum { DA_Q0, DA_Q1 } DAInterpolationType;

EXTERN int DASetInterpolationType(DA,DAInterpolationType);

#define DAXPeriodic(pt) ((pt)==DA_XPERIODIC||(pt)==DA_XYPERIODIC||(pt)==DA_XZPERIODIC||(pt)==DA_XYZPERIODIC)
#define DAYPeriodic(pt) ((pt)==DA_YPERIODIC||(pt)==DA_XYPERIODIC||(pt)==DA_YZPERIODIC||(pt)==DA_XYZPERIODIC)
#define DAZPeriodic(pt) ((pt)==DA_ZPERIODIC||(pt)==DA_XZPERIODIC||(pt)==DA_YZPERIODIC||(pt)==DA_XYZPERIODIC)

typedef enum { DA_X,DA_Y,DA_Z } DADirection;

/* Logging support */
extern int DA_COOKIE;
enum {DA_GlobalToLocal, DA_LocalToGlobal, DA_MAX_EVENTS};
extern int DAEvents[DA_MAX_EVENTS];
#define DALogEventBegin(e,o1,o2,o3,o4) PetscLogEventBegin(DAEvents[e],o1,o2,o3,o4)
#define DALogEventEnd(e,o1,o2,o3,o4)   PetscLogEventEnd(DAEvents[e],o1,o2,o3,o4)

EXTERN int   DACreate1d(MPI_Comm,DAPeriodicType,int,int,int,int*,DA *);
EXTERN int   DACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,int,int,int,int,int,int,int*,int*,DA *);
EXTERN int   DACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,int,int,int,int,int,int,int,int,int *,int *,int *,DA *);
EXTERN int   DADestroy(DA);
EXTERN int   DAView(DA,PetscViewer);

EXTERN int   DAPrintHelp(DA);

EXTERN int   DAGlobalToLocalBegin(DA,Vec,InsertMode,Vec);
EXTERN int   DAGlobalToLocalEnd(DA,Vec,InsertMode,Vec);
EXTERN int   DAGlobalToNaturalBegin(DA,Vec,InsertMode,Vec);
EXTERN int   DAGlobalToNaturalEnd(DA,Vec,InsertMode,Vec);
EXTERN int   DANaturalToGlobalBegin(DA,Vec,InsertMode,Vec);
EXTERN int   DANaturalToGlobalEnd(DA,Vec,InsertMode,Vec);
EXTERN int   DALocalToLocalBegin(DA,Vec,InsertMode,Vec);
EXTERN int   DALocalToLocalEnd(DA,Vec,InsertMode,Vec);
EXTERN int   DALocalToGlobal(DA,Vec,InsertMode,Vec);
EXTERN int   DALocalToGlobalBegin(DA,Vec,Vec);
EXTERN int   DALocalToGlobalEnd(DA,Vec,Vec);
EXTERN int   DAGetOwnershipRange(DA,int **,int **,int **);
EXTERN int   DACreateGlobalVector(DA,Vec *);
EXTERN int   DACreateNaturalVector(DA,Vec *);
EXTERN int   DACreateLocalVector(DA,Vec *);
EXTERN int   DAGetLocalVector(DA,Vec *);
EXTERN int   DARestoreLocalVector(DA,Vec *);
EXTERN int   DAGetGlobalVector(DA,Vec *);
EXTERN int   DARestoreGlobalVector(DA,Vec *);
EXTERN int   DALoad(PetscViewer,int,int,int,DA *);
EXTERN int   DAGetCorners(DA,int*,int*,int*,int*,int*,int*);
EXTERN int   DAGetGhostCorners(DA,int*,int*,int*,int*,int*,int*);
EXTERN int   DAGetInfo(DA,int*,int*,int*,int*,int*,int*,int*,int*,int*,DAPeriodicType*,DAStencilType*);
EXTERN int   DAGetProcessorSubset(DA,DADirection,int,MPI_Comm*);
EXTERN int   DARefine(DA,MPI_Comm,DA*);

EXTERN int   DAGlobalToNaturalAllCreate(DA,VecScatter*);
EXTERN int   DANaturalAllToGlobalCreate(DA,VecScatter*);

EXTERN int   DAGetGlobalIndices(DA,int*,int**);
EXTERN int   DAGetISLocalToGlobalMapping(DA,ISLocalToGlobalMapping*);
EXTERN int   DAGetISLocalToGlobalMappingBlck(DA,ISLocalToGlobalMapping*);

EXTERN int   DAGetScatter(DA,VecScatter*,VecScatter*,VecScatter*);

EXTERN int   DAGetAO(DA,AO*);
EXTERN int   DASetCoordinates(DA,Vec); 
EXTERN int   DAGetCoordinates(DA,Vec *);
EXTERN int   DASetUniformCoordinates(DA,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN int   DASetFieldName(DA,int,const char[]);
EXTERN int   DAGetFieldName(DA,int,char **);

EXTERN int   DAVecGetArray(DA,Vec,void **);
EXTERN int   DAVecRestoreArray(DA,Vec,void **);

EXTERN int   DASplitComm2d(MPI_Comm,int,int,int,MPI_Comm*);

EXTERN int   MatRegisterDAAD(void);
EXTERN int   MatCreateDAAD(DA,Mat*);

/*S
     DALocalInfo - C struct that contains information about a structured grid and a processors logical
              location in it.

   Level: beginner

  Concepts: distributed array

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DA, DAGetLocalInfo(), DAGetInfo()
S*/
typedef struct {
  int            dim,dof,sw;
  DAPeriodicType pt;
  DAStencilType  st;
  int            mx,my,mz;    /* global number of grid points in each direction */
  int            xs,ys,zs;    /* starting point of this processor, excluding ghosts */
  int            xm,ym,zm;    /* number of grid points on this processor, excluding ghosts */
  int            gxs,gys,gzs;    /* starting point of this processor including ghosts */
  int            gxm,gym,gzm;    /* number of grid points on this processor including ghosts */
  DA             da;
} DALocalInfo;

EXTERN int DAGetLocalInfo(DA,DALocalInfo*);
typedef int (*DALocalFunction1)(DALocalInfo*,void*,void*,void*);
EXTERN int DAFormFunction1(DA,Vec,Vec,void*);
EXTERN int DAFormFunctioni1(DA,int,Vec,PetscScalar*,void*);
EXTERN int DAComputeJacobian1WithAdic(DA,Vec,Mat,void*);
EXTERN int DAComputeJacobian1WithAdifor(DA,Vec,Mat,void*);
EXTERN int DAMultiplyByJacobian1WithAdic(DA,Vec,Vec,Vec,void*);
EXTERN int DAMultiplyByJacobian1WithAdifor(DA,Vec,Vec,Vec,void*);
EXTERN int DAMultiplyByJacobian1WithAD(DA,Vec,Vec,Vec,void*);
EXTERN int DAComputeJacobian1(DA,Vec,Mat,void*);
EXTERN int DAGetLocalFunction(DA,DALocalFunction1*);
EXTERN int DASetLocalFunction(DA,DALocalFunction1);
EXTERN int DASetLocalFunctioni(DA,int (*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
EXTERN int DASetLocalJacobian(DA,DALocalFunction1);
EXTERN int DASetLocalAdicFunction_Private(DA,DALocalFunction1);
#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#  define DASetLocalAdicFunction(a,d) DASetLocalAdicFunction_Private(a,(DALocalFunction1)d)
#else
#  define DASetLocalAdicFunction(a,d) DASetLocalAdicFunction_Private(a,0)
#endif
EXTERN int DASetLocalAdicMFFunction_Private(DA,DALocalFunction1);
#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#  define DASetLocalAdicMFFunction(a,d) DASetLocalAdicMFFunction_Private(a,(DALocalFunction1)d)
#else
#  define DASetLocalAdicMFFunction(a,d) DASetLocalAdicMFFunction_Private(a,0)
#endif
EXTERN int DASetLocalAdicFunctioni_Private(DA,int (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#  define DASetLocalAdicFunctioni(a,d) DASetLocalAdicFunctioni_Private(a,(int (*)(DALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DASetLocalAdicFunctioni(a,d) DASetLocalAdicFunctioni_Private(a,0)
#endif
EXTERN int DASetLocalAdicMFFunctioni_Private(DA,int (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#  define DASetLocalAdicMFFunctioni(a,d) DASetLocalAdicMFFunctioni_Private(a,(int (*)(DALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DASetLocalAdicMFFunctioni(a,d) DASetLocalAdicMFFunctioni_Private(a,0)
#endif
EXTERN int DAFormFunctioniTest1(DA,void*);


#include "petscmat.h"
EXTERN int DAGetColoring(DA,ISColoringType,ISColoring *);
EXTERN int DAGetMatrix(DA,MatType,Mat *);
EXTERN int DASetGetMatrix(DA,int (*)(DA,MatType,Mat *));
EXTERN int DAGetInterpolation(DA,DA,Mat*,Vec*);

EXTERN int DAGetAdicArray(DA,PetscTruth,void**,void**,int*);
EXTERN int DARestoreAdicArray(DA,PetscTruth,void**,void**,int*);
EXTERN int DAGetAdicMFArray(DA,PetscTruth,void**,void**,int*);
EXTERN int DARestoreAdicMFArray(DA,PetscTruth,void**,void**,int*);
EXTERN int DAGetArray(DA,PetscTruth,void**);
EXTERN int DARestoreArray(DA,PetscTruth,void**);
EXTERN int ad_DAGetArray(DA,PetscTruth,void**);
EXTERN int ad_DARestoreArray(DA,PetscTruth,void**);
EXTERN int admf_DAGetArray(DA,PetscTruth,void**);
EXTERN int admf_DARestoreArray(DA,PetscTruth,void**);

#include "petscpf.h"
EXTERN int DACreatePF(DA,PF*);

/*S
     VecPack - Abstract PETSc object that manages treating several distinct vectors as if they
        were one.   The VecPack routines allow one to manage a nonlinear solver that works on a
        vector that consists of several distinct parts. This is mostly used for LNKS solvers, 
        that is design optimization problems that are written as a nonlinear system

   Level: beginner

  Concepts: multi-component, LNKS solvers

.seealso:  VecPackCreate(), VecPackDestroy()
S*/
typedef struct _p_VecPack *VecPack;

EXTERN int VecPackCreate(MPI_Comm,VecPack*);
EXTERN int VecPackDestroy(VecPack);
EXTERN int VecPackAddArray(VecPack,int);
EXTERN int VecPackAddDA(VecPack,DA);
EXTERN int VecPackAddVecScatter(VecPack,VecScatter);
EXTERN int VecPackScatter(VecPack,Vec,...);
EXTERN int VecPackGather(VecPack,Vec,...);
EXTERN int VecPackGetAccess(VecPack,Vec,...);
EXTERN int VecPackRestoreAccess(VecPack,Vec,...);
EXTERN int VecPackGetLocalVectors(VecPack,...);
EXTERN int VecPackGetEntries(VecPack,...);
EXTERN int VecPackRestoreLocalVectors(VecPack,...);
EXTERN int VecPackCreateGlobalVector(VecPack,Vec*);
EXTERN int VecPackGetGlobalIndices(VecPack,...);
EXTERN int VecPackRefine(VecPack,MPI_Comm,VecPack*);
EXTERN int VecPackGetInterpolation(VecPack,VecPack,Mat*,Vec*);

#include "petscsnes.h"

/*S
     DM - Abstract PETSc object that manages an abstract grid object
          
   Level: intermediate

  Concepts: grids, grid refinement

   Notes: The DA object and the VecPack object are examples of DMs

.seealso:  VecPackCreate(), DA, VecPack
S*/
typedef struct _p_DM* DM;

EXTERN int DMView(DM,PetscViewer);
EXTERN int DMDestroy(DM);
EXTERN int DMCreateGlobalVector(DM,Vec*);
EXTERN int DMGetColoring(DM,ISColoringType,ISColoring*);
EXTERN int DMGetMatrix(DM,MatType,Mat*);
EXTERN int DMGetInterpolation(DM,DM,Mat*,Vec*);
EXTERN int DMRefine(DM,MPI_Comm,DM*);
EXTERN int DMGetInterpolationScale(DM,DM,Mat,Vec*);

/*S
     DMMG -  Data structure to easily manage multi-level non-linear solvers on grids managed by DM
          
   Level: intermediate

  Concepts: multigrid, Newton-multigrid

.seealso:  VecPackCreate(), DA, VecPack, DM, DMMGCreate()
S*/
typedef struct _p_DMMG *DMMG;
struct _p_DMMG {
  DM         dm;                   /* grid information for this level */
  Vec        x,b,r;                /* global vectors used in multigrid preconditioner for this level*/
  Mat        J;                    /* matrix on this level */
  Mat        R;                    /* restriction to next coarser level (not defined on level 0) */
  int        nlevels;              /* number of levels above this one (total number of levels on level 0)*/
  MPI_Comm   comm;
  int        (*solve)(DMMG*,int);
  void       *user;         
  PetscTruth galerkin;             /* for A_c = R*A*R^T */

  /* SLES only */
  SLES       sles;             
  int        (*rhs)(DMMG,Vec);
  PetscTruth matricesset;              /* User had called DMMGSetSLES() and the matrices have been computed */

  /* SNES only */
  Mat           B;
  Vec           Rscale;                /* scaling to restriction before computing Jacobian */
  int           (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);  
  int           (*computefunction)(SNES,Vec,Vec,void*);  

  PetscTruth    updatejacobian;        /* compute new Jacobian when DMMGComputeJacobian_Multigrid() is called */
  int           updatejacobianperiod;  /* how often, inside a SNES, the Jacobian is recomputed */

  MatFDColoring    fdcoloring;            /* only used with FD coloring for Jacobian */  
  SNES             snes;                  
  int              (*initialguess)(SNES,Vec,void*);
  Vec              w,work1,work2;         /* global vectors */
  Vec              lwork1;
};

EXTERN int DMMGCreate(MPI_Comm,int,void*,DMMG**);
EXTERN int DMMGDestroy(DMMG*);
EXTERN int DMMGSetUp(DMMG*);
EXTERN int DMMGSetSLES(DMMG*,int (*)(DMMG,Vec),int (*)(DMMG,Mat));
EXTERN int DMMGSetSNES(DMMG*,int (*)(SNES,Vec,Vec,void*),int (*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*));
EXTERN int DMMGSetInitialGuess(DMMG*,int (*)(SNES,Vec,void*));
EXTERN int DMMGView(DMMG*,PetscViewer);
EXTERN int DMMGSolve(DMMG*);
EXTERN int DMMGSetUseMatrixFree(DMMG*);
EXTERN int DMMGSetDM(DMMG*,DM);
EXTERN int DMMGSetUpLevel(DMMG*,SLES,int);
EXTERN int DMMGSetUseGalerkinCoarse(DMMG*);

EXTERN int DMMGSetSNESLocal_Private(DMMG*,DALocalFunction1,DALocalFunction1,DALocalFunction1,DALocalFunction1);
#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#  define DMMGSetSNESLocal(dmmg,function,jacobian,ad_function,admf_function) \
  DMMGSetSNESLocal_Private(dmmg,(DALocalFunction1)function,(DALocalFunction1)jacobian,(DALocalFunction1)(ad_function),(DALocalFunction1)(admf_function))
#else
#  define DMMGSetSNESLocal(dmmg,function,jacobian,ad_function,admf_function) DMMGSetSNESLocal_Private(dmmg,(DALocalFunction1)function,(DALocalFunction1)jacobian,(DALocalFunction1)0,(DALocalFunction1)0)
#endif

EXTERN int DMMGSetSNESLocali_Private(DMMG*,int (*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*),int (*)(DALocalInfo*,MatStencil*,void*,void*,void*),int (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#  define DMMGSetSNESLocali(dmmg,function,ad_function,admf_function) DMMGSetSNESLocali_Private(dmmg,(int (*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*))function,(int (*)(DALocalInfo*,MatStencil*,void*,void*,void*))(ad_function),(int (*)(DALocalInfo*,MatStencil*,void*,void*,void*))(admf_function))
#else
#  define DMMGSetSNESLocali(dmmg,function,ad_function,admf_function) DMMGSetSNESLocali_Private(dmmg,(int (*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*))function,0,0)
#endif

#define DMMGGetb(ctx)              (ctx)[(ctx)[0]->nlevels-1]->b
#define DMMGGetr(ctx)              (ctx)[(ctx)[0]->nlevels-1]->r

/*MC
   DMMGGetx - Returns the solution vector from a DMMG solve on the finest grid

   Synopsis:
   Vec DMMGGetx(DMMG *dmmg)

   Not Collective, but resulting vector is parallel

   Input Parameters:
.   dmmg - DMMG solve context

   Level: intermediate

   Fortran Usage:
.     DMMGGetx(DMMG dmmg,Vec x,int ierr)

.seealso: DMMGCreate(), DMMGSetSNES(), DMMGSetSLES(), DMMGSetSNESLocal()

M*/
#define DMMGGetx(ctx)              (ctx)[(ctx)[0]->nlevels-1]->x

#define DMMGGetJ(ctx)              (ctx)[(ctx)[0]->nlevels-1]->J
#define DMMGGetB(ctx)              (ctx)[(ctx)[0]->nlevels-1]->B
#define DMMGGetFine(ctx)           (ctx)[(ctx)[0]->nlevels-1]
#define DMMGGetSLES(ctx)           (ctx)[(ctx)[0]->nlevels-1]->sles
#define DMMGGetSNES(ctx)           (ctx)[(ctx)[0]->nlevels-1]->snes
#define DMMGGetDA(ctx)             (DA)((ctx)[(ctx)[0]->nlevels-1]->dm)
#define DMMGGetVecPack(ctx)        (VecPack)((ctx)[(ctx)[0]->nlevels-1]->dm)
#define DMMGGetUser(ctx,level)     ((ctx)[levels]->user)
#define DMMGSetUser(ctx,level,usr) 0,(ctx)[levels]->user = usr
#define DMMGGetLevels(ctx)         (ctx)[0]->nlevels

#endif



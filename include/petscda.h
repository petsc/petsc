/*
      Regular array object, for easy parallelism of simple grid 
   problems on regular distributed arrays.
*/
#if !defined(__PETSCDA_H)
#define __PETSCDA_H
#include "petscvec.h"
#include "petscao.h"
PETSC_EXTERN_CXX_BEGIN

/*S
     DA - Abstract PETSc object that manages distributed field data for a single structured grid

   Level: beginner

  Concepts: distributed array

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), VecScatter, DACreate()
S*/
typedef struct _p_DA* DA;

/*E
    DAStencilType - Determines if the stencil extends only along the coordinate directions, or also
      to the northeast, northwest etc

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA, DACreate()
E*/
typedef enum { DA_STENCIL_STAR,DA_STENCIL_BOX } DAStencilType;

/*M
     DA_STENCIL_STAR - "Star"-type stencil. In logical grid coordinates, only (i,j,k), (i+s,j,k), (i,j+s,k),
                       (i,j,k+s) are in the stencil  NOT, for example, (i+s,j+s,k)

     Level: beginner

.seealso: DA_STENCIL_BOX, DAStencilType
M*/

/*M
     DA_STENCIL_Box - "Box"-type stencil. In logical grid coordinates, any of (i,j,k), (i+s,j+r,k+t) may 
                      be in the stencil.

     Level: beginner

.seealso: DA_STENCIL_STAR, DAStencilType
M*/

/*E
    DAPeriodicType - Is the domain periodic in one or more directions

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA, DACreate()
E*/
typedef enum { DA_NONPERIODIC,DA_XPERIODIC,DA_YPERIODIC,DA_XYPERIODIC,
               DA_XYZPERIODIC,DA_XZPERIODIC,DA_YZPERIODIC,DA_ZPERIODIC} 
               DAPeriodicType;

/*E
    DAInterpolationType - Defines the type of interpolation that will be returned by 
       DAGetInterpolation.

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA, DAGetInterpolation(), DASetInterpolationType(), DACreate()
E*/
typedef enum { DA_Q0, DA_Q1 } DAInterpolationType;

EXTERN PetscErrorCode DASetInterpolationType(DA,DAInterpolationType);

/*E
    DAElementType - Defines the type of elements that will be returned by 
       DAGetElements.

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA, DAGetInterpolation(), DASetInterpolationType(), 
          DASetElementType(), DAGetElements(), DARestoreElements(), DACreate()
E*/
typedef enum { DA_ELEMENT_P1, DA_ELEMENT_Q1 } DAElementType;

EXTERN PetscErrorCode DASetElementType(DA,DAElementType);
EXTERN PetscErrorCode DAGetElements(DA,PetscInt *,const PetscInt*[]);
EXTERN PetscErrorCode DARestoreElements(DA,PetscInt *,const PetscInt*[]);


#define DAXPeriodic(pt) ((pt)==DA_XPERIODIC||(pt)==DA_XYPERIODIC||(pt)==DA_XZPERIODIC||(pt)==DA_XYZPERIODIC)
#define DAYPeriodic(pt) ((pt)==DA_YPERIODIC||(pt)==DA_XYPERIODIC||(pt)==DA_YZPERIODIC||(pt)==DA_XYZPERIODIC)
#define DAZPeriodic(pt) ((pt)==DA_ZPERIODIC||(pt)==DA_XZPERIODIC||(pt)==DA_YZPERIODIC||(pt)==DA_XYZPERIODIC)

typedef enum { DA_X,DA_Y,DA_Z } DADirection;

extern PetscCookie DA_COOKIE;
extern PetscEvent  DA_GlobalToLocal, DA_LocalToGlobal, DA_LocalADFunction;

EXTERN PetscErrorCode   DACreate1d(MPI_Comm,DAPeriodicType,PetscInt,PetscInt,PetscInt,PetscInt*,DA *);
EXTERN PetscErrorCode   DACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,DA *);
EXTERN PetscErrorCode   DACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*,DA*);
EXTERN PetscErrorCode   DACreate(MPI_Comm,PetscInt,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*,DA*);
EXTERN PetscErrorCode   DADestroy(DA);
EXTERN PetscErrorCode   DAView(DA,PetscViewer);
EXTERN PetscErrorCode   DAPrintHelp(DA);

EXTERN PetscErrorCode   DAGlobalToLocalBegin(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DAGlobalToLocalEnd(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DAGlobalToNaturalBegin(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DAGlobalToNaturalEnd(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DANaturalToGlobalBegin(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DANaturalToGlobalEnd(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DALocalToLocalBegin(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DALocalToLocalEnd(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DALocalToGlobal(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode   DALocalToGlobalBegin(DA,Vec,Vec);
EXTERN PetscErrorCode   DALocalToGlobalEnd(DA,Vec,Vec);
EXTERN PetscErrorCode   DAGetOwnershipRange(DA,PetscInt **,PetscInt **,PetscInt **);
EXTERN PetscErrorCode   DACreateGlobalVector(DA,Vec *);
EXTERN PetscErrorCode   DACreateNaturalVector(DA,Vec *);
EXTERN PetscErrorCode   DACreateLocalVector(DA,Vec *);
EXTERN PetscErrorCode   DAGetLocalVector(DA,Vec *);
EXTERN PetscErrorCode   DARestoreLocalVector(DA,Vec *);
EXTERN PetscErrorCode   DAGetGlobalVector(DA,Vec *);
EXTERN PetscErrorCode   DARestoreGlobalVector(DA,Vec *);
EXTERN PetscErrorCode   DALoad(PetscViewer,PetscInt,PetscInt,PetscInt,DA *);
EXTERN PetscErrorCode   DAGetCorners(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode   DAGetGhostCorners(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode   DAGetInfo(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,DAPeriodicType*,DAStencilType*);
EXTERN PetscErrorCode   DAGetProcessorSubset(DA,DADirection,PetscInt,MPI_Comm*);
EXTERN PetscErrorCode   DARefine(DA,MPI_Comm,DA*);

EXTERN PetscErrorCode   DAGlobalToNaturalAllCreate(DA,VecScatter*);
EXTERN PetscErrorCode   DANaturalAllToGlobalCreate(DA,VecScatter*);

EXTERN PetscErrorCode   DAGetGlobalIndices(DA,PetscInt*,PetscInt**);
EXTERN PetscErrorCode   DAGetISLocalToGlobalMapping(DA,ISLocalToGlobalMapping*);
EXTERN PetscErrorCode   DAGetISLocalToGlobalMappingBlck(DA,ISLocalToGlobalMapping*);

EXTERN PetscErrorCode   DAGetScatter(DA,VecScatter*,VecScatter*,VecScatter*);

EXTERN PetscErrorCode   DAGetAO(DA,AO*);
EXTERN PetscErrorCode   DASetCoordinates(DA,Vec); 
EXTERN PetscErrorCode   DAGetCoordinates(DA,Vec *);
EXTERN PetscErrorCode   DAGetGhostedCoordinates(DA,Vec *);
EXTERN PetscErrorCode   DAGetCoordinateDA(DA,DA *);
EXTERN PetscErrorCode   DASetUniformCoordinates(DA,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode   DASetFieldName(DA,PetscInt,const char[]);
EXTERN PetscErrorCode   DAGetFieldName(DA,PetscInt,char **);

EXTERN PetscErrorCode   DAVecGetArray(DA,Vec,void *);
EXTERN PetscErrorCode   DAVecRestoreArray(DA,Vec,void *);

EXTERN PetscErrorCode   DAVecGetArrayDOF(DA,Vec,void *);
EXTERN PetscErrorCode   DAVecRestoreArrayDOF(DA,Vec,void *);

EXTERN PetscErrorCode   DASplitComm2d(MPI_Comm,PetscInt,PetscInt,PetscInt,MPI_Comm*);

/*S
     SDA - This provides a simplified interface to the DA distributed
           array object in PETSc. This is intended for people who are
           NOT using PETSc vectors or objects but just want to distribute
           simple rectangular arrays amoung a number of procesors and have
           PETSc handle moving the ghost-values when needed.

          In certain applications this can serve as a replacement for 
          BlockComm (which is apparently being phased out?).


   Level: beginner

  Concepts: simplified distributed array

.seealso:  SDACreate1d(), SDACreate2d(), SDACreate3d(), SDADestroy(), DA, SDALocalToLocalBegin(),
           SDALocalToLocalEnd(), SDAGetCorners(), SDAGetGhostCorners()
S*/
typedef struct _SDA* SDA;

EXTERN PetscErrorCode   SDACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*,SDA*);
EXTERN PetscErrorCode   SDACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,SDA*);
EXTERN PetscErrorCode   SDACreate1d(MPI_Comm,DAPeriodicType,PetscInt,PetscInt,PetscInt,PetscInt*,SDA*);
EXTERN PetscErrorCode   SDADestroy(SDA);
EXTERN PetscErrorCode   SDALocalToLocalBegin(SDA,PetscScalar*,InsertMode,PetscScalar*);
EXTERN PetscErrorCode   SDALocalToLocalEnd(SDA,PetscScalar*,InsertMode,PetscScalar*);
EXTERN PetscErrorCode   SDAGetCorners(SDA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode   SDAGetGhostCorners(SDA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode   MatRegisterDAAD(void);
EXTERN PetscErrorCode   MatCreateDAAD(DA,Mat*);

/*S
     DALocalInfo - C struct that contains information about a structured grid and a processors logical
              location in it.

   Level: beginner

  Concepts: distributed array

  Developer note: Then entries in this struct are int instead of PetscInt so that the elements may
                  be extracted in Fortran as if from an integer array

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DA, DAGetLocalInfo(), DAGetInfo()
S*/
typedef struct {
  int            dim,dof,sw;
  DAPeriodicType pt;
  DAStencilType  st;
  int            mx,my,mz;    /* global number of grid points in each direction */
  int            xs,ys,zs;    /* starting pointd of this processor, excluding ghosts */
  int            xm,ym,zm;    /* number of grid points on this processor, excluding ghosts */
  int            gxs,gys,gzs;    /* starting point of this processor including ghosts */
  int            gxm,gym,gzm;    /* number of grid points on this processor including ghosts */
  DA             da;
} DALocalInfo;

/*MC
      DAForEachPointBegin2d - Starts a loop over the local part of a two dimensional DA

   Synopsis:
   void  DAForEachPointBegin2d(DALocalInfo *info,PetscInt i,PetscInt j);
   
   Level: intermediate

.seealso: DAForEachPointEnd2d(), DAVecGetArray()
M*/
#define DAForEachPointBegin2d(info,i,j) {\
  PetscInt _xints = info->xs,_xinte = info->xs+info->xm,_yints = info->ys,_yinte = info->ys+info->ym;\
  for (j=_yints; j<_yinte; j++) {\
    for (i=_xints; i<_xinte; i++) {\

/*MC
      DAForEachPointEnd2d - Ends a loop over the local part of a two dimensional DA

   Synopsis:
   void  DAForEachPointEnd2d;
   
   Level: intermediate

.seealso: DAForEachPointBegin2d(), DAVecGetArray()
M*/
#define DAForEachPointEnd2d }}}

/*MC
      DACoor2d - Structure for holding 2d (x and y) coordinates.

    Level: intermediate

    Sample Usage:
      DACoor2d **coors;
      Vec      vcoors;
      DA       cda;     

      DAGetCoordinates(da,&vcoors); 
      DAGetCoordinateDA(da,&cda);
      DAVecGetArray(dac,vcoors,&coors);
      DAGetCorners(dac,&mstart,&nstart,0,&m,&n,0)
      for (i=mstart; i<mstart+m; i++) {
        for (j=nstart; j<nstart+n; j++) {
          x = coors[j][i].x;
          y = coors[j][i].y;
          ......
        }
      }
      DAVecRestoreArray(dac,vcoors,&coors);

.seealso: DACoor3d, DAForEachPointBegin(), DAGetCoordinateDA(), DAGetCoordinates(), DAGetGhostCoordinates()
M*/
typedef struct {PetscScalar x,y;} DACoor2d;

/*MC
      DACoor3d - Structure for holding 3d (x, y and z) coordinates.

    Level: intermediate

    Sample Usage:
      DACoor3d **coors;
      Vec      vcoors;
      DA       cda;     

      DAGetCoordinates(da,&vcoors); 
      DAGetCoordinateDA(da,&cda);
      DAVecGetArray(dac,vcoors,&coors);
      DAGetCorners(dac,&mstart,&nstart,&pstart,&m,&n,&p)
      for (i=mstart; i<mstart+m; i++) {
        for (j=nstart; j<nstart+n; j++) {
          for (k=pstart; k<pstart+p; k++) {
            x = coors[k][j][i].x;
            y = coors[k][j][i].y;
            z = coors[k][j][i].z;
          ......
        }
      }
      DAVecRestoreArray(dac,vcoors,&coors);

.seealso: DACoor2d, DAForEachPointBegin(), DAGetCoordinateDA(), DAGetCoordinates(), DAGetGhostCoordinates()
M*/
typedef struct {PetscScalar x,y,z;} DACoor3d;
    
EXTERN PetscErrorCode DAGetLocalInfo(DA,DALocalInfo*);
typedef PetscErrorCode (*DALocalFunction1)(DALocalInfo*,void*,void*,void*);
EXTERN PetscErrorCode DAFormFunction1(DA,Vec,Vec,void*);
EXTERN PetscErrorCode DAFormFunctioni1(DA,PetscInt,Vec,PetscScalar*,void*);
EXTERN PetscErrorCode DAComputeJacobian1WithAdic(DA,Vec,Mat,void*);
EXTERN PetscErrorCode DAComputeJacobian1WithAdifor(DA,Vec,Mat,void*);
EXTERN PetscErrorCode DAMultiplyByJacobian1WithAdic(DA,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode DAMultiplyByJacobian1WithAdifor(DA,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode DAMultiplyByJacobian1WithAD(DA,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode DAComputeJacobian1(DA,Vec,Mat,void*);
EXTERN PetscErrorCode DAGetLocalFunction(DA,DALocalFunction1*);
EXTERN PetscErrorCode DASetLocalFunction(DA,DALocalFunction1);
EXTERN PetscErrorCode DASetLocalFunctioni(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
EXTERN PetscErrorCode DASetLocalJacobian(DA,DALocalFunction1);
EXTERN PetscErrorCode DASetLocalAdicFunction_Private(DA,DALocalFunction1);

/*MC
       DASetLocalAdicFunction - Caches in a DA a local function computed by ADIC/ADIFOR

   Collective on DA

   Synopsis:
   PetscErrorCode DASetLocalAdicFunction(DA da,DALocalFunction1 ad_lf)
   
   Input Parameter:
+  da - initial distributed array
-  ad_lf - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian()
M*/
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicFunction(a,d) DASetLocalAdicFunction_Private(a,(DALocalFunction1)d)
#else
#  define DASetLocalAdicFunction(a,d) DASetLocalAdicFunction_Private(a,0)
#endif

EXTERN PetscErrorCode DASetLocalAdicMFFunction_Private(DA,DALocalFunction1);
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicMFFunction(a,d) DASetLocalAdicMFFunction_Private(a,(DALocalFunction1)d)
#else
#  define DASetLocalAdicMFFunction(a,d) DASetLocalAdicMFFunction_Private(a,0)
#endif
EXTERN PetscErrorCode DASetLocalAdicFunctioni_Private(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicFunctioni(a,d) DASetLocalAdicFunctioni_Private(a,(PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DASetLocalAdicFunctioni(a,d) DASetLocalAdicFunctioni_Private(a,0)
#endif
EXTERN PetscErrorCode DASetLocalAdicMFFunctioni_Private(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicMFFunctioni(a,d) DASetLocalAdicMFFunctioni_Private(a,(PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DASetLocalAdicMFFunctioni(a,d) DASetLocalAdicMFFunctioni_Private(a,0)
#endif
EXTERN PetscErrorCode DAFormFunctioniTest1(DA,void*);

#include "petscmat.h"
EXTERN PetscErrorCode DAGetColoring(DA,ISColoringType,ISColoring *);
EXTERN PetscErrorCode DAGetMatrix(DA,const MatType,Mat *);
EXTERN PetscErrorCode DASetGetMatrix(DA,PetscErrorCode (*)(DA,const MatType,Mat *));
EXTERN PetscErrorCode DAGetInterpolation(DA,DA,Mat*,Vec*);
EXTERN PetscErrorCode DAGetInjection(DA,DA,VecScatter*);
EXTERN PetscErrorCode DASetBlockFills(DA,PetscInt*,PetscInt*);
EXTERN PetscErrorCode DASetRefinementFactor(DA,PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode DAGetRefinementFactor(DA,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode DAGetAdicArray(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode DARestoreAdicArray(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode DAGetAdicMFArray(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode DARestoreAdicMFArray(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode DAGetArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode DARestoreArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode ad_DAGetArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode ad_DARestoreArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode admf_DAGetArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode admf_DARestoreArray(DA,PetscTruth,void**);

#include "petscpf.h"
EXTERN PetscErrorCode DACreatePF(DA,PF*);

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

EXTERN PetscErrorCode VecPackCreate(MPI_Comm,VecPack*);
EXTERN PetscErrorCode VecPackDestroy(VecPack);
EXTERN PetscErrorCode VecPackAddArray(VecPack,PetscInt);
EXTERN PetscErrorCode VecPackAddDA(VecPack,DA);
EXTERN PetscErrorCode VecPackAddVecScatter(VecPack,VecScatter);
EXTERN PetscErrorCode VecPackScatter(VecPack,Vec,...);
EXTERN PetscErrorCode VecPackGather(VecPack,Vec,...);
EXTERN PetscErrorCode VecPackGetAccess(VecPack,Vec,...);
EXTERN PetscErrorCode VecPackRestoreAccess(VecPack,Vec,...);
EXTERN PetscErrorCode VecPackGetLocalVectors(VecPack,...);
EXTERN PetscErrorCode VecPackGetEntries(VecPack,...);
EXTERN PetscErrorCode VecPackRestoreLocalVectors(VecPack,...);
EXTERN PetscErrorCode VecPackCreateGlobalVector(VecPack,Vec*);
EXTERN PetscErrorCode VecPackGetGlobalIndices(VecPack,...);
EXTERN PetscErrorCode VecPackRefine(VecPack,MPI_Comm,VecPack*);
EXTERN PetscErrorCode VecPackGetInterpolation(VecPack,VecPack,Mat*,Vec*);

#include "petscsnes.h"

/*S
     DM - Abstract PETSc object that manages an abstract grid object
          
   Level: intermediate

  Concepts: grids, grid refinement

   Notes: The DA object and the VecPack object are examples of DMs

          Though the DA objects require the petscsnes.h include files the DM library is
    NOT dependent on the SNES or KSP library. In fact, the KSP and SNES libraries depend on
    DM. (This is not great design, but not trivial to fix).

.seealso:  VecPackCreate(), DA, VecPack
S*/
typedef struct _p_DM* DM;

EXTERN PetscErrorCode DMView(DM,PetscViewer);
EXTERN PetscErrorCode DMDestroy(DM);
EXTERN PetscErrorCode DMCreateGlobalVector(DM,Vec*);
EXTERN PetscErrorCode DMGetColoring(DM,ISColoringType,ISColoring*);
EXTERN PetscErrorCode DMGetMatrix(DM,const MatType,Mat*);
EXTERN PetscErrorCode DMGetInterpolation(DM,DM,Mat*,Vec*);
EXTERN PetscErrorCode DMGetInjection(DM,DM,VecScatter*);
EXTERN PetscErrorCode DMRefine(DM,MPI_Comm,DM*);
EXTERN PetscErrorCode DMGetInterpolationScale(DM,DM,Mat,Vec*);

typedef struct NLF_DAAD* NLF;

/*S
     DMMG -  Data structure to easily manage multi-level non-linear solvers on grids managed by DM
          
   Level: intermediate

  Concepts: multigrid, Newton-multigrid

.seealso:  VecPackCreate(), DA, VecPack, DM, DMMGCreate(), DMMGSetKSP(), DMMGSetSNES()
S*/
typedef struct _p_DMMG *DMMG;
struct _p_DMMG {
  DM             dm;                   /* grid information for this level */
  Vec            x,b,r;                /* global vectors used in multigrid preconditioner for this level*/
  Mat            J;                    /* matrix on this level */
  Mat            R;                    /* restriction to next coarser level (not defined on level 0) */
  PetscInt       nlevels;              /* number of levels above this one (total number of levels on level 0)*/
  MPI_Comm       comm;
  PetscErrorCode (*solve)(DMMG*,PetscInt);
  void           *user;         
  PetscTruth     galerkin;                  /* for A_c = R*A*R^T */

  /* KSP only */
  KSP            ksp;             
  PetscErrorCode (*rhs)(DMMG,Vec);
  PetscTruth     matricesset;               /* User had called DMMGSetKSP() and the matrices have been computed */

  /* SNES only */
  Mat            B;
  Vec            Rscale;                 /* scaling to restriction before computing Jacobian */
  PetscErrorCode (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);  
  PetscErrorCode (*computefunction)(SNES,Vec,Vec,void*);  

  PetscTruth     updatejacobian;         /* compute new Jacobian when DMMGComputeJacobian_Multigrid() is called */
  PetscInt       updatejacobianperiod;   /* how often, inside a SNES, the Jacobian is recomputed */

  MatFDColoring  fdcoloring;             /* only used with FD coloring for Jacobian */  
  SNES           snes;                  
  PetscErrorCode (*initialguess)(DMMG,Vec);
  Vec            w,work1,work2;         /* global vectors */
  Vec            lwork1;

  /* FAS only */
  NLF            nlf;                   /* FAS smoother object */
  VecScatter     inject;                /* inject from this level to the next coarsest */
  PetscTruth     monitor,monitorall;
  PetscInt       presmooth,postsmooth,coarsesmooth;
  PetscReal      rtol,abstol,rrtol;       /* convergence tolerance */   
  
};

EXTERN PetscErrorCode DMMGCreate(MPI_Comm,PetscInt,void*,DMMG**);
EXTERN PetscErrorCode DMMGDestroy(DMMG*);
EXTERN PetscErrorCode DMMGSetUp(DMMG*);
EXTERN PetscErrorCode DMMGSetKSP(DMMG*,PetscErrorCode (*)(DMMG,Vec),PetscErrorCode (*)(DMMG,Mat));
EXTERN PetscErrorCode DMMGSetSNES(DMMG*,PetscErrorCode (*)(SNES,Vec,Vec,void*),PetscErrorCode (*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*));
EXTERN PetscErrorCode DMMGSetInitialGuess(DMMG*,PetscErrorCode (*)(DMMG,Vec));
EXTERN PetscErrorCode DMMGInitialGuessCurrent(DMMG,Vec);
EXTERN PetscErrorCode DMMGView(DMMG*,PetscViewer);
EXTERN PetscErrorCode DMMGSolve(DMMG*);
EXTERN PetscErrorCode DMMGSetUseMatrixFree(DMMG*);
EXTERN PetscErrorCode DMMGSetDM(DMMG*,DM);
EXTERN PetscErrorCode DMMGSetUpLevel(DMMG*,KSP,PetscInt);
EXTERN PetscErrorCode DMMGSetUseGalerkinCoarse(DMMG*);
EXTERN PetscErrorCode DMMGSetNullSpace(DMMG*,PetscTruth,PetscInt,PetscErrorCode (*)(DMMG,Vec[]));

EXTERN PetscErrorCode DMMGSetSNESLocal_Private(DMMG*,DALocalFunction1,DALocalFunction1,DALocalFunction1,DALocalFunction1);
#if defined(PETSC_HAVE_ADIC)
#  define DMMGSetSNESLocal(dmmg,function,jacobian,ad_function,admf_function) \
  DMMGSetSNESLocal_Private(dmmg,(DALocalFunction1)function,(DALocalFunction1)jacobian,(DALocalFunction1)(ad_function),(DALocalFunction1)(admf_function))
#else
#  define DMMGSetSNESLocal(dmmg,function,jacobian,ad_function,admf_function) DMMGSetSNESLocal_Private(dmmg,(DALocalFunction1)function,(DALocalFunction1)jacobian,(DALocalFunction1)0,(DALocalFunction1)0)
#endif

EXTERN PetscErrorCode DMMGSetSNESLocali_Private(DMMG*,PetscErrorCode(*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*),PetscErrorCode(*)(DALocalInfo*,MatStencil*,void*,void*,void*),PetscErrorCode(*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMMGSetSNESLocali(dmmg,function,ad_function,admf_function) DMMGSetSNESLocali_Private(dmmg,(PetscErrorCode(*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*))function,(PetscErrorCode(*)(DALocalInfo*,MatStencil*,void*,void*,void*))(ad_function),(PetscErrorCode(*)(DALocalInfo*,MatStencil*,void*,void*,void*))(admf_function))
#else
#  define DMMGSetSNESLocali(dmmg,function,ad_function,admf_function) DMMGSetSNESLocali_Private(dmmg,(PetscErrorCode(*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*))function,0,0)
#endif

/*MC
   DMMGGetb - Returns the right hand side vector from a DMMG solve on the finest grid

   Synopsis:
   Vec DMMGGetb(DMMG *dmmg)

   Not Collective, but resulting vector is parallel

   Input Parameters:
.   dmmg - DMMG solve context

   Level: intermediate

   Fortran Usage:
.     DMMGGetb(DMMG dmmg,Vec b,PetscErrorCode ierr)

.seealso: DMMGCreate(), DMMGSetSNES(), DMMGSetKSP(), DMMGSetSNESLocal(), DMMGGetb()

M*/
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
.     DMMGGetx(DMMG dmmg,Vec x,PetscErrorCode ierr)

.seealso: DMMGCreate(), DMMGSetSNES(), DMMGSetKSP(), DMMGSetSNESLocal()

M*/
#define DMMGGetx(ctx)              (ctx)[(ctx)[0]->nlevels-1]->x

/*MC
   DMMGGetJ - Returns the Jacobian (matrix) for the finest level

   Synopsis:
   Mat DMMGGetJ(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetB()

M*/
#define DMMGGetJ(ctx)              (ctx)[(ctx)[0]->nlevels-1]->J

/*MC
   DMMGGetComm - Returns the MPI_Comm for the finest level

   Synopsis:
   MPI_Comm DMMGGetJ(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ()

M*/
#define DMMGGetComm(ctx)           (ctx)[(ctx)[0]->nlevels-1]->comm

/*MC
   DMMGGetB - Returns the matrix for the finest level used to construct the preconditioner; usually 
              the same as the Jacobian

   Synopsis:
   Mat DMMGGetJ(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ()

M*/
#define DMMGGetB(ctx)              (ctx)[(ctx)[0]->nlevels-1]->B

/*MC
   DMMGGetFine - Returns the DMMG associated with the finest level

   Synopsis:
   DMMG DMMGGetFine(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ()

M*/
#define DMMGGetFine(ctx)           (ctx)[(ctx)[0]->nlevels-1]


/*MC
   DMMGGetKSP - Gets the KSP object (linear solver object) for the finest level

   Synopsis:
   KSP DMMGGetKSP(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

   Notes: If this is a linear problem (i.e. DMMGSetKSP() was used) then this is the 
     master linear solver. If this is a nonlinear problem (i.e. DMMGSetSNES() was used) this
     returns the KSP (linear solver) that is associated with the SNES (nonlinear solver)

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ(), KSPGetSNES()

M*/
#define DMMGGetKSP(ctx)            (ctx)[(ctx)[0]->nlevels-1]->ksp

/*MC
   DMMGGetSNES - Gets the SNES object (nonlinear solver) for the finest level

   Synopsis:
   SNES DMMGGetSNES(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

   Notes: If this is a linear problem (i.e. DMMGSetKSP() was used) then this returns PETSC_NULL

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ(), KSPGetKSP()

M*/
#define DMMGGetSNES(ctx)           (ctx)[(ctx)[0]->nlevels-1]->snes

/*MC
   DMMGGetDA - Gets the DA object on the finest level

   Synopsis:
   DA DMMGGetDA(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

   Notes: Use only if the DMMG was created with a DA, not a VecPack

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ(), KSPGetKSP(), DMMGGetVecPack()

M*/
#define DMMGGetDA(ctx)             (DA)((ctx)[(ctx)[0]->nlevels-1]->dm)

/*MC
   DMMGGetVecPack - Gets the VecPack object on the finest level

   Synopsis:
   VecPack DMMGGetVecPack(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

   Notes: Use only if the DMMG was created with a DA, not a VecPack

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ(), KSPGetKSP(), DMMGGetDA()

M*/
#define DMMGGetVecPack(ctx)        (VecPack)((ctx)[(ctx)[0]->nlevels-1]->dm)

/*MC
   DMMGGetUser - Returns the user context for a particular level

   Synopsis:
   void* DMMGGetUser(DMMG *dmmg,PetscInt level)

   Not Collective

   Input Parameters:
+   dmmg - DMMG solve context
-   level - the number of the level you want the context for

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser()

M*/
#define DMMGGetUser(ctx,level)     ((ctx)[level]->user)

/*MC
   DMMGSetUser - Sets the user context for a particular level

   Synopsis:
   PetscErrorCode DMMGSetUser(DMMG *dmmg,PetscInt level,void *ctx)

   Not Collective

   Input Parameters:
+   dmmg - DMMG solve context
.   level - the number of the level you want the context for
-   ctx - the context

   Level: intermediate

   Note: if the context is the same for each level just pass it in with 
         DMMGCreate() and don't call this macro

.seealso: DMMGCreate(), DMMGGetUser()

M*/
#define DMMGSetUser(ctx,level,usr) ((ctx)[level]->user = usr,0)

/*MC
   DMMGGetLevels - Gets the number of levels in a DMMG object

   Synopsis:
   PetscInt DMMGGetLevels(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGGetUser()

M*/
#define DMMGGetLevels(ctx)         (ctx)[0]->nlevels

/*MC
   DMMGGetDMMG - Returns the DMMG struct for the finest level

   Synopsis:
   DMMG DMMGGetDMMG(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetB()

M*/
#define DMMGGetDMMG(ctx)              (ctx)[(ctx)[0]->nlevels-1]

PETSC_EXTERN_CXX_END
#endif

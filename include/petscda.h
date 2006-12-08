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

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), VecScatter, DACreate(), DM, VecPack
S*/
typedef struct _p_DA* DA;

/*E
    DAStencilType - Determines if the stencil extends only along the coordinate directions, or also
      to the northeast, northwest etc

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA, DACreate()
E*/
typedef enum { DA_STENCIL_STAR,DA_STENCIL_BOX } DAStencilType;

/*MC
     DA_STENCIL_STAR - "Star"-type stencil. In logical grid coordinates, only (i,j,k), (i+s,j,k), (i,j+s,k),
                       (i,j,k+s) are in the stencil  NOT, for example, (i+s,j+s,k)

     Level: beginner

.seealso: DA_STENCIL_BOX, DAStencilType
M*/

/*MC
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

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetInterpolationType(DA,DAInterpolationType);

/*E
    DAElementType - Defines the type of elements that will be returned by 
       DAGetElements.

   Level: beginner

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA, DAGetInterpolation(), DASetInterpolationType(), 
          DASetElementType(), DAGetElements(), DARestoreElements(), DACreate()
E*/
typedef enum { DA_ELEMENT_P1, DA_ELEMENT_Q1 } DAElementType;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetElementType(DA,DAElementType);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetElements(DA,PetscInt *,const PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DARestoreElements(DA,PetscInt *,const PetscInt*[]);


#define DAXPeriodic(pt) ((pt)==DA_XPERIODIC||(pt)==DA_XYPERIODIC||(pt)==DA_XZPERIODIC||(pt)==DA_XYZPERIODIC)
#define DAYPeriodic(pt) ((pt)==DA_YPERIODIC||(pt)==DA_XYPERIODIC||(pt)==DA_YZPERIODIC||(pt)==DA_XYZPERIODIC)
#define DAZPeriodic(pt) ((pt)==DA_ZPERIODIC||(pt)==DA_XZPERIODIC||(pt)==DA_YZPERIODIC||(pt)==DA_XYZPERIODIC)

typedef enum { DA_X,DA_Y,DA_Z } DADirection;

extern PetscCookie PETSCDM_DLLEXPORT DA_COOKIE;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreate1d(MPI_Comm,DAPeriodicType,PetscInt,PetscInt,PetscInt,PetscInt*,DA *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,DA *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*,DA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreate(MPI_Comm,PetscInt,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*,DA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DADestroy(DA);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAView(DA,PetscViewer);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGlobalToLocalBegin(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGlobalToLocalEnd(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGlobalToNaturalBegin(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGlobalToNaturalEnd(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DANaturalToGlobalBegin(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DANaturalToGlobalEnd(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DALocalToLocalBegin(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DALocalToLocalEnd(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DALocalToGlobal(DA,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DALocalToGlobalBegin(DA,Vec,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DALocalToGlobalEnd(DA,Vec,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetOwnershipRange(DA,PetscInt **,PetscInt **,PetscInt **);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreateGlobalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreateNaturalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreateLocalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetLocalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DARestoreLocalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetGlobalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DARestoreGlobalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DALoad(PetscViewer,PetscInt,PetscInt,PetscInt,DA *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetCorners(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetGhostCorners(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetInfo(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,DAPeriodicType*,DAStencilType*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetProcessorSubset(DA,DADirection,PetscInt,MPI_Comm*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DARefine(DA,MPI_Comm,DA*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGlobalToNaturalAllCreate(DA,VecScatter*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DANaturalAllToGlobalCreate(DA,VecScatter*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetGlobalIndices(DA,PetscInt*,PetscInt**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetISLocalToGlobalMapping(DA,ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetISLocalToGlobalMappingBlck(DA,ISLocalToGlobalMapping*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetScatter(DA,VecScatter*,VecScatter*,VecScatter*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetAO(DA,AO*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DASetCoordinates(DA,Vec); 
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetCoordinates(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetGhostedCoordinates(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetCoordinateDA(DA,DA *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DASetUniformCoordinates(DA,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DASetFieldName(DA,PetscInt,const char[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetFieldName(DA,PetscInt,char **);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAVecGetArray(DA,Vec,void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAVecRestoreArray(DA,Vec,void *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAVecGetArrayDOF(DA,Vec,void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAVecRestoreArrayDOF(DA,Vec,void *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DASplitComm2d(MPI_Comm,PetscInt,PetscInt,PetscInt,MPI_Comm*);

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
typedef struct _n_SDA* SDA;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*,SDA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,SDA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDACreate1d(MPI_Comm,DAPeriodicType,PetscInt,PetscInt,PetscInt,PetscInt*,SDA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDADestroy(SDA);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDALocalToLocalBegin(SDA,PetscScalar*,InsertMode,PetscScalar*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDALocalToLocalEnd(SDA,PetscScalar*,InsertMode,PetscScalar*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDAGetCorners(SDA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDAGetGhostCorners(SDA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDAArrayView(SDA,PetscScalar*,PetscViewer);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    MatRegisterDAAD(void);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    MatCreateDAAD(DA,Mat*);

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
  PetscInt       dim,dof,sw;
  PetscInt       mx,my,mz;    /* global number of grid points in each direction */
  PetscInt       xs,ys,zs;    /* starting pointd of this processor, excluding ghosts */
  PetscInt       xm,ym,zm;    /* number of grid points on this processor, excluding ghosts */
  PetscInt       gxs,gys,gzs;    /* starting point of this processor including ghosts */
  PetscInt       gxm,gym,gzm;    /* number of grid points on this processor including ghosts */
  DAPeriodicType pt;
  DAStencilType  st;
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
    
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetLocalInfo(DA,DALocalInfo*);
typedef PetscErrorCode (*DALocalFunction1)(DALocalInfo*,void*,void*,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAFormFunctionLocal(DA, DALocalFunction1, Vec, Vec, void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAFormFunctionLocalGhost(DA, DALocalFunction1, Vec, Vec, void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAFormJacobianLocal(DA, DALocalFunction1, Vec, Mat, void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAFormFunction1(DA,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAFormFunction(DA,PetscErrorCode (*)(void),Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAFormFunctioni1(DA,PetscInt,Vec,PetscScalar*,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAFormFunctionib1(DA,PetscInt,Vec,PetscScalar*,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAComputeJacobian1WithAdic(DA,Vec,Mat,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAComputeJacobian1WithAdifor(DA,Vec,Mat,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAMultiplyByJacobian1WithAdic(DA,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAMultiplyByJacobian1WithAdifor(DA,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAMultiplyByJacobian1WithAD(DA,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAComputeJacobian1(DA,Vec,Mat,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetLocalFunction(DA,DALocalFunction1*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalFunction(DA,DALocalFunction1);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalFunctioni(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalFunctionib(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalJacobian(DA,DALocalFunction1);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalAdicFunction_Private(DA,DALocalFunction1);

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

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalAdicMFFunction_Private(DA,DALocalFunction1);
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicMFFunction(a,d) DASetLocalAdicMFFunction_Private(a,(DALocalFunction1)d)
#else
#  define DASetLocalAdicMFFunction(a,d) DASetLocalAdicMFFunction_Private(a,0)
#endif
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalAdicFunctioni_Private(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicFunctioni(a,d) DASetLocalAdicFunctioni_Private(a,(PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DASetLocalAdicFunctioni(a,d) DASetLocalAdicFunctioni_Private(a,0)
#endif
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalAdicMFFunctioni_Private(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicMFFunctioni(a,d) DASetLocalAdicMFFunctioni_Private(a,(PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DASetLocalAdicMFFunctioni(a,d) DASetLocalAdicMFFunctioni_Private(a,0)
#endif

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalAdicFunctionib_Private(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicFunctionib(a,d) DASetLocalAdicFunctionib_Private(a,(PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DASetLocalAdicFunctionib(a,d) DASetLocalAdicFunctionib_Private(a,0)
#endif
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalAdicMFFunctionib_Private(DA,PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DASetLocalAdicMFFunctionib(a,d) DASetLocalAdicMFFunctionib_Private(a,(PetscErrorCode (*)(DALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DASetLocalAdicMFFunctionib(a,d) DASetLocalAdicMFFunctionib_Private(a,0)
#endif

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAFormFunctioniTest1(DA,void*);

#include "petscmat.h"
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetColoring(DA,ISColoringType,ISColoring *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetMatrix(DA, MatType,Mat *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetGetMatrix(DA,PetscErrorCode (*)(DA, MatType,Mat *));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetInterpolation(DA,DA,Mat*,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetInjection(DA,DA,VecScatter*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetBlockFills(DA,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetMatPreallocateOnly(DA,PetscTruth);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetRefinementFactor(DA,PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetRefinementFactor(DA,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetAdicArray(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DARestoreAdicArray(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetAdicMFArray(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetAdicMFArray4(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetAdicMFArray9(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetAdicMFArrayb(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DARestoreAdicMFArray(DA,PetscTruth,void**,void**,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DARestoreArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  ad_DAGetArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  ad_DARestoreArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  admf_DAGetArray(DA,PetscTruth,void**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  admf_DARestoreArray(DA,PetscTruth,void**);

#include "petscpf.h"
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DACreatePF(DA,PF*);

/*S
     VecPack - Abstract PETSc object that manages treating several distinct vectors as if they
        were one.   The VecPack routines allow one to manage a nonlinear solver that works on a
        vector that consists of several distinct parts. This is mostly used for LNKS solvers, 
        that is design optimization problems that are written as a nonlinear system

   Level: beginner

  Concepts: multi-component, LNKS solvers

.seealso:  VecPackCreate(), VecPackDestroy(), DM
S*/
typedef struct _p_VecPack* VecPack;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackCreate(MPI_Comm,VecPack*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackDestroy(VecPack);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackAddArray(VecPack,PetscMPIInt,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackAddDA(VecPack,DA);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackAddVecScatter(VecPack,VecScatter);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackScatter(VecPack,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackGather(VecPack,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackGetAccess(VecPack,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackRestoreAccess(VecPack,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackGetLocalVectors(VecPack,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackGetEntries(VecPack,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackRestoreLocalVectors(VecPack,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackCreateGlobalVector(VecPack,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackGetGlobalIndices(VecPack,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackRefine(VecPack,MPI_Comm,VecPack*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackGetInterpolation(VecPack,VecPack,Mat*,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackGetMatrix(VecPack,MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  VecPackGetColoring(VecPack,ISColoringType,ISColoring*);

/*S
     Slice - Abstract PETSc object that manages distributed field data for a simple unstructured matrix

   Level: beginner

  Concepts: distributed array

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), VecScatter, DACreate(), VecPackCreate(), VecPack
S*/
typedef struct _p_Sliced* Sliced;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedView(Sliced,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedCreate(MPI_Comm,Sliced*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedDestroy(Sliced);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedCreateGlobalVector(Sliced,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedGetMatrix(Sliced, MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedGetGlobalIndices(Sliced,PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedSetPreallocation(Sliced,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedSetGhosts(Sliced,PetscInt,PetscInt,PetscInt,const PetscInt[]);

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

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMView(DM,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDestroy(DM);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCreateGlobalVector(DM,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetColoring(DM,ISColoringType,ISColoring*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetMatrix(DM, MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetInterpolation(DM,DM,Mat*,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetInjection(DM,DM,VecScatter*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMRefine(DM,MPI_Comm,DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCoarsen(DM,MPI_Comm,DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMRefineHierarchy(DM,PetscInt,DM**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCoarsenHierarchy(DM,PetscInt,DM**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetInterpolationScale(DM,DM,Mat,Vec*);

typedef struct NLF_DAAD* NLF;

#include <petscbag.h>

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabOpen(MPI_Comm, const char [], PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabDestroy(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabOutputBag(PetscViewer, const char [], PetscBag);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabOutputVec(PetscViewer, const char [], Vec);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabOutputVecDA(PetscViewer, const char [], Vec, DA);

PETSC_EXTERN_CXX_END
#endif

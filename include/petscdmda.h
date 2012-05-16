#if !defined(__PETSCDMDA_H)
#define __PETSCDMDA_H

#include "petscdm.h"
#include "petscpf.h"
#include "petscao.h"

/*E
    DMDAStencilType - Determines if the stencil extends only along the coordinate directions, or also
      to the northeast, northwest etc

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum { DMDA_STENCIL_STAR,DMDA_STENCIL_BOX } DMDAStencilType;

/*MC
     DMDA_STENCIL_STAR - "Star"-type stencil. In logical grid coordinates, only (i,j,k), (i+s,j,k), (i,j+s,k),
                       (i,j,k+s) are in the stencil  NOT, for example, (i+s,j+s,k)

     Level: beginner

.seealso: DMDA_STENCIL_BOX, DMDAStencilType
M*/

/*MC
     DMDA_STENCIL_BOX - "Box"-type stencil. In logical grid coordinates, any of (i,j,k), (i+s,j+r,k+t) may 
                      be in the stencil.

     Level: beginner

.seealso: DMDA_STENCIL_STAR, DMDAStencilType
M*/

/*E
    DMDABoundaryType - Describes the choice for fill of ghost cells on physical domain boundaries.

   Level: beginner

   A boundary may be of type DMDA_BOUNDARY_NONE (no ghost nodes), DMDA_BOUNDARY_GHOST (ghost nodes 
   exist but aren't filled, you can put values into them and then apply a stencil that uses those ghost locations),
   DMDA_BOUNDARY_MIRROR (not yet implemented), or DMDA_BOUNDARY_PERIODIC
   (ghost nodes filled by the opposite edge of the domain).

   Note: This is information for the boundary of the __PHYSICAL__ domain. It has nothing to do with boundaries between 
     processes, that width is always determined by the stencil width, see DMDASetStencilWidth().

.seealso: DMDASetBoundaryType(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum { DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_MIRROR, DMDA_BOUNDARY_PERIODIC } DMDABoundaryType;

PETSC_EXTERN const char *DMDABoundaryTypes[];

/*E
    DMDAInterpolationType - Defines the type of interpolation that will be returned by 
       DMCreateInterpolation.

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateInterpolation(), DMDASetInterpolationType(), DMDACreate()
E*/
typedef enum { DMDA_Q0, DMDA_Q1 } DMDAInterpolationType;

PETSC_EXTERN PetscErrorCode DMDASetInterpolationType(DM,DMDAInterpolationType);
PETSC_EXTERN PetscErrorCode DMDAGetInterpolationType(DM,DMDAInterpolationType*);

/*E
    DMDAElementType - Defines the type of elements that will be returned by 
       DMDAGetElements()

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateInterpolation(), DMDASetInterpolationType(), 
          DMDASetElementType(), DMDAGetElements(), DMDARestoreElements(), DMDACreate()
E*/
typedef enum { DMDA_ELEMENT_P1, DMDA_ELEMENT_Q1 } DMDAElementType;

PETSC_EXTERN PetscErrorCode DMDASetElementType(DM,DMDAElementType);
PETSC_EXTERN PetscErrorCode DMDAGetElementType(DM,DMDAElementType*);
PETSC_EXTERN PetscErrorCode DMDAGetElements(DM,PetscInt *,PetscInt *,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode DMDARestoreElements(DM,PetscInt *,PetscInt *,const PetscInt*[]);

typedef enum { DMDA_X,DMDA_Y,DMDA_Z } DMDADirection;

#define MATSEQUSFFT        "sequsfft"

PETSC_EXTERN PetscErrorCode DMDACreate(MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMDASetDim(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMDASetSizes(DM,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DMDACreate1d(MPI_Comm,DMDABoundaryType,PetscInt,PetscInt,PetscInt,const PetscInt[],DM *);
PETSC_EXTERN PetscErrorCode DMDACreate2d(MPI_Comm,DMDABoundaryType,DMDABoundaryType,DMDAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],DM*);
PETSC_EXTERN PetscErrorCode DMDACreate3d(MPI_Comm,DMDABoundaryType,DMDABoundaryType,DMDABoundaryType,DMDAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],DM*);

PETSC_EXTERN PetscErrorCode DMDAGlobalToNaturalBegin(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMDAGlobalToNaturalEnd(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMDANaturalToGlobalBegin(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMDANaturalToGlobalEnd(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMDALocalToLocalBegin(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMDALocalToLocalEnd(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMDACreateNaturalVector(DM,Vec *);

PETSC_EXTERN PetscErrorCode DMDAGetCorners(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDAGetGhostCorners(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDAGetInfo(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,DMDABoundaryType*,DMDABoundaryType*,DMDABoundaryType*,DMDAStencilType*);
PETSC_EXTERN PetscErrorCode DMDAGetProcessorSubset(DM,DMDADirection,PetscInt,MPI_Comm*);
PETSC_EXTERN PetscErrorCode DMDAGetProcessorSubsets(DM,DMDADirection,MPI_Comm*);

PETSC_EXTERN PetscErrorCode DMDAGlobalToNaturalAllCreate(DM,VecScatter*);
PETSC_EXTERN PetscErrorCode DMDANaturalAllToGlobalCreate(DM,VecScatter*);

PETSC_EXTERN PetscErrorCode DMDAGetGlobalIndices(DM,PetscInt*,PetscInt**);

PETSC_EXTERN PetscErrorCode DMDAGetScatter(DM,VecScatter*,VecScatter*,VecScatter*);
PETSC_EXTERN PetscErrorCode DMDAGetNeighbors(DM,const PetscMPIInt**);

PETSC_EXTERN PetscErrorCode DMDAGetAO(DM,AO*);
PETSC_EXTERN PetscErrorCode DMDASetCoordinates(DM,Vec);
PETSC_EXTERN PetscErrorCode DMDASetGhostedCoordinates(DM,Vec);
PETSC_EXTERN PetscErrorCode DMDAGetCoordinates(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMDAGetGhostedCoordinates(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMDAGetCoordinateDA(DM,DM *);
PETSC_EXTERN PetscErrorCode DMDASetUniformCoordinates(DM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode DMDAGetBoundingBox(DM,PetscReal[],PetscReal[]);
PETSC_EXTERN PetscErrorCode DMDAGetLocalBoundingBox(DM,PetscReal[],PetscReal[]);
/* function to wrap coordinates around boundary */
PETSC_EXTERN PetscErrorCode DMDAMapCoordsToPeriodicDomain(DM,PetscScalar*,PetscScalar*);

PETSC_EXTERN PetscErrorCode DMDAGetReducedDA(DM,PetscInt,DM*);

PETSC_EXTERN PetscErrorCode DMDASetFieldName(DM,PetscInt,const char[]);
PETSC_EXTERN PetscErrorCode DMDAGetFieldName(DM,PetscInt,const char**);

PETSC_EXTERN PetscErrorCode DMDASetBoundaryType(DM,DMDABoundaryType,DMDABoundaryType,DMDABoundaryType);
PETSC_EXTERN PetscErrorCode DMDASetDof(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMDASetStencilWidth(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMDASetOwnershipRanges(DM,const PetscInt[],const PetscInt[],const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMDAGetOwnershipRanges(DM,const PetscInt**,const PetscInt**,const PetscInt**);
PETSC_EXTERN PetscErrorCode DMDASetNumProcs(DM, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMDASetStencilType(DM, DMDAStencilType);

PETSC_EXTERN PetscErrorCode DMDAVecGetArray(DM,Vec,void *);
PETSC_EXTERN PetscErrorCode DMDAVecRestoreArray(DM,Vec,void *);

PETSC_EXTERN PetscErrorCode DMDAVecGetArrayDOF(DM,Vec,void *);
PETSC_EXTERN PetscErrorCode DMDAVecRestoreArrayDOF(DM,Vec,void *);

PETSC_EXTERN PetscErrorCode DMDASplitComm2d(MPI_Comm,PetscInt,PetscInt,PetscInt,MPI_Comm*);

/*S
     DMDALocalInfo - C struct that contains information about a structured grid and a processors logical
              location in it.

   Level: beginner

  Concepts: distributed array

  Developer note: Then entries in this struct are int instead of PetscInt so that the elements may
                  be extracted in Fortran as if from an integer array

.seealso:  DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DM, DMDAGetLocalInfo(), DMDAGetInfo()
S*/
typedef struct {
  PetscInt       dim,dof,sw;
  PetscInt       mx,my,mz;    /* global number of grid points in each direction */
  PetscInt       xs,ys,zs;    /* starting point of this processor, excluding ghosts */
  PetscInt       xm,ym,zm;    /* number of grid points on this processor, excluding ghosts */
  PetscInt       gxs,gys,gzs;    /* starting point of this processor including ghosts */
  PetscInt       gxm,gym,gzm;    /* number of grid points on this processor including ghosts */
  DMDABoundaryType bx,by,bz; /* type of ghost nodes at boundary */
  DMDAStencilType  st;
  DM             da;
} DMDALocalInfo;

/*MC
      DMDAForEachPointBegin2d - Starts a loop over the local part of a two dimensional DMDA

   Synopsis:
   void  DMDAForEachPointBegin2d(DALocalInfo *info,PetscInt i,PetscInt j);
   
   Not Collective

   Level: intermediate

.seealso: DMDAForEachPointEnd2d(), DMDAVecGetArray()
M*/
#define DMDAForEachPointBegin2d(info,i,j) {\
  PetscInt _xints = info->xs,_xinte = info->xs+info->xm,_yints = info->ys,_yinte = info->ys+info->ym;\
  for (j=_yints; j<_yinte; j++) {\
    for (i=_xints; i<_xinte; i++) {\

/*MC
      DMDAForEachPointEnd2d - Ends a loop over the local part of a two dimensional DMDA

   Synopsis:
   void  DMDAForEachPointEnd2d;
   
   Not Collective

   Level: intermediate

.seealso: DMDAForEachPointBegin2d(), DMDAVecGetArray()
M*/
#define DMDAForEachPointEnd2d }}}

/*MC
      DMDACoor2d - Structure for holding 2d (x and y) coordinates.

    Level: intermediate

    Sample Usage:
      DMDACoor2d **coors;
      Vec      vcoors;
      DM       cda;     

      DMDAGetCoordinates(da,&vcoors); 
      DMDAGetCoordinateDA(da,&cda);
      DMDAVecGetArray(cda,vcoors,&coors);
      DMDAGetCorners(cda,&mstart,&nstart,0,&m,&n,0)
      for (i=mstart; i<mstart+m; i++) {
        for (j=nstart; j<nstart+n; j++) {
          x = coors[j][i].x;
          y = coors[j][i].y;
          ......
        }
      }
      DMDAVecRestoreArray(dac,vcoors,&coors);

.seealso: DMDACoor3d, DMDAForEachPointBegin(), DMDAGetCoordinateDA(), DMDAGetCoordinates(), DMDAGetGhostCoordinates()
M*/
typedef struct {PetscScalar x,y;} DMDACoor2d;

/*MC
      DMDACoor3d - Structure for holding 3d (x, y and z) coordinates.

    Level: intermediate

    Sample Usage:
      DMDACoor3d ***coors;
      Vec      vcoors;
      DM       cda;     

      DMDAGetCoordinates(da,&vcoors); 
      DMDAGetCoordinateDA(da,&cda);
      DMDAVecGetArray(cda,vcoors,&coors);
      DMDAGetCorners(cda,&mstart,&nstart,&pstart,&m,&n,&p)
      for (i=mstart; i<mstart+m; i++) {
        for (j=nstart; j<nstart+n; j++) {
          for (k=pstart; k<pstart+p; k++) {
            x = coors[k][j][i].x;
            y = coors[k][j][i].y;
            z = coors[k][j][i].z;
          ......
        }
      }
      DMDAVecRestoreArray(dac,vcoors,&coors);

.seealso: DMDACoor2d, DMDAForEachPointBegin(), DMDAGetCoordinateDA(), DMDAGetCoordinates(), DMDAGetGhostCoordinates()
M*/
typedef struct {PetscScalar x,y,z;} DMDACoor3d;
    
PETSC_EXTERN PetscErrorCode DMDAGetLocalInfo(DM,DMDALocalInfo*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMDALocalFunction1)(DMDALocalInfo*,void*,void*,void*);
PETSC_EXTERN PetscErrorCode DMDAComputeFunctionLocal(DM, DMDALocalFunction1, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMDAComputeFunctionLocalGhost(DM, DMDALocalFunction1, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMDAFormJacobianLocal(DM, DMDALocalFunction1, Vec, Mat, void *);
PETSC_EXTERN PetscErrorCode DMDAComputeFunction1(DM,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode DMDAComputeFunction(DM,PetscErrorCode (*)(void),Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode DMDAComputeFunctioni1(DM,PetscInt,Vec,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMDAComputeFunctionib1(DM,PetscInt,Vec,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMDAComputeJacobian1WithAdic(DM,Vec,Mat,void*);
PETSC_EXTERN PetscErrorCode DMDAComputeJacobian1WithAdifor(DM,Vec,Mat,void*);
PETSC_EXTERN PetscErrorCode DMDAMultiplyByJacobian1WithAdic(DM,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode DMDAMultiplyByJacobian1WithAdifor(DM,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode DMDAMultiplyByJacobian1WithAD(DM,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode DMDAComputeJacobian1(DM,Vec,Mat,void*);
PETSC_EXTERN PetscErrorCode DMDAGetLocalFunction(DM,DMDALocalFunction1*);
PETSC_EXTERN PetscErrorCode DMDASetLocalFunction(DM,DMDALocalFunction1);
PETSC_EXTERN PetscErrorCode DMDASetLocalFunctioni(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
PETSC_EXTERN PetscErrorCode DMDASetLocalFunctionib(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
PETSC_EXTERN PetscErrorCode DMDAGetLocalJacobian(DM,DMDALocalFunction1*);
PETSC_EXTERN PetscErrorCode DMDASetLocalJacobian(DM,DMDALocalFunction1);
PETSC_EXTERN PetscErrorCode DMDASetLocalAdicFunction_Private(DM,DMDALocalFunction1);

PETSC_EXTERN PetscErrorCode MatSetDM(Mat,DM);
PETSC_EXTERN PetscErrorCode MatRegisterDAAD(void);
PETSC_EXTERN PetscErrorCode MatCreateDAAD(DM,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSeqUSFFT(Vec,DM,Mat*);

/*MC
       DMDASetLocalAdicFunction - Caches in a DM a local function computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode DMDASetLocalAdicFunction(DM da,DMDALocalFunction1 ad_lf)
   
   Logically Collective on DM

   Input Parameter:
+  da - initial distributed array
-  ad_lf - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDAGetLocalFunction(), DMDASetLocalFunction(),
          DMDASetLocalJacobian()
M*/
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicFunction(a,d) DMDASetLocalAdicFunction_Private(a,(DMDALocalFunction1)d)
#else
#  define DMDASetLocalAdicFunction(a,d) DMDASetLocalAdicFunction_Private(a,0)
#endif

PETSC_EXTERN PetscErrorCode DMDASetLocalAdicMFFunction_Private(DM,DMDALocalFunction1);
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunction(a,d) DMDASetLocalAdicMFFunction_Private(a,(DMDALocalFunction1)d)
#else
#  define DMDASetLocalAdicMFFunction(a,d) DMDASetLocalAdicMFFunction_Private(a,0)
#endif
PETSC_EXTERN PetscErrorCode DMDASetLocalAdicFunctioni_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicFunctioni(a,d) DMDASetLocalAdicFunctioni_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicFunctioni(a,d) DMDASetLocalAdicFunctioni_Private(a,0)
#endif
PETSC_EXTERN PetscErrorCode DMDASetLocalAdicMFFunctioni_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunctioni(a,d) DMDASetLocalAdicMFFunctioni_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicMFFunctioni(a,d) DMDASetLocalAdicMFFunctioni_Private(a,0)
#endif

PETSC_EXTERN PetscErrorCode DMDASetLocalAdicFunctionib_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicFunctionib(a,d) DMDASetLocalAdicFunctionib_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicFunctionib(a,d) DMDASetLocalAdicFunctionib_Private(a,0)
#endif
PETSC_EXTERN PetscErrorCode DMDASetLocalAdicMFFunctionib_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunctionib(a,d) DMDASetLocalAdicMFFunctionib_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicMFFunctionib(a,d) DMDASetLocalAdicMFFunctionib_Private(a,0)
#endif

PETSC_EXTERN PetscErrorCode DMDAComputeFunctioniTest1(DM,void*);
PETSC_EXTERN PetscErrorCode DMDASetGetMatrix(DM,PetscErrorCode (*)(DM, const MatType,Mat *));
PETSC_EXTERN PetscErrorCode DMDASetBlockFills(DM,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDASetRefinementFactor(DM,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DMDAGetRefinementFactor(DM,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode DMDAGetAdicArray(DM,PetscBool ,void*,void*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDARestoreAdicArray(DM,PetscBool ,void*,void*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDAGetAdicMFArray(DM,PetscBool ,void*,void*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDAGetAdicMFArray4(DM,PetscBool ,void*,void*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDAGetAdicMFArray9(DM,PetscBool ,void*,void*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDAGetAdicMFArrayb(DM,PetscBool ,void*,void*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDARestoreAdicMFArray(DM,PetscBool ,void*,void*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMDAGetArray(DM,PetscBool ,void*);
PETSC_EXTERN PetscErrorCode DMDARestoreArray(DM,PetscBool ,void*);
PETSC_EXTERN PetscErrorCode ad_DAGetArray(DM,PetscBool ,void*);
PETSC_EXTERN PetscErrorCode ad_DARestoreArray(DM,PetscBool ,void*);
PETSC_EXTERN PetscErrorCode admf_DAGetArray(DM,PetscBool ,void*);
PETSC_EXTERN PetscErrorCode admf_DARestoreArray(DM,PetscBool ,void*);

PETSC_EXTERN PetscErrorCode DMDACreatePF(DM,PF*);

PETSC_EXTERN PetscErrorCode DMDACreateSection(DM, PetscInt[], PetscInt[], PetscInt[], PetscInt[]);

#define DMDA_FILE_CLASSID 1211220
#endif

#if !defined(__PETSCDMDA_H)
#define __PETSCDMDA_H

#include "petscdm.h"
#include "petscpf.h"
#include "petscao.h"
PETSC_EXTERN_CXX_BEGIN

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

extern const char *DMDABoundaryTypes[];

/*E
    DMDAInterpolationType - Defines the type of interpolation that will be returned by 
       DMCreateInterpolation.

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateInterpolation(), DMDASetInterpolationType(), DMDACreate()
E*/
typedef enum { DMDA_Q0, DMDA_Q1 } DMDAInterpolationType;

extern PetscErrorCode   DMDASetInterpolationType(DM,DMDAInterpolationType);
extern PetscErrorCode   DMDAGetInterpolationType(DM,DMDAInterpolationType*);

/*E
    DMDAElementType - Defines the type of elements that will be returned by 
       DMDAGetElements()

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateInterpolation(), DMDASetInterpolationType(), 
          DMDASetElementType(), DMDAGetElements(), DMDARestoreElements(), DMDACreate()
E*/
typedef enum { DMDA_ELEMENT_P1, DMDA_ELEMENT_Q1 } DMDAElementType;

extern PetscErrorCode   DMDASetElementType(DM,DMDAElementType);
extern PetscErrorCode   DMDAGetElementType(DM,DMDAElementType*);
extern PetscErrorCode   DMDAGetElements(DM,PetscInt *,PetscInt *,const PetscInt*[]);
extern PetscErrorCode   DMDARestoreElements(DM,PetscInt *,PetscInt *,const PetscInt*[]);

typedef enum { DMDA_X,DMDA_Y,DMDA_Z } DMDADirection;

#define MATSEQUSFFT        "sequsfft"

extern PetscErrorCode  DMDACreate(MPI_Comm,DM*);
extern PetscErrorCode  DMDASetDim(DM,PetscInt);
extern PetscErrorCode  DMDASetSizes(DM,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode     DMDACreate1d(MPI_Comm,DMDABoundaryType,PetscInt,PetscInt,PetscInt,const PetscInt[],DM *);
extern PetscErrorCode     DMDACreate2d(MPI_Comm,DMDABoundaryType,DMDABoundaryType,DMDAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],DM*);
extern PetscErrorCode     DMDACreate3d(MPI_Comm,DMDABoundaryType,DMDABoundaryType,DMDABoundaryType,DMDAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],DM*);

extern PetscErrorCode     DMDAGlobalToNaturalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDAGlobalToNaturalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDANaturalToGlobalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDANaturalToGlobalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDALocalToLocalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDALocalToLocalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDACreateNaturalVector(DM,Vec *);

extern PetscErrorCode     DMDAGetCorners(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode     DMDAGetGhostCorners(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode     DMDAGetInfo(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,DMDABoundaryType*,DMDABoundaryType*,DMDABoundaryType*,DMDAStencilType*);
extern PetscErrorCode     DMDAGetProcessorSubset(DM,DMDADirection,PetscInt,MPI_Comm*);
extern PetscErrorCode     DMDAGetProcessorSubsets(DM,DMDADirection,MPI_Comm*);

extern PetscErrorCode     DMDAGlobalToNaturalAllCreate(DM,VecScatter*);
extern PetscErrorCode     DMDANaturalAllToGlobalCreate(DM,VecScatter*);

extern PetscErrorCode     DMDAGetGlobalIndices(DM,PetscInt*,PetscInt**);

extern PetscErrorCode     DMDAGetScatter(DM,VecScatter*,VecScatter*,VecScatter*);
extern PetscErrorCode     DMDAGetNeighbors(DM,const PetscMPIInt**);

extern PetscErrorCode     DMDAGetAO(DM,AO*);
extern PetscErrorCode     DMDASetCoordinates(DM,Vec);
extern PetscErrorCode     DMDASetGhostedCoordinates(DM,Vec);
extern PetscErrorCode     DMDAGetCoordinates(DM,Vec *);
extern PetscErrorCode     DMDAGetGhostedCoordinates(DM,Vec *);
extern PetscErrorCode     DMDAGetCoordinateDA(DM,DM *);
extern PetscErrorCode     DMDASetUniformCoordinates(DM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode     DMDAGetBoundingBox(DM,PetscReal[],PetscReal[]);
extern PetscErrorCode     DMDAGetLocalBoundingBox(DM,PetscReal[],PetscReal[]);
/* function to wrap coordinates around boundary */
extern PetscErrorCode     DMDAMapCoordsToPeriodicDomain(DM,PetscScalar*,PetscScalar*);

extern PetscErrorCode     DMDAGetReducedDA(DM,PetscInt,DM*);

extern PetscErrorCode     DMDASetFieldName(DM,PetscInt,const char[]);
extern PetscErrorCode     DMDAGetFieldName(DM,PetscInt,const char**);

extern PetscErrorCode  DMDASetBoundaryType(DM,DMDABoundaryType,DMDABoundaryType,DMDABoundaryType);
extern PetscErrorCode  DMDASetDof(DM, PetscInt);
extern PetscErrorCode  DMDASetStencilWidth(DM, PetscInt);
extern PetscErrorCode  DMDASetOwnershipRanges(DM,const PetscInt[],const PetscInt[],const PetscInt[]);
extern PetscErrorCode  DMDAGetOwnershipRanges(DM,const PetscInt**,const PetscInt**,const PetscInt**);
extern PetscErrorCode  DMDASetNumProcs(DM, PetscInt, PetscInt, PetscInt);
extern PetscErrorCode  DMDASetStencilType(DM, DMDAStencilType);

extern PetscErrorCode     DMDAVecGetArray(DM,Vec,void *);
extern PetscErrorCode     DMDAVecRestoreArray(DM,Vec,void *);

extern PetscErrorCode     DMDAVecGetArrayDOF(DM,Vec,void *);
extern PetscErrorCode     DMDAVecRestoreArrayDOF(DM,Vec,void *);

extern PetscErrorCode     DMDASplitComm2d(MPI_Comm,PetscInt,PetscInt,PetscInt,MPI_Comm*);

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
    
extern PetscErrorCode   DMDAGetLocalInfo(DM,DMDALocalInfo*);
typedef PetscErrorCode (*DMDALocalFunction1)(DMDALocalInfo*,void*,void*,void*);
extern PetscErrorCode   DMDAComputeFunctionLocal(DM, DMDALocalFunction1, Vec, Vec, void *);
extern PetscErrorCode   DMDAComputeFunctionLocalGhost(DM, DMDALocalFunction1, Vec, Vec, void *);
extern PetscErrorCode   DMDAFormJacobianLocal(DM, DMDALocalFunction1, Vec, Mat, void *);
extern PetscErrorCode   DMDAComputeFunction1(DM,Vec,Vec,void*);
extern PetscErrorCode   DMDAComputeFunction(DM,PetscErrorCode (*)(void),Vec,Vec,void*);
extern PetscErrorCode   DMDAComputeFunctioni1(DM,PetscInt,Vec,PetscScalar*,void*);
extern PetscErrorCode   DMDAComputeFunctionib1(DM,PetscInt,Vec,PetscScalar*,void*);
extern PetscErrorCode   DMDAComputeJacobian1WithAdic(DM,Vec,Mat,void*);
extern PetscErrorCode   DMDAComputeJacobian1WithAdifor(DM,Vec,Mat,void*);
extern PetscErrorCode   DMDAMultiplyByJacobian1WithAdic(DM,Vec,Vec,Vec,void*);
extern PetscErrorCode   DMDAMultiplyByJacobian1WithAdifor(DM,Vec,Vec,Vec,void*);
extern PetscErrorCode   DMDAMultiplyByJacobian1WithAD(DM,Vec,Vec,Vec,void*);
extern PetscErrorCode   DMDAComputeJacobian1(DM,Vec,Mat,void*);
extern PetscErrorCode   DMDAGetLocalFunction(DM,DMDALocalFunction1*);
extern PetscErrorCode   DMDASetLocalFunction(DM,DMDALocalFunction1);
extern PetscErrorCode   DMDASetLocalFunctioni(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
extern PetscErrorCode   DMDASetLocalFunctionib(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
extern PetscErrorCode   DMDAGetLocalJacobian(DM,DMDALocalFunction1*);
extern PetscErrorCode   DMDASetLocalJacobian(DM,DMDALocalFunction1);
extern PetscErrorCode   DMDASetLocalAdicFunction_Private(DM,DMDALocalFunction1);

extern PetscErrorCode MatSetDM(Mat,DM);
extern PetscErrorCode MatRegisterDAAD(void);
extern PetscErrorCode MatCreateDAAD(DM,Mat*);
extern PetscErrorCode MatCreateSeqUSFFT(Vec,DM,Mat*);

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

extern PetscErrorCode   DMDASetLocalAdicMFFunction_Private(DM,DMDALocalFunction1);
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunction(a,d) DMDASetLocalAdicMFFunction_Private(a,(DMDALocalFunction1)d)
#else
#  define DMDASetLocalAdicMFFunction(a,d) DMDASetLocalAdicMFFunction_Private(a,0)
#endif
extern PetscErrorCode   DMDASetLocalAdicFunctioni_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicFunctioni(a,d) DMDASetLocalAdicFunctioni_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicFunctioni(a,d) DMDASetLocalAdicFunctioni_Private(a,0)
#endif
extern PetscErrorCode   DMDASetLocalAdicMFFunctioni_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunctioni(a,d) DMDASetLocalAdicMFFunctioni_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicMFFunctioni(a,d) DMDASetLocalAdicMFFunctioni_Private(a,0)
#endif

extern PetscErrorCode   DMDASetLocalAdicFunctionib_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicFunctionib(a,d) DMDASetLocalAdicFunctionib_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicFunctionib(a,d) DMDASetLocalAdicFunctionib_Private(a,0)
#endif
extern PetscErrorCode   DMDASetLocalAdicMFFunctionib_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunctionib(a,d) DMDASetLocalAdicMFFunctionib_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicMFFunctionib(a,d) DMDASetLocalAdicMFFunctionib_Private(a,0)
#endif

extern PetscErrorCode   DMDAComputeFunctioniTest1(DM,void*);
extern PetscErrorCode   DMDASetGetMatrix(DM,PetscErrorCode (*)(DM, const MatType,Mat *));
extern PetscErrorCode   DMDASetBlockFills(DM,PetscInt*,PetscInt*);
extern PetscErrorCode   DMDASetRefinementFactor(DM,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode   DMDAGetRefinementFactor(DM,PetscInt*,PetscInt*,PetscInt*);

extern PetscErrorCode   DMDAGetAdicArray(DM,PetscBool ,void*,void*,PetscInt*);
extern PetscErrorCode   DMDARestoreAdicArray(DM,PetscBool ,void*,void*,PetscInt*);
extern PetscErrorCode   DMDAGetAdicMFArray(DM,PetscBool ,void*,void*,PetscInt*);
extern PetscErrorCode   DMDAGetAdicMFArray4(DM,PetscBool ,void*,void*,PetscInt*);
extern PetscErrorCode   DMDAGetAdicMFArray9(DM,PetscBool ,void*,void*,PetscInt*);
extern PetscErrorCode   DMDAGetAdicMFArrayb(DM,PetscBool ,void*,void*,PetscInt*);
extern PetscErrorCode   DMDARestoreAdicMFArray(DM,PetscBool ,void*,void*,PetscInt*);
extern PetscErrorCode   DMDAGetArray(DM,PetscBool ,void*);
extern PetscErrorCode   DMDARestoreArray(DM,PetscBool ,void*);
extern PetscErrorCode   ad_DAGetArray(DM,PetscBool ,void*);
extern PetscErrorCode   ad_DARestoreArray(DM,PetscBool ,void*);
extern PetscErrorCode   admf_DAGetArray(DM,PetscBool ,void*);
extern PetscErrorCode   admf_DARestoreArray(DM,PetscBool ,void*);

extern PetscErrorCode   DMDACreatePF(DM,PF*);

extern PetscErrorCode DMDACreateSection(DM, PetscInt[], PetscInt[], PetscInt[], PetscInt[]);

#define DMDA_FILE_CLASSID 1211220
PETSC_EXTERN_CXX_END
#endif

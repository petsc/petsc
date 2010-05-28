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

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), VecScatter, DACreate(), DM, DMComposite
S*/
typedef struct _p_DA* DA;

#define DAType char*
#define DA1D "da1d"
#define DA2D "da2d"
#define DA3D "da3d"

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
     DA_STENCIL_BOX - "Box"-type stencil. In logical grid coordinates, any of (i,j,k), (i+s,j+r,k+t) may 
                      be in the stencil.

     Level: beginner

.seealso: DA_STENCIL_STAR, DAStencilType
M*/

/*E
    DAPeriodicType - Is the domain periodic in one or more directions

   Level: beginner

   DA_XYZGHOSTED means that ghost points are put around all the physical boundaries
   in the local representation of the Vec (i.e. DACreate/GetLocalVector().

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DA, DACreate()
E*/
typedef enum { DA_NONPERIODIC,DA_XPERIODIC,DA_YPERIODIC,DA_XYPERIODIC,
               DA_XYZPERIODIC,DA_XZPERIODIC,DA_YZPERIODIC,DA_ZPERIODIC,DA_XYZGHOSTED} DAPeriodicType;
extern const char *DAPeriodicTypes[];

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
/*MC
   DAGetElements - same as DMGetElements()
   uses DA instead of DM as input

   Level: beginner
M*/
#define DAGetElements(da,a,b)      DMGetElements((DM)da,a,b)
/*MC
   DARestoreElements - same as DMRestoreElements()
   uses DA instead of DM as input

   Level: beginner
M*/
#define DARestoreElements(da,a,b)  DMRestoreElements((DM)da,a,b)


#define DAXPeriodic(pt) ((pt)==DA_XPERIODIC||(pt)==DA_XYPERIODIC||(pt)==DA_XZPERIODIC||(pt)==DA_XYZPERIODIC)
#define DAYPeriodic(pt) ((pt)==DA_YPERIODIC||(pt)==DA_XYPERIODIC||(pt)==DA_YZPERIODIC||(pt)==DA_XYZPERIODIC)
#define DAZPeriodic(pt) ((pt)==DA_ZPERIODIC||(pt)==DA_XZPERIODIC||(pt)==DA_YZPERIODIC||(pt)==DA_XYZPERIODIC)

typedef enum { DA_X,DA_Y,DA_Z } DADirection;

extern PetscCookie PETSCDM_DLLEXPORT DM_COOKIE;

#define MATSEQUSFFT        "sequsfft"

EXTERN PetscErrorCode PETSCDM_DLLEXPORT DACreate(MPI_Comm,DA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetDim(DA,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetSizes(DA,PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreate1d(MPI_Comm,DAPeriodicType,PetscInt,PetscInt,PetscInt,const PetscInt[],DA *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],DA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],DA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetOptionsPrefix(DA,const char []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DAViewFromOptions(DA, const char []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetFromOptions(DA);
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
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetOwnershipRanges(DA,const PetscInt*[],const PetscInt*[],const PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreateGlobalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreateLocalVector(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACreateNaturalVector(DA,Vec *);
#define  DAGetLocalVector(da,v)      DMGetLocalVector((DM)da,v)
#define  DARestoreLocalVector(da,v)  DMRestoreLocalVector((DM)da,v)
#define  DAGetGlobalVector(da,v)     DMGetGlobalVector((DM)da,v)
#define  DARestoreGlobalVector(da,v) DMRestoreGlobalVector((DM)da,v)
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DALoad(PetscViewer,PetscInt,PetscInt,PetscInt,DA *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetCorners(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetGhostCorners(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetInfo(DA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,DAPeriodicType*,DAStencilType*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetProcessorSubset(DA,DADirection,PetscInt,MPI_Comm*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DARefine(DA,MPI_Comm,DA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACoarsen(DA,MPI_Comm,DA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DARefineHierarchy(DA,PetscInt,DA[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DACoarsenHierarchy(DA,PetscInt,DA[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGlobalToNaturalAllCreate(DA,VecScatter*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DANaturalAllToGlobalCreate(DA,VecScatter*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetGlobalIndices(DA,PetscInt*,PetscInt**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetISLocalToGlobalMapping(DA,ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetISLocalToGlobalMappingBlck(DA,ISLocalToGlobalMapping*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetScatter(DA,VecScatter*,VecScatter*,VecScatter*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetNeighbors(DA,const PetscMPIInt**);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetAO(DA,AO*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DASetCoordinates(DA,Vec); 
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetCoordinates(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetGhostedCoordinates(DA,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetCoordinateDA(DA,DA *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DASetUniformCoordinates(DA,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetBoundingBox(DA,PetscReal[],PetscReal[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetLocalBoundingBox(DA,PetscReal[],PetscReal[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DASetFieldName(DA,PetscInt,const char[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAGetFieldName(DA,PetscInt,char **);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetPeriodicity(DA, DAPeriodicType);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetDof(DA, int);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetStencilWidth(DA, int);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetVertexDivision(DA, const PetscInt[], const PetscInt[], const PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetNumProcs(DA, PetscInt, PetscInt, PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetStencilType(DA, DAStencilType);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAVecGetArray(DA,Vec,void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAVecRestoreArray(DA,Vec,void *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAVecGetArrayDOF(DA,Vec,void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DAVecRestoreArrayDOF(DA,Vec,void *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DASplitComm2d(MPI_Comm,PetscInt,PetscInt,PetscInt,MPI_Comm*);

/* Dynamic creation and loading functions */
extern PetscFList DAList;
extern PetscTruth DARegisterAllCalled;
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DASetType(DA, const DAType);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DAGetType(DA, const DAType *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DARegister(const char[],const char[],const char[],PetscErrorCode (*)(DA));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DARegisterAll(const char []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DARegisterDestroy(void);

/*MC
  DARegisterDynamic - Adds a new DA component implementation

  Synopsis:
  PetscErrorCode DARegisterDynamic(const char *name,const char *path,const char *func_name, PetscErrorCode (*create_func)(DA))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of routine to create method context
- create_func - The creation routine itself

  Notes:
  DARegisterDynamic() may be called multiple times to add several user-defined DAs

  If dynamic libraries are used, then the fourth input argument (routine_create) is ignored.

  Sample usage:
.vb
    DARegisterDynamic("my_da","/home/username/my_lib/lib/libO/solaris/libmy.a", "MyDACreate", MyDACreate);
.ve

  Then, your DA type can be chosen with the procedural interface via
.vb
    DACreate(MPI_Comm, DA *);
    DASetType(DA,"my_da_name");
.ve
   or at runtime via the option
.vb
    -da_type my_da_name
.ve

  Notes: $PETSC_ARCH occuring in pathname will be replaced with appropriate values.
         If your function is not being put into a shared library then use DARegister() instead
        
  Level: advanced

.keywords: DA, register
.seealso: DARegisterAll(), DARegisterDestroy(), DARegister()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define DARegisterDynamic(a,b,c,d) DARegister(a,b,c,0)
#else
#define DARegisterDynamic(a,b,c,d) DARegister(a,b,c,d)
#endif

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

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],SDA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],SDA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDACreate1d(MPI_Comm,DAPeriodicType,PetscInt,PetscInt,PetscInt,const PetscInt[],SDA*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDADestroy(SDA);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDALocalToLocalBegin(SDA,PetscScalar*,InsertMode,PetscScalar*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDALocalToLocalEnd(SDA,PetscScalar*,InsertMode,PetscScalar*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDAGetCorners(SDA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDAGetGhostCorners(SDA,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    SDAArrayView(SDA,PetscScalar*,PetscViewer);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    MatRegisterDAAD(void);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    MatCreateDAAD(DA,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT   MatCreateSeqUSFFT(Vec, DA,Mat*);

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
   
   Not Collective

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
   
   Not Collective

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
      DAVecGetArray(cda,vcoors,&coors);
      DAGetCorners(cda,&mstart,&nstart,0,&m,&n,0)
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
      DACoor3d ***coors;
      Vec      vcoors;
      DA       cda;     

      DAGetCoordinates(da,&vcoors); 
      DAGetCoordinateDA(da,&cda);
      DAVecGetArray(cda,vcoors,&coors);
      DAGetCorners(cda,&mstart,&nstart,&pstart,&m,&n,&p)
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
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetLocalJacobian(DA,DALocalFunction1*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalJacobian(DA,DALocalFunction1);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetLocalAdicFunction_Private(DA,DALocalFunction1);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MatSetDA(Mat,DA);

/*MC
       DASetLocalAdicFunction - Caches in a DA a local function computed by ADIC/ADIFOR

   Synopsis:
   PetscErrorCode DASetLocalAdicFunction(DA da,DALocalFunction1 ad_lf)
   
   Collective on DA

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

/*S
     DM - Abstract PETSc object that manages an abstract grid object
          
   Level: intermediate

  Concepts: grids, grid refinement

   Notes: The DA object and the DMComposite object are examples of DMs

          Though the DA objects require the petscsnes.h include files the DM library is
    NOT dependent on the SNES or KSP library. In fact, the KSP and SNES libraries depend on
    DM. (This is not great design, but not trivial to fix).

.seealso:  DMCompositeCreate(), DA, DMComposite
S*/
typedef struct _p_DM* DM;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMView(DM,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDestroy(DM);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCreateGlobalVector(DM,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCreateLocalVector(DM,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetLocalVector(DM,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMRestoreLocalVector(DM,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetGlobalVector(DM,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMRestoreGlobalVector(DM,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetColoring(DM,ISColoringType,const MatType,ISColoring*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetMatrix(DM, const MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetInterpolation(DM,DM,Mat*,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetInjection(DM,DM,VecScatter*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMRefine(DM,MPI_Comm,DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCoarsen(DM,MPI_Comm,DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMRefineHierarchy(DM,PetscInt,DM[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCoarsenHierarchy(DM,PetscInt,DM[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetInterpolationScale(DM,DM,Mat,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetAggregates(DM,DM,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGlobalToLocalBegin(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGlobalToLocalEnd(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMLocalToGlobal(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetElements(DM,PetscInt *,const PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMRestoreElements(DM,PetscInt *,const PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMFinalizePackage(void);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetColoring(DA,ISColoringType,const MatType,ISColoring *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetMatrix(DA, const MatType,Mat *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DASetGetMatrix(DA,PetscErrorCode (*)(DA, const MatType,Mat *));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetInterpolation(DA,DA,Mat*,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DAGetAggregates(DA,DA,Mat*);
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
     DMComposite - Abstract PETSc object that manages treating several distinct vectors as if they
        were one.   The DMComposite routines allow one to manage a nonlinear solver that works on a
        vector that consists of several distinct parts. This is mostly used for LNKS solvers, 
        that is design optimization problems that are written as a nonlinear system

   Level: beginner

  Concepts: multi-component, LNKS solvers

.seealso:  DMCompositeCreate(), DMCompositeDestroy(), DM
S*/
typedef struct _p_DMComposite* DMComposite;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeCreate(MPI_Comm,DMComposite*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeDestroy(DMComposite);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeView(DMComposite,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeAddArray(DMComposite,PetscMPIInt,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeAddDM(DMComposite,DM);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeSetCoupling(DMComposite,PetscErrorCode (*)(DMComposite,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeSetContext(DMComposite,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetContext(DMComposite,void**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeAddVecScatter(DMComposite,VecScatter);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeScatter(DMComposite,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGather(DMComposite,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetAccess(DMComposite,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetNumberDM(DMComposite,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeRestoreAccess(DMComposite,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetLocalVectors(DMComposite,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetEntries(DMComposite,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeRestoreLocalVectors(DMComposite,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeCreateGlobalVector(DMComposite,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeCreateLocalVector(DMComposite,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetLocalISs(DMComposite,IS*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetGlobalISs(DMComposite,IS*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeRefine(DMComposite,MPI_Comm,DMComposite*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetInterpolation(DMComposite,DMComposite,Mat*,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetMatrix(DMComposite,const MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetColoring(DMComposite,ISColoringType,const MatType,ISColoring*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGlobalToLocalBegin(DMComposite,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGlobalToLocalEnd(DMComposite,Vec,InsertMode,Vec);

/*S
     Slice - Abstract PETSc object that manages distributed field data for a simple unstructured matrix

   Level: beginner

  Concepts: distributed array

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), VecScatter, DACreate(), DMCompositeCreate(), DMComposite
S*/
typedef struct _p_Sliced* Sliced;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedView(Sliced,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedCreate(MPI_Comm,Sliced*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedDestroy(Sliced);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedCreateGlobalVector(Sliced,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedGetMatrix(Sliced, const MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedGetGlobalIndices(Sliced,PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedSetPreallocation(Sliced,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedSetBlockFills(Sliced,const PetscInt*,const PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  SlicedSetGhosts(Sliced,PetscInt,PetscInt,PetscInt,const PetscInt[]);


typedef struct NLF_DAAD* NLF;

#include <petscbag.h>

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabOpen(MPI_Comm, const char [], PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabDestroy(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabOutputBag(PetscViewer, const char [], PetscBag);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabOutputVec(PetscViewer, const char [], Vec);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMatlabOutputVecDA(PetscViewer, const char [], Vec, DA);


/*S
  ADDA - Abstract PETSc object that manages distributed field data for a single structured grid
         These are for any number of dimensions.

  Level: advanced. 

  Concepts: distributed array
.seealso: DA, DACreate(), ADDACreate()
S*/
typedef struct _p_ADDA* ADDA;

extern PetscCookie PETSCDM_DLLEXPORT ADDA_COOKIE;

PetscErrorCode PETSCDM_DLLEXPORT ADDACreate(MPI_Comm,PetscInt,PetscInt*,PetscInt*,PetscInt,PetscTruth*,ADDA*);
PetscErrorCode PETSCDM_DLLEXPORT ADDADestroy(ADDA);

/* DM interface functions */
PetscErrorCode PETSCDM_DLLEXPORT ADDAView(ADDA,PetscViewer);
PetscErrorCode PETSCDM_DLLEXPORT ADDACreateGlobalVector(ADDA,Vec*);
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetColoring(ADDA,ISColoringType,const MatType,ISColoring*);
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetMatrix(ADDA,const MatType, Mat*);
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetInterpolation(ADDA,ADDA,Mat*,Vec*);
PetscErrorCode PETSCDM_DLLEXPORT ADDARefine(ADDA, MPI_Comm,ADDA *);
PetscErrorCode PETSCDM_DLLEXPORT ADDACoarsen(ADDA, MPI_Comm, ADDA*);
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetInjection(ADDA, ADDA, VecScatter*);
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetAggregates(ADDA, ADDA, Mat *);

/* functions only supported by ADDA */
PetscErrorCode PETSCDM_DLLEXPORT ADDASetRefinement(ADDA, PetscInt *,PetscInt);
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetCorners(ADDA, PetscInt **, PetscInt **);
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetGhostCorners(ADDA, PetscInt **, PetscInt **);
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetMatrixNS(ADDA, ADDA, const MatType , Mat *);

/* functions to set values in vectors and matrices */
struct _ADDAIdx_s {
  PetscInt     *x;               /* the coordinates, user has to make sure it is the correct size! */
  PetscInt     d;                /* indexes the dof */
};
typedef struct _ADDAIdx_s ADDAIdx;

PetscErrorCode PETSCDM_DLLEXPORT ADDAMatSetValues(Mat, ADDA, PetscInt, const ADDAIdx[], ADDA, PetscInt,
						  const ADDAIdx[], const PetscScalar[], InsertMode);

PetscTruth ADDAHCiterStartup(const PetscInt, const PetscInt *const, const PetscInt *const, PetscInt *const);
PetscTruth ADDAHCiter(const PetscInt, const PetscInt *const, const PetscInt *const, PetscInt *const);

PETSC_EXTERN_CXX_END
#endif

/*
      Objects to manage the interactions between the mesh data structures and the algebraic objects
*/
#if !defined(__PETSCDA_H)
#define __PETSCDA_H
#include "petscvec.h"
#include "petscao.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMInitializePackage(const char[]);

/*S
     DM - Abstract PETSc object that manages an abstract grid object
          
   Level: intermediate

  Concepts: grids, grid refinement

   Notes: The DMDA object and the Composite object are examples of DMs

          Though the DMDA objects require the petscsnes.h include files the DM library is
    NOT dependent on the SNES or KSP library. In fact, the KSP and SNES libraries depend on
    DM. (This is not great design, but not trivial to fix).

.seealso:  DMCompositeCreate(), DMDACreate()
S*/
typedef struct _p_DM* DM;

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
    DMDAPeriodicType - Is the domain periodic in one or more directions

   Level: beginner

   DMDA_XYZGHOSTED means that ghost points are put around all the physical boundaries
   in the local representation of the Vec (i.e. DMDACreate/GetLocalVector().

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum { DMDA_NONPERIODIC,DMDA_XPERIODIC,DMDA_YPERIODIC,DMDA_XYPERIODIC,
               DMDA_XYZPERIODIC,DMDA_XZPERIODIC,DMDA_YZPERIODIC,DMDA_ZPERIODIC,DMDA_XYZGHOSTED} DMDAPeriodicType;
extern const char *DMDAPeriodicTypes[];

/*E
    DMDAInterpolationType - Defines the type of interpolation that will be returned by 
       DMGetInterpolation.

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGetInterpolation(), DMDASetInterpolationType(), DMDACreate()
E*/
typedef enum { DMDA_Q0, DMDA_Q1 } DMDAInterpolationType;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetInterpolationType(DM,DMDAInterpolationType);

/*E
    DMDAElementType - Defines the type of elements that will be returned by 
       DMGetElements()

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGetInterpolation(), DMDASetInterpolationType(), 
          DMDASetElementType(), DMGetElements(), DMRestoreElements(), DMDACreate()
E*/
typedef enum { DMDA_ELEMENT_P1, DMDA_ELEMENT_Q1 } DMDAElementType;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetElementType(DM,DMDAElementType);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetElementType(DM,DMDAElementType*);

#define DMDAXPeriodic(pt) ((pt)==DMDA_XPERIODIC||(pt)==DMDA_XYPERIODIC||(pt)==DMDA_XZPERIODIC||(pt)==DMDA_XYZPERIODIC)
#define DMDAYPeriodic(pt) ((pt)==DMDA_YPERIODIC||(pt)==DMDA_XYPERIODIC||(pt)==DMDA_YZPERIODIC||(pt)==DMDA_XYZPERIODIC)
#define DMDAZPeriodic(pt) ((pt)==DMDA_ZPERIODIC||(pt)==DMDA_XZPERIODIC||(pt)==DMDA_YZPERIODIC||(pt)==DMDA_XYZPERIODIC)

typedef enum { DMDA_X,DMDA_Y,DMDA_Z } DMDADirection;

extern PetscClassId PETSCDM_DLLEXPORT DM_CLASSID;

#define MATSEQUSFFT        "sequsfft"

EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDACreate(MPI_Comm,DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDASetDim(DM,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDASetSizes(DM,PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDACreate1d(MPI_Comm,DMDAPeriodicType,PetscInt,PetscInt,PetscInt,const PetscInt[],DM *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDACreate2d(MPI_Comm,DMDAPeriodicType,DMDAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDACreate3d(MPI_Comm,DMDAPeriodicType,DMDAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMSetOptionsPrefix(DM,const char []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMSetVecType(DM,const VecType);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGlobalToNaturalBegin(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGlobalToNaturalEnd(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDANaturalToGlobalBegin(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDANaturalToGlobalEnd(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDALocalToLocalBegin(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDALocalToLocalEnd(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDACreateNaturalVector(DM,Vec *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDALoad(PetscViewer,PetscInt,PetscInt,PetscInt,DM *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetCorners(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetGhostCorners(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetInfo(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,DMDAPeriodicType*,DMDAStencilType*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetProcessorSubset(DM,DMDADirection,PetscInt,MPI_Comm*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetProcessorSubsets(DM,DMDADirection,MPI_Comm*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGlobalToNaturalAllCreate(DM,VecScatter*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDANaturalAllToGlobalCreate(DM,VecScatter*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetGlobalIndices(DM,PetscInt*,PetscInt**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetISLocalToGlobalMapping(DM,ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetISLocalToGlobalMappingBlck(DM,ISLocalToGlobalMapping*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetScatter(DM,VecScatter*,VecScatter*,VecScatter*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetNeighbors(DM,const PetscMPIInt**);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetAO(DM,AO*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDASetCoordinates(DM,Vec); 
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetCoordinates(DM,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetGhostedCoordinates(DM,Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetCoordinateDA(DM,DM *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDASetUniformCoordinates(DM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetBoundingBox(DM,PetscReal[],PetscReal[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetLocalBoundingBox(DM,PetscReal[],PetscReal[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDASetFieldName(DM,PetscInt,const char[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAGetFieldName(DM,PetscInt,const char**);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDASetPeriodicity(DM, DMDAPeriodicType);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDASetDof(DM, int);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDASetStencilWidth(DM, PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDASetOwnershipRanges(DM,const PetscInt[],const PetscInt[],const PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDAGetOwnershipRanges(DM,const PetscInt**,const PetscInt**,const PetscInt**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDASetNumProcs(DM, PetscInt, PetscInt, PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMDASetStencilType(DM, DMDAStencilType);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAVecGetArray(DM,Vec,void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAVecRestoreArray(DM,Vec,void *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAVecGetArrayDOF(DM,Vec,void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDAVecRestoreArrayDOF(DM,Vec,void *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    DMDASplitComm2d(MPI_Comm,PetscInt,PetscInt,PetscInt,MPI_Comm*);

/*E
    DMType - String with the name of a PETSc DM or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:myveccreate()

   Level: beginner

.seealso: DMSetType(), DM
E*/

#define DMType char*
#define DMDA        "da"
#define DMADDA      "adda"
#define DMCOMPOSITE "composite"
#define DMSLICED    "sliced"

extern PetscFList DMList;
extern PetscBool  DMRegisterAllCalled;
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMCreate(MPI_Comm,DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMSetType(DM, const DMType);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMGetType(DM, const DMType *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMRegister(const char[],const char[],const char[],PetscErrorCode (*)(DM));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMRegisterAll(const char []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMRegisterDestroy(void);

/*MC
  DMRegisterDynamic - Adds a new DM component implementation

  Synopsis:
  PetscErrorCode DMRegisterDynamic(const char *name,const char *path,const char *func_name, PetscErrorCode (*create_func)(DM))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of routine to create method context
- create_func - The creation routine itself

  Notes:
  DMRegisterDynamic() may be called multiple times to add several user-defined DMs

  If dynamic libraries are used, then the fourth input argument (routine_create) is ignored.

  Sample usage:
.vb
    DMRegisterDynamic("my_da","/home/username/my_lib/lib/libO/solaris/libmy.a", "MyDMCreate", MyDMCreate);
.ve

  Then, your DM type can be chosen with the procedural interface via
.vb
    DMCreate(MPI_Comm, DM *);
    DMSetType(DM,"my_da_name");
.ve
   or at runtime via the option
.vb
    -da_type my_da_name
.ve

  Notes: $PETSC_ARCH occuring in pathname will be replaced with appropriate values.
         If your function is not being put into a shared library then use DMRegister() instead
        
  Level: advanced

.keywords: DM, register
.seealso: DMRegisterAll(), DMRegisterDestroy(), DMRegister()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define DMRegisterDynamic(a,b,c,d) DMRegister(a,b,c,0)
#else
#define DMRegisterDynamic(a,b,c,d) DMRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSCDM_DLLEXPORT    MatRegisterDAAD(void);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT    MatCreateDAAD(DM,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT   MatCreateSeqUSFFT(Vec, DM,Mat*);

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
  PetscInt       xs,ys,zs;    /* starting pointd of this processor, excluding ghosts */
  PetscInt       xm,ym,zm;    /* number of grid points on this processor, excluding ghosts */
  PetscInt       gxs,gys,gzs;    /* starting point of this processor including ghosts */
  PetscInt       gxm,gym,gzm;    /* number of grid points on this processor including ghosts */
  DMDAPeriodicType pt;
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
    
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetLocalInfo(DM,DMDALocalInfo*);
typedef PetscErrorCode (*DMDALocalFunction1)(DMDALocalInfo*,void*,void*,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAFormFunctionLocal(DM, DMDALocalFunction1, Vec, Vec, void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAFormFunctionLocalGhost(DM, DMDALocalFunction1, Vec, Vec, void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAFormJacobianLocal(DM, DMDALocalFunction1, Vec, Mat, void *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAFormFunction1(DM,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAFormFunction(DM,PetscErrorCode (*)(void),Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAFormFunctioni1(DM,PetscInt,Vec,PetscScalar*,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAFormFunctionib1(DM,PetscInt,Vec,PetscScalar*,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAComputeJacobian1WithAdic(DM,Vec,Mat,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAComputeJacobian1WithAdifor(DM,Vec,Mat,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAMultiplyByJacobian1WithAdic(DM,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAMultiplyByJacobian1WithAdifor(DM,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAMultiplyByJacobian1WithAD(DM,Vec,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAComputeJacobian1(DM,Vec,Mat,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetLocalFunction(DM,DMDALocalFunction1*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalFunction(DM,DMDALocalFunction1);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalFunctioni(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalFunctionib(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetLocalJacobian(DM,DMDALocalFunction1*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalJacobian(DM,DMDALocalFunction1);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalAdicFunction_Private(DM,DMDALocalFunction1);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  MatSetDA(Mat,DM);

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

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalAdicMFFunction_Private(DM,DMDALocalFunction1);
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunction(a,d) DMDASetLocalAdicMFFunction_Private(a,(DMDALocalFunction1)d)
#else
#  define DMDASetLocalAdicMFFunction(a,d) DMDASetLocalAdicMFFunction_Private(a,0)
#endif
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalAdicFunctioni_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicFunctioni(a,d) DMDASetLocalAdicFunctioni_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicFunctioni(a,d) DMDASetLocalAdicFunctioni_Private(a,0)
#endif
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalAdicMFFunctioni_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunctioni(a,d) DMDASetLocalAdicMFFunctioni_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicMFFunctioni(a,d) DMDASetLocalAdicMFFunctioni_Private(a,0)
#endif

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalAdicFunctionib_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicFunctionib(a,d) DMDASetLocalAdicFunctionib_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicFunctionib(a,d) DMDASetLocalAdicFunctionib_Private(a,0)
#endif
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetLocalAdicMFFunctionib_Private(DM,PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMDASetLocalAdicMFFunctionib(a,d) DMDASetLocalAdicMFFunctionib_Private(a,(PetscErrorCode (*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))d)
#else
#  define DMDASetLocalAdicMFFunctionib(a,d) DMDASetLocalAdicMFFunctionib_Private(a,0)
#endif

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAFormFunctioniTest1(DM,void*);

#include "petscmat.h"


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
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSetFromOptions(DM);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSetUp(DM);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetInterpolationScale(DM,DM,Mat,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetAggregates(DM,DM,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGlobalToLocalBegin(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGlobalToLocalEnd(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMLocalToGlobalBegin(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMLocalToGlobalEnd(DM,Vec,InsertMode,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetElements(DM,PetscInt *,const PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMRestoreElements(DM,PetscInt *,const PetscInt*[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSetContext(DM,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetContext(DM,void**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSetInitialGuess(DM,PetscErrorCode (*)(DM,Vec));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSetFunction(DM,PetscErrorCode (*)(DM,Vec,Vec));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSetJacobian(DM,PetscErrorCode (*)(DM,Vec,Mat,Mat,MatStructure *));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMHasInitialGuess(DM,PetscBool *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMHasFunction(DM,PetscBool *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMHasJacobian(DM,PetscBool *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMComputeInitialGuess(DM,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMComputeFunction(DM,Vec,Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMComputeJacobian(DM,Vec,Mat,Mat,MatStructure *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMComputeJacobianDefault(DM,Vec,Mat,Mat,MatStructure *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMFinalizePackage(void);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMGetMatrix(DM, const MatType,Mat *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetGetMatrix(DM,PetscErrorCode (*)(DM, const MatType,Mat *));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetBlockFills(DM,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetMatPreallocateOnly(DM,PetscBool );
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDASetRefinementFactor(DM,PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetRefinementFactor(DM,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetAdicArray(DM,PetscBool ,void*,void*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDARestoreAdicArray(DM,PetscBool ,void*,void*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetAdicMFArray(DM,PetscBool ,void*,void*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetAdicMFArray4(DM,PetscBool ,void*,void*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetAdicMFArray9(DM,PetscBool ,void*,void*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetAdicMFArrayb(DM,PetscBool ,void*,void*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDARestoreAdicMFArray(DM,PetscBool ,void*,void*,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDAGetArray(DM,PetscBool ,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDARestoreArray(DM,PetscBool ,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  ad_DAGetArray(DM,PetscBool ,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  ad_DARestoreArray(DM,PetscBool ,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  admf_DAGetArray(DM,PetscBool ,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  admf_DARestoreArray(DM,PetscBool ,void*);

#include "petscpf.h"
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMDACreatePF(DM,PF*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeCreate(MPI_Comm,DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeAddArray(DM,PetscMPIInt,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeAddDM(DM,DM);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeSetCoupling(DM,PetscErrorCode (*)(DM,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeSetContext(DM,void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetContext(DM,void**);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeAddVecScatter(DM,VecScatter);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeScatter(DM,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGather(DM,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetAccess(DM,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetNumberDM(DM,PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeRestoreAccess(DM,Vec,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetLocalVectors(DM,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetEntries(DM,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeRestoreLocalVectors(DM,...);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetLocalISs(DM,IS*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMCompositeGetGlobalISs(DM,IS*[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSlicedCreate(MPI_Comm,DM*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSlicedGetGlobalIndices(DM,PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSlicedSetPreallocation(DM,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSlicedSetBlockFills(DM,const PetscInt*,const PetscInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT  DMSlicedSetGhosts(DM,PetscInt,PetscInt,PetscInt,const PetscInt[]);


typedef struct NLF_DAAD* NLF;

#include <petscbag.h>

EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerBinaryMatlabOpen(MPI_Comm, const char [], PetscViewer*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerBinaryMatlabDestroy(PetscViewer);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerBinaryMatlabOutputBag(PetscViewer, const char [], PetscBag);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerBinaryMatlabOutputVec(PetscViewer, const char [], Vec);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerBinaryMatlabOutputVecDA(PetscViewer, const char [], Vec, DM);


PetscErrorCode PETSCDM_DLLEXPORT DMADDACreate(MPI_Comm,PetscInt,PetscInt*,PetscInt*,PetscInt,PetscBool *,DM*);
PetscErrorCode PETSCDM_DLLEXPORT DMADDASetParameters(DM,PetscInt,PetscInt*,PetscInt*,PetscInt,PetscBool*);
PetscErrorCode PETSCDM_DLLEXPORT DMADDASetRefinement(DM, PetscInt *,PetscInt);
PetscErrorCode PETSCDM_DLLEXPORT DMADDAGetCorners(DM, PetscInt **, PetscInt **);
PetscErrorCode PETSCDM_DLLEXPORT DMADDAGetGhostCorners(DM, PetscInt **, PetscInt **);
PetscErrorCode PETSCDM_DLLEXPORT DMADDAGetMatrixNS(DM, DM, const MatType , Mat *);

/* functions to set values in vectors and matrices */
struct _ADDAIdx_s {
  PetscInt     *x;               /* the coordinates, user has to make sure it is the correct size! */
  PetscInt     d;                /* indexes the dof */
};
typedef struct _ADDAIdx_s ADDAIdx;

PetscErrorCode PETSCDM_DLLEXPORT DMADDAMatSetValues(Mat, DM, PetscInt, const ADDAIdx[], DM, PetscInt, const ADDAIdx[], const PetscScalar[], InsertMode);
PetscBool  ADDAHCiterStartup(const PetscInt, const PetscInt *const, const PetscInt *const, PetscInt *const);
PetscBool  ADDAHCiter(const PetscInt, const PetscInt *const, const PetscInt *const, PetscInt *const);

PETSC_EXTERN_CXX_END
#endif

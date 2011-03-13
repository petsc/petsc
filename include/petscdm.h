/*
      Objects to manage the interactions between the mesh data structures and the algebraic objects
*/
#if !defined(__PETSCDA_H)
#define __PETSCDA_H
#include "petscmat.h"
#include "petscao.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode  DMInitializePackage(const char[]);
/*S
     DM - Abstract PETSc object that manages an abstract grid object

   Level: intermediate

  Concepts: grids, grid refinement

   Notes: The DMDACreate() based object and the DMCompositeCreate() based object are examples of DMs

          Though the DM objects require the petscsnes.h include files the DM library is
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
    DMDABoundaryType - Describes the choice for fill of ghost cells on domain boundaries.

   Level: beginner

   A boundary may be of type DMDA_BOUNDARY_NONE (no ghost nodes), DMDA_BOUNDARY_GHOST (ghost nodes 
   exist but aren't filled), DMDA_BOUNDARY_MIRROR (not yet implemented), or DMDA_BOUNDARY_PERIODIC
   (ghost nodes filled by the opposite edge of the domain).

.seealso: DMDASetBoundaryType(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum { DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_MIRROR, DMDA_BOUNDARY_PERIODIC } DMDABoundaryType;

extern const char *DMDABoundaryTypes[];

/*E
    DMDAInterpolationType - Defines the type of interpolation that will be returned by 
       DMGetInterpolation.

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGetInterpolation(), DMDASetInterpolationType(), DMDACreate()
E*/
typedef enum { DMDA_Q0, DMDA_Q1 } DMDAInterpolationType;

extern PetscErrorCode   DMDASetInterpolationType(DM,DMDAInterpolationType);

/*E
    DMDAElementType - Defines the type of elements that will be returned by 
       DMGetElements()

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGetInterpolation(), DMDASetInterpolationType(), 
          DMDASetElementType(), DMGetElements(), DMRestoreElements(), DMDACreate()
E*/
typedef enum { DMDA_ELEMENT_P1, DMDA_ELEMENT_Q1 } DMDAElementType;

extern PetscErrorCode   DMDASetElementType(DM,DMDAElementType);
extern PetscErrorCode   DMDAGetElementType(DM,DMDAElementType*);

typedef enum { DMDA_X,DMDA_Y,DMDA_Z } DMDADirection;

extern PetscClassId  DM_CLASSID;

#define MATSEQUSFFT        "sequsfft"

extern PetscErrorCode  DMDACreate(MPI_Comm,DM*);
extern PetscErrorCode  DMDASetDim(DM,PetscInt);
extern PetscErrorCode  DMDASetSizes(DM,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode     DMDACreate1d(MPI_Comm,DMDABoundaryType,PetscInt,PetscInt,PetscInt,const PetscInt[],DM *);
extern PetscErrorCode     DMDACreate2d(MPI_Comm,DMDABoundaryType,DMDABoundaryType,DMDAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],DM*);
extern PetscErrorCode     DMDACreate3d(MPI_Comm,DMDABoundaryType,DMDABoundaryType,DMDABoundaryType,DMDAStencilType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],DM*);
extern PetscErrorCode  DMSetOptionsPrefix(DM,const char []);
extern PetscErrorCode  DMSetVecType(DM,const VecType);

extern PetscErrorCode     DMDAGlobalToNaturalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDAGlobalToNaturalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDANaturalToGlobalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDANaturalToGlobalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDALocalToLocalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDALocalToLocalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode     DMDACreateNaturalVector(DM,Vec *);

extern PetscErrorCode     DMDALoad(PetscViewer,PetscInt,PetscInt,PetscInt,DM *);
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
extern PetscErrorCode     DMDASetFieldName(DM,PetscInt,const char[]);
extern PetscErrorCode     DMDAGetFieldName(DM,PetscInt,const char**);

extern PetscErrorCode  DMDASetBoundaryType(DM,DMDABoundaryType,DMDABoundaryType,DMDABoundaryType);
extern PetscErrorCode  DMDASetDof(DM, int);
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
#define DMMESH      "mesh"
#define DMCARTESIAN "cartesian"

extern PetscFList DMList;
extern PetscBool  DMRegisterAllCalled;
extern PetscErrorCode  DMCreate(MPI_Comm,DM*);
extern PetscErrorCode  DMSetType(DM, const DMType);
extern PetscErrorCode  DMGetType(DM, const DMType *);
extern PetscErrorCode  DMRegister(const char[],const char[],const char[],PetscErrorCode (*)(DM));
extern PetscErrorCode  DMRegisterAll(const char []);
extern PetscErrorCode  DMRegisterDestroy(void);

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

extern PetscErrorCode     MatRegisterDAAD(void);
extern PetscErrorCode     MatCreateDAAD(DM,Mat*);
extern PetscErrorCode    MatCreateSeqUSFFT(Vec, DM,Mat*);

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
extern PetscErrorCode   DMDAFormFunctionLocal(DM, DMDALocalFunction1, Vec, Vec, void *);
extern PetscErrorCode   DMDAFormFunctionLocalGhost(DM, DMDALocalFunction1, Vec, Vec, void *);
extern PetscErrorCode   DMDAFormJacobianLocal(DM, DMDALocalFunction1, Vec, Mat, void *);
extern PetscErrorCode   DMDAFormFunction1(DM,Vec,Vec,void*);
extern PetscErrorCode   DMDAFormFunction(DM,PetscErrorCode (*)(void),Vec,Vec,void*);
extern PetscErrorCode   DMDAFormFunctioni1(DM,PetscInt,Vec,PetscScalar*,void*);
extern PetscErrorCode   DMDAFormFunctionib1(DM,PetscInt,Vec,PetscScalar*,void*);
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

extern PetscErrorCode   MatSetDA(Mat,DM);

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

extern PetscErrorCode   DMDAFormFunctioniTest1(DM,void*);

extern PetscErrorCode   DMView(DM,PetscViewer);
extern PetscErrorCode   DMDestroy(DM);
extern PetscErrorCode   DMCreateGlobalVector(DM,Vec*);
extern PetscErrorCode   DMCreateLocalVector(DM,Vec*);
extern PetscErrorCode   DMGetLocalVector(DM,Vec *);
extern PetscErrorCode   DMRestoreLocalVector(DM,Vec *);
extern PetscErrorCode   DMGetGlobalVector(DM,Vec *);
extern PetscErrorCode   DMRestoreGlobalVector(DM,Vec *);
extern PetscErrorCode   DMGetLocalToGlobalMapping(DM,ISLocalToGlobalMapping*);
extern PetscErrorCode   DMGetLocalToGlobalMappingBlock(DM,ISLocalToGlobalMapping*);
extern PetscErrorCode   DMGetBlockSize(DM,PetscInt*);
extern PetscErrorCode   DMGetColoring(DM,ISColoringType,const MatType,ISColoring*);
extern PetscErrorCode   DMGetMatrix(DM, const MatType,Mat*);
extern PetscErrorCode   DMGetInterpolation(DM,DM,Mat*,Vec*);
extern PetscErrorCode   DMGetInjection(DM,DM,VecScatter*);
extern PetscErrorCode   DMRefine(DM,MPI_Comm,DM*);
extern PetscErrorCode   DMCoarsen(DM,MPI_Comm,DM*);
extern PetscErrorCode   DMRefineHierarchy(DM,PetscInt,DM[]);
extern PetscErrorCode   DMCoarsenHierarchy(DM,PetscInt,DM[]);
extern PetscErrorCode   DMSetFromOptions(DM);
extern PetscErrorCode   DMSetUp(DM);
extern PetscErrorCode   DMGetInterpolationScale(DM,DM,Mat,Vec*);
extern PetscErrorCode   DMGetAggregates(DM,DM,Mat*);
extern PetscErrorCode   DMGlobalToLocalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode   DMGlobalToLocalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode   DMLocalToGlobalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode   DMLocalToGlobalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode   DMGetElements(DM,PetscInt *,PetscInt *,const PetscInt*[]);
extern PetscErrorCode   DMRestoreElements(DM,PetscInt *,PetscInt *,const PetscInt*[]);

extern PetscErrorCode   DMSetContext(DM,void*);
extern PetscErrorCode   DMGetContext(DM,void**);
extern PetscErrorCode   DMSetInitialGuess(DM,PetscErrorCode (*)(DM,Vec));
extern PetscErrorCode   DMSetFunction(DM,PetscErrorCode (*)(DM,Vec,Vec));
extern PetscErrorCode   DMSetJacobian(DM,PetscErrorCode (*)(DM,Vec,Mat,Mat,MatStructure *));
extern PetscErrorCode   DMHasInitialGuess(DM,PetscBool *);
extern PetscErrorCode   DMHasFunction(DM,PetscBool *);
extern PetscErrorCode   DMHasJacobian(DM,PetscBool *);
extern PetscErrorCode   DMComputeInitialGuess(DM,Vec);
extern PetscErrorCode   DMComputeFunction(DM,Vec,Vec);
extern PetscErrorCode   DMComputeJacobian(DM,Vec,Mat,Mat,MatStructure *);
extern PetscErrorCode   DMComputeJacobianDefault(DM,Vec,Mat,Mat,MatStructure *);
extern PetscErrorCode   DMFinalizePackage(void);

extern PetscErrorCode   DMDASetGetMatrix(DM,PetscErrorCode (*)(DM, const MatType,Mat *));
extern PetscErrorCode   DMDASetBlockFills(DM,PetscInt*,PetscInt*);
extern PetscErrorCode   DMDASetMatPreallocateOnly(DM,PetscBool );
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

#include "petscpf.h"
extern PetscErrorCode   DMDACreatePF(DM,PF*);

extern PetscErrorCode   DMCompositeCreate(MPI_Comm,DM*);
extern PetscErrorCode   DMCompositeAddArray(DM,PetscMPIInt,PetscInt);
extern PetscErrorCode   DMCompositeAddDM(DM,DM);
extern PetscErrorCode   DMCompositeSetCoupling(DM,PetscErrorCode (*)(DM,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt));
extern PetscErrorCode   DMCompositeSetContext(DM,void*);
extern PetscErrorCode   DMCompositeGetContext(DM,void**);
extern PetscErrorCode   DMCompositeAddVecScatter(DM,VecScatter);
extern PetscErrorCode   DMCompositeScatter(DM,Vec,...);
extern PetscErrorCode   DMCompositeGather(DM,Vec,InsertMode,...);
extern PetscErrorCode   DMCompositeGetAccess(DM,Vec,...);
extern PetscErrorCode   DMCompositeGetNumberDM(DM,PetscInt*);
extern PetscErrorCode   DMCompositeRestoreAccess(DM,Vec,...);
extern PetscErrorCode   DMCompositeGetLocalVectors(DM,...);
extern PetscErrorCode   DMCompositeGetEntries(DM,...);
extern PetscErrorCode   DMCompositeRestoreLocalVectors(DM,...);
extern PetscErrorCode   DMCompositeGetGlobalISs(DM,IS*[]);
extern PetscErrorCode   DMCompositeGetLocalISs(DM,IS**);
extern PetscErrorCode   DMCompositeGetISLocalToGlobalMappings(DM,ISLocalToGlobalMapping**);

extern PetscErrorCode   DMSlicedCreate(MPI_Comm,DM*);
extern PetscErrorCode   DMSlicedGetGlobalIndices(DM,PetscInt*[]);
extern PetscErrorCode   DMSlicedSetPreallocation(DM,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
extern PetscErrorCode   DMSlicedSetBlockFills(DM,const PetscInt*,const PetscInt*);
extern PetscErrorCode   DMSlicedSetGhosts(DM,PetscInt,PetscInt,PetscInt,const PetscInt[]);


typedef struct NLF_DAAD* NLF;

#include <petscbag.h>

extern PetscErrorCode  PetscViewerBinaryMatlabOpen(MPI_Comm, const char [], PetscViewer*);
extern PetscErrorCode  PetscViewerBinaryMatlabDestroy(PetscViewer);
extern PetscErrorCode  PetscViewerBinaryMatlabOutputBag(PetscViewer, const char [], PetscBag);
extern PetscErrorCode  PetscViewerBinaryMatlabOutputVec(PetscViewer, const char [], Vec);
extern PetscErrorCode  PetscViewerBinaryMatlabOutputVecDA(PetscViewer, const char [], Vec, DM);


PetscErrorCode  DMADDACreate(MPI_Comm,PetscInt,PetscInt*,PetscInt*,PetscInt,PetscBool *,DM*);
PetscErrorCode  DMADDASetParameters(DM,PetscInt,PetscInt*,PetscInt*,PetscInt,PetscBool*);
PetscErrorCode  DMADDASetRefinement(DM, PetscInt *,PetscInt);
PetscErrorCode  DMADDAGetCorners(DM, PetscInt **, PetscInt **);
PetscErrorCode  DMADDAGetGhostCorners(DM, PetscInt **, PetscInt **);
PetscErrorCode  DMADDAGetMatrixNS(DM, DM, const MatType , Mat *);

/* functions to set values in vectors and matrices */
struct _ADDAIdx_s {
  PetscInt     *x;               /* the coordinates, user has to make sure it is the correct size! */
  PetscInt     d;                /* indexes the dof */
};
typedef struct _ADDAIdx_s ADDAIdx;

PetscErrorCode  DMADDAMatSetValues(Mat, DM, PetscInt, const ADDAIdx[], DM, PetscInt, const ADDAIdx[], const PetscScalar[], InsertMode);
PetscBool  ADDAHCiterStartup(const PetscInt, const PetscInt *const, const PetscInt *const, PetscInt *const);
PetscBool  ADDAHCiter(const PetscInt, const PetscInt *const, const PetscInt *const, PetscInt *const);

/* Mesh Section */
#if defined(PETSC_HAVE_SIEVE) && defined(__cplusplus)

#include <sieve/Mesh.hh>
#include <sieve/CartesianSieve.hh>
#include <sieve/Distribution.hh>
#include <sieve/Generator.hh>

extern PetscErrorCode DMMeshCreate(MPI_Comm, DM*);
extern PetscErrorCode DMMeshGetMesh(DM, ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode DMMeshSetMesh(DM, const ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode DMMeshGetGlobalScatter(DM, VecScatter *);
extern PetscErrorCode DMMeshFinalize();

extern PetscErrorCode DMMeshDistribute(DM, const char[], DM*);
extern PetscErrorCode DMMeshDistributeByFace(DM, const char[], DM*);
extern PetscErrorCode DMMeshGenerate(DM, PetscBool , DM *);
extern PetscErrorCode DMMeshRefine(DM, double, PetscBool , DM*);
extern PetscErrorCode DMMeshLoad(PetscViewer, DM);
extern PetscErrorCode DMMeshGetMaximumDegree(DM, PetscInt *);

extern PetscErrorCode DMMeshGetLabelSize(DM, const char[], PetscInt *);
extern PetscErrorCode DMMeshGetLabelIds(DM, const char[], PetscInt *);
extern PetscErrorCode DMMeshGetStratumSize(DM, const char [], PetscInt, PetscInt *);
extern PetscErrorCode DMMeshGetStratum(DM, const char [], PetscInt, PetscInt *);

extern PetscErrorCode DMCartesianCreate(MPI_Comm, DM *);
extern PetscErrorCode DMMeshCartesianGetMesh(DM, ALE::Obj<ALE::CartesianMesh>&);
extern PetscErrorCode DMMeshCartesianSetMesh(DM, const ALE::Obj<ALE::CartesianMesh>&);

extern PetscErrorCode DMMeshGetCoordinates(DM, PetscBool , PetscInt *, PetscInt *, PetscReal *[]);
extern PetscErrorCode DMMeshGetElements(DM, PetscBool , PetscInt *, PetscInt *, PetscInt *[]);
extern PetscErrorCode DMMeshGetCone(DM, PetscInt, PetscInt *, PetscInt *[]);

extern PetscErrorCode restrictVector(Vec, Vec, InsertMode);
extern PetscErrorCode assembleVectorComplete(Vec, Vec, InsertMode);
extern PetscErrorCode assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode updateOperator(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);
extern PetscErrorCode updateOperatorGeneral(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);

/*S
  SectionReal - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionRealCreate(), SectionRealDestroy(), Mesh, DMMeshCreate()
S*/
typedef struct _p_SectionReal* SectionReal;

/* Logging support */
extern PetscClassId  SECTIONREAL_CLASSID;

extern PetscErrorCode  SectionRealCreate(MPI_Comm,SectionReal*);
extern PetscErrorCode  SectionRealDestroy(SectionReal);
extern PetscErrorCode  SectionRealView(SectionReal,PetscViewer);
extern PetscErrorCode  SectionRealDuplicate(SectionReal,SectionReal*);

extern PetscErrorCode  SectionRealGetSection(SectionReal,ALE::Obj<PETSC_MESH_TYPE::real_section_type>&);
extern PetscErrorCode  SectionRealSetSection(SectionReal,const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&);
extern PetscErrorCode  SectionRealGetBundle(SectionReal,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode  SectionRealSetBundle(SectionReal,const ALE::Obj<PETSC_MESH_TYPE>&);

extern PetscErrorCode  SectionRealDistribute(SectionReal, DM, SectionReal *);
extern PetscErrorCode  SectionRealRestrict(SectionReal, PetscInt, PetscScalar *[]);
extern PetscErrorCode  SectionRealUpdate(SectionReal, PetscInt, const PetscScalar [], InsertMode);
extern PetscErrorCode  SectionRealZero(SectionReal);
extern PetscErrorCode  SectionRealCreateLocalVector(SectionReal, Vec*);
extern PetscErrorCode  SectionRealAddSpace(SectionReal);
extern PetscErrorCode  SectionRealGetFibration(SectionReal, const PetscInt, SectionReal *);
extern PetscErrorCode  SectionRealToVec(SectionReal, DM, ScatterMode, Vec);
extern PetscErrorCode  SectionRealToVec(SectionReal, VecScatter, ScatterMode, Vec);
extern PetscErrorCode  SectionRealNorm(SectionReal, DM, NormType, PetscReal *);
extern PetscErrorCode  SectionRealAXPY(SectionReal, DM, PetscScalar, SectionReal);
extern PetscErrorCode  SectionRealComplete(SectionReal);
extern PetscErrorCode  SectionRealSet(SectionReal, PetscReal);
extern PetscErrorCode  SectionRealGetFiberDimension(SectionReal, PetscInt, PetscInt*);
extern PetscErrorCode  SectionRealSetFiberDimension(SectionReal, PetscInt, const PetscInt);
extern PetscErrorCode  SectionRealSetFiberDimensionField(SectionReal, PetscInt, const PetscInt, const PetscInt);
extern PetscErrorCode  SectionRealGetSize(SectionReal, PetscInt *);
extern PetscErrorCode  SectionRealAllocate(SectionReal);
extern PetscErrorCode  SectionRealClear(SectionReal);

extern PetscErrorCode  SectionRealRestrictClosure(SectionReal, DM, PetscInt, PetscInt, PetscScalar []);
extern PetscErrorCode  SectionRealRestrictClosure(SectionReal, DM, PetscInt, const PetscScalar *[]);
extern PetscErrorCode  SectionRealUpdateClosure(SectionReal, DM, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode DMMeshHasSectionReal(DM, const char [], PetscBool  *);
extern PetscErrorCode DMMeshGetSectionReal(DM, const char [], SectionReal *);
extern PetscErrorCode DMMeshSetSectionReal(DM, SectionReal);
extern PetscErrorCode DMMeshCreateMatrix(DM, SectionReal, MatType, Mat *);
extern PetscErrorCode DMMeshCreateVector(DM, SectionReal, Vec *);
extern PetscErrorCode DMMeshCreateGlobalScatter(DM, SectionReal, VecScatter *);
extern PetscErrorCode assembleVector(Vec, DM, SectionReal, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode assembleMatrix(Mat, DM, SectionReal, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode DMMeshCreateGlobalRealVector(DM, SectionReal, Vec *);
extern PetscErrorCode DMMeshGetGlobalScatter(DM,VecScatter *);
extern PetscErrorCode DMMeshCreateGlobalScatter(DM,SectionReal,VecScatter *);
extern PetscErrorCode DMMeshGetLocalFunction(DM, PetscErrorCode (**)(DM, SectionReal, SectionReal, void*));
extern PetscErrorCode DMMeshSetLocalFunction(DM, PetscErrorCode (*)(DM, SectionReal, SectionReal, void*));
extern PetscErrorCode DMMeshGetLocalJacobian(DM, PetscErrorCode (**)(DM, SectionReal, Mat, void*));
extern PetscErrorCode DMMeshSetLocalJacobian(DM, PetscErrorCode (*)(DM, SectionReal, Mat, void*));
extern PetscErrorCode DMMeshFormFunction(DM, SectionReal, SectionReal, void*);
extern PetscErrorCode DMMeshFormJacobian(DM, SectionReal, Mat, void*);
extern PetscErrorCode DMMeshInterpolatePoints(DM, SectionReal, int, double *, double **);

/*S
  SectionInt - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionIntCreate(), SectionIntDestroy(), DM, DMMeshCreate()
S*/
typedef struct _p_SectionInt* SectionInt;

/* Logging support */
extern PetscClassId  SECTIONINT_CLASSID;

extern PetscErrorCode  SectionIntCreate(MPI_Comm,SectionInt*);
extern PetscErrorCode  SectionIntDestroy(SectionInt);
extern PetscErrorCode  SectionIntView(SectionInt,PetscViewer);

extern PetscErrorCode  SectionIntGetSection(SectionInt,ALE::Obj<PETSC_MESH_TYPE::int_section_type>&);
extern PetscErrorCode  SectionIntSetSection(SectionInt,const ALE::Obj<PETSC_MESH_TYPE::int_section_type>&);
extern PetscErrorCode  SectionIntGetBundle(SectionInt,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode  SectionIntSetBundle(SectionInt,const ALE::Obj<PETSC_MESH_TYPE>&);

extern PetscErrorCode  SectionIntDistribute(SectionInt, DM, SectionInt *);
extern PetscErrorCode  SectionIntRestrict(SectionInt, PetscInt, PetscInt *[]);
extern PetscErrorCode  SectionIntUpdate(SectionInt, PetscInt, const PetscInt [], InsertMode);
extern PetscErrorCode  SectionIntZero(SectionInt);
extern PetscErrorCode  SectionIntComplete(SectionInt);
extern PetscErrorCode  SectionIntGetFiberDimension(SectionInt, PetscInt, PetscInt*);
extern PetscErrorCode  SectionIntSetFiberDimension(SectionInt, PetscInt, const PetscInt);
extern PetscErrorCode  SectionIntSetFiberDimensionField(SectionInt, PetscInt, const PetscInt, const PetscInt);
extern PetscErrorCode  SectionIntGetSize(SectionInt, PetscInt *);
extern PetscErrorCode  SectionIntAllocate(SectionInt);
extern PetscErrorCode  SectionIntClear(SectionInt);

extern PetscErrorCode  SectionIntAddSpace(SectionInt);
extern PetscErrorCode  SectionIntGetFibration(SectionInt, const PetscInt, SectionInt *);
extern PetscErrorCode  SectionIntSet(SectionInt, PetscInt);

extern PetscErrorCode  SectionIntRestrictClosure(SectionInt, DM, PetscInt, PetscInt, PetscInt []);
extern PetscErrorCode  SectionIntUpdateClosure(SectionInt, DM, PetscInt, PetscInt [], InsertMode);

extern PetscErrorCode  DMMeshHasSectionInt(DM, const char [], PetscBool  *);
extern PetscErrorCode  DMMeshGetSectionInt(DM, const char [], SectionInt *);
extern PetscErrorCode  DMMeshSetSectionInt(DM, SectionInt);

typedef PetscErrorCode (*DMMeshLocalFunction1)(DM,SectionReal,SectionReal,void*);
typedef PetscErrorCode (*DMMeshLocalJacobian1)(DM,SectionReal,Mat,void*);

/* Misc Mesh functions*/
extern PetscErrorCode DMMeshSetMaxDof(DM, PetscInt);

/* Helper functions for simple distributions */
extern PetscErrorCode DMMeshGetVertexMatrix(DM, MatType, Mat *);
extern PetscErrorCode DMMeshGetVertexSectionReal(DM, const char[], PetscInt, SectionReal *);
PetscPolymorphicSubroutine(DMMeshGetVertexSectionReal,(DM dm, PetscInt fiberDim, SectionReal *section),(dm,"default",fiberDim,section))
extern PetscErrorCode DMMeshGetVertexSectionInt(DM, const char[], PetscInt, SectionInt *);
PetscPolymorphicSubroutine(DMMeshGetVertexSectionInt,(DM dm, PetscInt fiberDim, SectionInt *section),(dm,"default",fiberDim,section))
extern PetscErrorCode DMMeshGetCellMatrix(DM, MatType, Mat *);
extern PetscErrorCode DMMeshGetCellSectionReal(DM, const char[], PetscInt, SectionReal *);
PetscPolymorphicSubroutine(DMMeshGetCellSectionReal,(DM dm, PetscInt fiberDim, SectionReal *section),(dm,"default",fiberDim,section))
extern PetscErrorCode DMMeshGetCellSectionInt(DM, const char[], PetscInt, SectionInt *);
PetscPolymorphicSubroutine(DMMeshGetCellSectionInt,(DM dm, PetscInt fiberDim, SectionInt *section),(dm,"default",fiberDim,section))

/* Support for various mesh formats */
extern PetscErrorCode DMMeshCreateExodus(MPI_Comm, const char [], DM *);
extern PetscErrorCode DMMeshExodusGetInfo(DM, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);

extern PetscErrorCode DMWriteVTKHeader(PetscViewer);
extern PetscErrorCode DMWriteVTKVertices(DM, PetscViewer);
extern PetscErrorCode DMWriteVTKElements(DM, PetscViewer);
extern PetscErrorCode DMWritePCICEVertices(DM, PetscViewer);
extern PetscErrorCode DMWritePCICEElements(DM, PetscViewer);
extern PetscErrorCode DMWritePyLithVertices(DM, PetscViewer);
extern PetscErrorCode DMWritePyLithElements(DM, SectionReal, PetscViewer);
extern PetscErrorCode DMWritePyLithVerticesLocal(DM, PetscViewer);
extern PetscErrorCode WDMritePyLithElementsLocal(DM, SectionReal, PetscViewer);

typedef struct {
  int           numQuadPoints, numBasisFuncs;
  const double *quadPoints, *quadWeights, *basis, *basisDer;
} PetscQuadrature;

#endif /* Mesh section */

PETSC_EXTERN_CXX_END
#endif


/*
    Defines the basic header of all PETSc objects.
*/

#if !defined(_PETSCHEAD_H)
#define _PETSCHEAD_H
#include <petscsys.h>

/*
   All major PETSc data structures have a common core; this is defined
   below by PETSCHEADER.

   PetscHeaderCreate() should be used whenever creating a PETSc structure.
*/

/*
   PetscOps: structure of core operations that all PETSc objects support.

      getcomm()         - Gets the object's communicator.
      view()            - Is the routine for viewing the entire PETSc object; for
                          example, MatView() is the general matrix viewing routine.
                          This is used by PetscObjectView((PetscObject)obj) to allow
                          viewing any PETSc object.
      destroy()         - Is the routine for destroying the entire PETSc object;
                          for example,MatDestroy() is the general matrix
                          destruction routine.
                          This is used by PetscObjectDestroy((PetscObject*)&obj) to allow
                          destroying any PETSc object.
      compose()         - Associates a PETSc object with another PETSc object with a name
      query()           - Returns a different PETSc object that has been associated
                          with the first object using a name.
      composefunction() - Attaches an a function to a PETSc object with a name.
      queryfunction()   - Requests a registered function that has been attached to a PETSc object.
*/

typedef struct {
   PetscErrorCode (*getcomm)(PetscObject,MPI_Comm *);
   PetscErrorCode (*view)(PetscObject,PetscViewer);
   PetscErrorCode (*destroy)(PetscObject*);
   PetscErrorCode (*compose)(PetscObject,const char[],PetscObject);
   PetscErrorCode (*query)(PetscObject,const char[],PetscObject *);
   PetscErrorCode (*composefunction)(PetscObject,const char[],void (*)(void));
   PetscErrorCode (*queryfunction)(PetscObject,const char[],void (**)(void));
} PetscOps;

typedef enum {PETSC_FORTRAN_CALLBACK_CLASS,PETSC_FORTRAN_CALLBACK_SUBTYPE,PETSC_FORTRAN_CALLBACK_MAXTYPE} PetscFortranCallbackType;
typedef int PetscFortranCallbackId;
#define PETSC_SMALLEST_FORTRAN_CALLBACK ((PetscFortranCallbackId)1000)
PETSC_EXTERN PetscErrorCode PetscFortranCallbackRegister(PetscClassId,const char*,PetscFortranCallbackId*);
PETSC_EXTERN PetscErrorCode PetscFortranCallbackGetSizes(PetscClassId,PetscInt*,PetscInt*);

typedef struct {
  void (*func)(void);
  void *ctx;
} PetscFortranCallback;

/*
   All PETSc objects begin with the fields defined in PETSCHEADER.
   The PetscObject is a way of examining these fields regardless of
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
#define PETSC_MAX_OPTIONS_HANDLER 5
typedef struct _p_PetscObject {
  PetscClassId         classid;
  PetscOps             *bops;
  MPI_Comm             comm;
  PetscInt             type;
  PetscLogDouble       flops,time,mem,memchildren;
  PetscObjectId        id;
  PetscInt             refct;
  PetscMPIInt          tag;
  PetscFunctionList    qlist;
  PetscObjectList      olist;
  char                 *class_name;    /*  for example, "Vec" */
  char                 *description;
  char                 *mansec;
  char                 *type_name;     /*  this is the subclass, for example VECSEQ which equals "seq" */
  PetscObject          parent;
  PetscObjectId        parentid;
  char*                name;
  char                 *prefix;
  PetscInt             tablevel;
  void                 *cpp;
  PetscObjectState     state;
  PetscInt             int_idmax,        intstar_idmax;
  PetscObjectState     *intcomposedstate,*intstarcomposedstate;
  PetscInt             *intcomposeddata, **intstarcomposeddata;
  PetscInt             real_idmax,        realstar_idmax;
  PetscObjectState     *realcomposedstate,*realstarcomposedstate;
  PetscReal            *realcomposeddata, **realstarcomposeddata;
  PetscInt             scalar_idmax,        scalarstar_idmax;
  PetscObjectState     *scalarcomposedstate,*scalarstarcomposedstate;
  PetscScalar          *scalarcomposeddata, **scalarstarcomposeddata;
  void                 (**fortran_func_pointers)(void);                  /* used by Fortran interface functions to stash user provided Fortran functions */
  PetscInt             num_fortran_func_pointers;                        /* number of Fortran function pointers allocated */
  PetscFortranCallback *fortrancallback[PETSC_FORTRAN_CALLBACK_MAXTYPE];
  PetscInt             num_fortrancallback[PETSC_FORTRAN_CALLBACK_MAXTYPE];
  void                 *python_context;
  PetscErrorCode       (*python_destroy)(void*);

  PetscInt             noptionhandler;
  PetscErrorCode       (*optionhandler[PETSC_MAX_OPTIONS_HANDLER])(PetscObject,void*);
  PetscErrorCode       (*optiondestroy[PETSC_MAX_OPTIONS_HANDLER])(PetscObject,void*);
  void                 *optionctx[PETSC_MAX_OPTIONS_HANDLER];
  PetscPrecision       precision;
  PetscBool            optionsprinted;
#if defined(PETSC_HAVE_AMS)
  PetscInt             amsmem;
  PetscBool            amspublishblock; /* if PETSC_TRUE and publishing objects then will block at PetscObjectAMSBlock() */
  PetscBool            amsblock;
#endif
} _p_PetscObject;

#define PETSCHEADER(ObjectOps) \
  _p_PetscObject hdr;          \
  ObjectOps      *ops

#define  PETSCFREEDHEADER -1

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*PetscObjectFunction)(PetscObject*); /* force cast in next macro to NEVER use extern "C" style */
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*PetscObjectViewerFunction)(PetscObject,PetscViewer);

/*@C
    PetscHeaderCreate - Creates a PETSc object of a particular class, indicated by tp

    Input Parameters:
+   tp - the data structure type of the object (for example _p_Vec)
.   pops - the data structure type of the objects operations (for example VecOps)
.   classid - the classid associated with this object (for example VEC_CLASSID)
.   class_name - string name of class; should be static (for example "Vec")
.   com - the MPI Communicator
.   des - the destroy routine for this object (for example VecDestroy())
-   vie - the view routine for this object (for example VecView())

    Output Parameter:
.   h - the newly created object

    Level: developer

   Developer Note: This currently is a CPP macro because it takes the types (for example _p_Vec and VecOps) as arguments

.seealso: PetscHeaderDestroy(), PetscClassIdRegister()

@*/
#define PetscHeaderCreate(h,tp,pops,classid,class_name,descr,mansec,com,des,vie) \
  (PetscNew(struct tp,&(h)) ||                                                  \
   PetscNew(PetscOps,&(((PetscObject)(h))->bops)) ||                            \
   PetscNew(pops,&((h)->ops)) ||                                                \
   PetscHeaderCreate_Private((PetscObject)h,classid,class_name,descr,mansec,com,(PetscObjectFunction)des,(PetscObjectViewerFunction)vie) || \
   PetscLogObjectCreate(h) ||                                                   \
   PetscLogObjectMemory((PetscObject)h, sizeof(struct tp) + sizeof(PetscOps) + sizeof(pops)))

PETSC_EXTERN PetscErrorCode PetscComposedQuantitiesDestroy(PetscObject obj);
PETSC_EXTERN PetscErrorCode PetscHeaderCreate_Private(PetscObject,PetscClassId,const char[],const char[],const char[],MPI_Comm,PetscErrorCode (*)(PetscObject*),PetscErrorCode (*)(PetscObject,PetscViewer));

/*@C
    PetscHeaderDestroy - Final step in destroying a PetscObject

    Input Parameters:
.   h - the header created with PetscHeaderCreate()

    Level: developer

   Developer Note: This currently is a CPP macro because it accesses (*h)->ops which is a field in the derived class but not the PetscObject base class

.seealso: PetscHeaderCreate()
@*/
#define PetscHeaderDestroy(h)                         \
  (PetscHeaderDestroy_Private((PetscObject)(*h)) ||   \
   PetscFree((*h)->ops) ||                            \
   PetscFree(*h))

PETSC_EXTERN PetscErrorCode PetscHeaderDestroy_Private(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectCopyFortranFunctionPointers(PetscObject,PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSetFortranCallback(PetscObject,PetscFortranCallbackType,PetscFortranCallbackId*,void(*)(void),void *ctx);
PETSC_EXTERN PetscErrorCode PetscObjectGetFortranCallback(PetscObject,PetscFortranCallbackType,PetscFortranCallbackId,void(**)(void),void **ctx);

PETSC_INTERN PetscErrorCode PetscCitationsInitialize(void);
PETSC_INTERN PetscErrorCode PetscOptionsFindPair_Private(const char[],const char[],char**,PetscBool*);

PETSC_EXTERN PetscBool PetscCheckPointer(const void*,PetscDataType);

/*
    Macros to test if a PETSc object is valid and if pointers are valid
*/
#if !defined(PETSC_USE_DEBUG)

#define PetscValidHeaderSpecific(h,ck,arg) do {} while (0)
#define PetscValidHeader(h,arg) do {} while (0)
#define PetscValidPointer(h,arg) do {} while (0)
#define PetscValidCharPointer(h,arg) do {} while (0)
#define PetscValidIntPointer(h,arg) do {} while (0)
#define PetscValidScalarPointer(h,arg) do {} while (0)
#define PetscValidRealPointer(h,arg) do {} while (0)

#else

#define PetscValidHeaderSpecific(h,ck,arg)                              \
  do {                                                                  \
    if (!h) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Object: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_OBJECT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object: Parameter # %d",arg); \
    if (((PetscObject)(h))->classid != ck) {                            \
      if (((PetscObject)(h))->classid == PETSCFREEDHEADER) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Object already free: Parameter # %d",arg); \
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong type of object: Parameter # %d",arg); \
    }                                                                   \
  } while (0)

#define PetscValidHeader(h,arg)                                         \
  do {                                                                  \
    if (!h) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Object: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_OBJECT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object: Parameter # %d",arg); \
    if (((PetscObject)(h))->classid == PETSCFREEDHEADER) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Object already free: Parameter # %d",arg); \
    else if (((PetscObject)(h))->classid < PETSC_SMALLEST_CLASSID || ((PetscObject)(h))->classid > PETSC_LARGEST_CLASSID) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid type of object: Parameter # %d",arg); \
  } while (0)

#define PetscValidPointer(h,arg)                                        \
  do {                                                                  \
    if (!h) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_CHAR)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer: Parameter # %d",arg); \
  } while (0)

#define PetscValidCharPointer(h,arg)                                    \
  do {                                                                  \
    if (!h) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);\
    if (!PetscCheckPointer(h,PETSC_CHAR)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to char: Parameter # %d",arg); \
  } while (0)

#define PetscValidIntPointer(h,arg)                                     \
  do {                                                                  \
    if (!h) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_INT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to Int: Parameter # %d",arg); \
  } while (0)

#define PetscValidScalarPointer(h,arg)                                  \
  do {                                                                  \
    if (!h) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_SCALAR)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscScalar: Parameter # %d",arg); \
  } while (0)

#define PetscValidRealPointer(h,arg)                                  \
  do {                                                                  \
    if (!h) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_REAL)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscReal: Parameter # %d",arg); \
  } while (0)

#endif

#if !defined(PETSC_USE_DEBUG)

#define PetscCheckSameType(a,arga,b,argb) do {} while (0)
#define PetscValidType(a,arg) do {} while (0)
#define PetscCheckSameComm(a,arga,b,argb) do {} while (0)
#define PetscCheckSameTypeAndComm(a,arga,b,argb) do {} while (0)
#define PetscValidLogicalCollectiveScalar(a,b,c) do {} while (0)
#define PetscValidLogicalCollectiveReal(a,b,c) do {} while (0)
#define PetscValidLogicalCollectiveInt(a,b,c) do {} while (0)
#define PetscValidLogicalCollectiveBool(a,b,c) do {} while (0)
#define PetscValidLogicalCollectiveEnum(a,b,c) do {} while (0)

#else

/*
    For example, in the dot product between two vectors,
  both vectors must be either Seq or MPI, not one of each
*/
#define PetscCheckSameType(a,arga,b,argb) \
  if (((PetscObject)a)->type != ((PetscObject)b)->type) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Objects not of same type: Argument # %d and %d",arga,argb);
/*
   Use this macro to check if the type is set
*/
#define PetscValidType(a,arg) \
  if (!((PetscObject)a)->type_name) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"%s object's type is not set: Argument # %d",((PetscObject)a)->class_name,arg);
/*
   Sometimes object must live on same communicator to inter-operate
*/
#define PetscCheckSameComm(a,arga,b,argb)                               \
  do {                                                                  \
    PetscErrorCode _6_ierr,__flag;                                      \
    _6_ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)a),PetscObjectComm((PetscObject)b),&__flag);CHKERRQ(_6_ierr);                                                   \
    if (__flag != MPI_CONGRUENT && __flag != MPI_IDENT) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the two objects: Argument # %d and %d flag %d",arga,argb,__flag); \
  } while (0)

#define PetscCheckSameTypeAndComm(a,arga,b,argb)        \
  do {                                                  \
    PetscCheckSameType(a,arga,b,argb);                  \
    PetscCheckSameComm(a,arga,b,argb);                  \
  } while (0)

#define PetscValidLogicalCollectiveScalar(a,b,c)                        \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscReal b1[2],b2[2];                                              \
    b1[0] = -PetscRealPart(b); b1[1] = PetscRealPart(b);                \
    _7_ierr = MPI_Allreduce(b1,b2,2,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)a));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)a),PETSC_ERR_ARG_WRONG,"Scalar value must be same on all processes, argument # %d",c); \
  } while (0)

#define PetscValidLogicalCollectiveReal(a,b,c)                          \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscReal b1[2],b2[2];                                              \
    b1[0] = -b; b1[1] = b;                                              \
    _7_ierr = MPI_Allreduce(b1,b2,2,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)a));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)a),PETSC_ERR_ARG_WRONG,"Real value must be same on all processes, argument # %d",c); \
  } while (0)

#define PetscValidLogicalCollectiveInt(a,b,c)                           \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscInt b1[2],b2[2];                                               \
    b1[0] = -b; b1[1] = b;                                              \
    _7_ierr = MPI_Allreduce(b1,b2,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)a));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)a),PETSC_ERR_ARG_WRONG,"Int value must be same on all processes, argument # %d",c); \
  } while (0)

#define PetscValidLogicalCollectiveBool(a,b,c)                          \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscMPIInt b1[2],b2[2];                                            \
    b1[0] = -(PetscMPIInt)b; b1[1] = (PetscMPIInt)b;                    \
    _7_ierr = MPI_Allreduce(b1,b2,2,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)a));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)a),PETSC_ERR_ARG_WRONG,"Bool value must be same on all processes, argument # %d",c); \
  } while (0)

#define PetscValidLogicalCollectiveEnum(a,b,c)                          \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscMPIInt b1[2],b2[2];                                            \
    b1[0] = -(PetscMPIInt)b; b1[1] = (PetscMPIInt)b;                    \
    _7_ierr = MPI_Allreduce(b1,b2,2,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)a));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)a),PETSC_ERR_ARG_WRONG,"Enum value must be same on all processes, argument # %d",c); \
  } while (0)

#endif

/*
   PetscTryMethod - Queries an object for a method, if it exists then calls it.
              These are intended to be used only inside PETSc functions.

   Level: developer

.seealso: PetscUseMethod()
*/
#define  PetscTryMethod(obj,A,B,C) \
  0;{ PetscErrorCode (*f)B, __ierr; \
    __ierr = PetscObjectQueryFunction((PetscObject)obj,A,&f);CHKERRQ(__ierr); \
    if (f) {__ierr = (*f)C;CHKERRQ(__ierr);}\
  }

/*
   PetscUseMethod - Queries an object for a method, if it exists then calls it, otherwise generates an error.
              These are intended to be used only inside PETSc functions.

   Level: developer

.seealso: PetscTryMethod()
*/
#define  PetscUseMethod(obj,A,B,C) \
  0;{ PetscErrorCode (*f)B, __ierr; \
    __ierr = PetscObjectQueryFunction((PetscObject)obj,A,&f);CHKERRQ(__ierr); \
    if (f) {__ierr = (*f)C;CHKERRQ(__ierr);}\
    else SETERRQ1(PetscObjectComm((PetscObject)obj),PETSC_ERR_SUP,"Cannot locate function %s in object",A); \
  }

/*MC
   PetscObjectStateIncrease - Increases the state of any PetscObject,
   regardless of the type.

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectStateIncrease(PetscObject obj)

   Logically Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example,
         PetscObjectStateIncrease((PetscObject)mat);

   Notes: object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for Vec and Mat objects.

   This routine is mostly for internal use by PETSc; a developer need only
   call it after explicit access to an object's internals. Routines such
   as VecSet() or MatScale() already call this routine. It is also called, as a
   precaution, in VecRestoreArray(), MatRestoreRow(), MatDenseRestoreArray().

   This routine is logically collective because state equality comparison needs to be possible without communication.

   Level: developer

   seealso: PetscObjectStateGet()

   Concepts: state

M*/
#define PetscObjectStateIncrease(obj) ((obj)->state++,0)

PETSC_EXTERN PetscErrorCode PetscObjectStateGet(PetscObject,PetscObjectState*);
PETSC_EXTERN PetscErrorCode PetscObjectStateSet(PetscObject,PetscObjectState);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataRegister(PetscInt*);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseInt(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseIntstar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseReal(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseRealstar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseScalar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseScalarstar(PetscObject);
PETSC_EXTERN PetscInt         PetscObjectComposedDataMax;
/*MC
   PetscObjectComposedDataSetInt - attach integer data to a PetscObject

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataSetInt(PetscObject obj,int id,int data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#define PetscObjectComposedDataSetInt(obj,id,data)                                      \
  ((((obj)->int_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseInt(obj)) ||  \
   ((obj)->intcomposeddata[id] = data,(obj)->intcomposedstate[id] = (obj)->state, 0))

/*MC
   PetscObjectComposedDataGetInt - retrieve integer data attached to an object

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataGetInt(PetscObject obj,int id,int data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectComposedDataGetInt(obj,id,data,flag)                            \
  ((((obj)->intcomposedstate && ((obj)->intcomposedstate[id] == (obj)->state)) ?   \
   (data = (obj)->intcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)

/*MC
   PetscObjectComposedDataSetIntstar - attach integer array data to a PetscObject

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataSetIntstar(PetscObject obj,int id,int *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#define PetscObjectComposedDataSetIntstar(obj,id,data)                                          \
  ((((obj)->intstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseIntstar(obj)) ||  \
   ((obj)->intstarcomposeddata[id] = data,(obj)->intstarcomposedstate[id] = (obj)->state, 0))

/*MC
   PetscObjectComposedDataGetIntstar - retrieve integer array data
   attached to an object

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataGetIntstar(PetscObject obj,int id,int *data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectComposedDataGetIntstar(obj,id,data,flag)                               \
  ((((obj)->intstarcomposedstate && ((obj)->intstarcomposedstate[id] == (obj)->state)) ?  \
   (data = (obj)->intstarcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)

/*MC
   PetscObjectComposedDataSetReal - attach real data to a PetscObject

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataSetReal(PetscObject obj,int id,PetscReal data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#define PetscObjectComposedDataSetReal(obj,id,data)                                       \
  ((((obj)->real_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseReal(obj)) ||  \
   ((obj)->realcomposeddata[id] = data,(obj)->realcomposedstate[id] = (obj)->state, 0))

/*MC
   PetscObjectComposedDataGetReal - retrieve real data attached to an object

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataGetReal(PetscObject obj,int id,PetscReal data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectComposedDataGetReal(obj,id,data,flag)                            \
  ((((obj)->realcomposedstate && ((obj)->realcomposedstate[id] == (obj)->state)) ?  \
   (data = (obj)->realcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)

/*MC
   PetscObjectComposedDataSetRealstar - attach real array data to a PetscObject

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataSetRealstar(PetscObject obj,int id,PetscReal *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#define PetscObjectComposedDataSetRealstar(obj,id,data)                                           \
  ((((obj)->realstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseRealstar(obj)) ||  \
   ((obj)->realstarcomposeddata[id] = data, (obj)->realstarcomposedstate[id] = (obj)->state, 0))

/*MC
   PetscObjectComposedDataGetRealstar - retrieve real array data
   attached to an object

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataGetRealstar(PetscObject obj,int id,PetscReal *data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectComposedDataGetRealstar(obj,id,data,flag)                                \
  ((((obj)->realstarcomposedstate && ((obj)->realstarcomposedstate[id] == (obj)->state)) ?  \
   (data = (obj)->realstarcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)

/*MC
   PetscObjectComposedDataSetScalar - attach scalar data to a PetscObject

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataSetScalar(PetscObject obj,int id,PetscScalar data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectComposedDataSetScalar(obj,id,data)                                        \
  ((((obj)->scalar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseScalar(obj)) || \
   ((obj)->scalarcomposeddata[id] = data,(obj)->scalarcomposedstate[id] = (obj)->state, 0))
#else
#define PetscObjectComposedDataSetScalar(obj,id,data) \
        PetscObjectComposedDataSetReal(obj,id,data)
#endif
/*MC
   PetscObjectComposedDataGetScalar - retrieve scalar data attached to an object

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataGetScalar(PetscObject obj,int id,PetscScalar data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectComposedDataGetScalar(obj,id,data,flag)                              \
  ((((obj)->scalarcomposedstate && ((obj)->scalarcomposedstate[id] == (obj)->state) ) ? \
   (data = (obj)->scalarcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)
#else
#define PetscObjectComposedDataGetScalar(obj,id,data,flag)                             \
        PetscObjectComposedDataGetReal(obj,id,data,flag)
#endif

/*MC
   PetscObjectComposedDataSetScalarstar - attach scalar array data to a PetscObject

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataSetScalarstar(PetscObject obj,int id,PetscScalar *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectComposedDataSetScalarstar(obj,id,data)                                             \
  ((((obj)->scalarstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseScalarstar(obj)) ||  \
   ((obj)->scalarstarcomposeddata[id] = data,(obj)->scalarstarcomposedstate[id] = (obj)->state, 0))
#else
#define PetscObjectComposedDataSetScalarstar(obj,id,data) \
        PetscObjectComposedDataSetRealstar(obj,id,data)
#endif
/*MC
   PetscObjectComposedDataGetScalarstar - retrieve scalar array data
   attached to an object

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscObjectComposedDataGetScalarstar(PetscObject obj,int id,PetscScalar *data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectComposedDataGetScalarstar(obj,id,data,flag)                                 \
  ((((obj)->scalarstarcomposedstate && ((obj)->scalarstarcomposedstate[id] == (obj)->state)) ? \
       (data = (obj)->scalarstarcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)
#else
#define PetscObjectComposedDataGetScalarstar(obj,id,data,flag)         \
        PetscObjectComposedDataGetRealstar(obj,id,data,flag)
#endif

PETSC_EXTERN PetscErrorCode PetscObjectGetId(PetscObject,PetscObjectId*);

PETSC_EXTERN PetscMPIInt Petsc_Counter_keyval;
PETSC_EXTERN PetscMPIInt Petsc_InnerComm_keyval;
PETSC_EXTERN PetscMPIInt Petsc_OuterComm_keyval;

/*
  PETSc communicators have this attribute, see
  PetscCommDuplicate(), PetscCommDestroy(), PetscCommGetNewTag(), PetscObjectGetName()
*/
typedef struct {
  PetscMPIInt tag;              /* next free tag value */
  PetscInt    refcount;         /* number of references, communicator can be freed when this reaches 0 */
  PetscInt    namecount;        /* used to generate the next name, as in Vec_0, Mat_1, ... */
} PetscCommCounter;

#if defined(PETSC_HAVE_CUSP)
/*E
    PetscCUSPFlag - indicates which memory (CPU, GPU, or none contains valid vector

   PETSC_CUSP_UNALLOCATED  - no memory contains valid matrix entries; NEVER used for vectors
   PETSC_CUSP_GPU - GPU has valid vector/matrix entries
   PETSC_CUSP_CPU - CPU has valid vector/matrix entries
   PETSC_CUSP_BOTH - Both GPU and CPU have valid vector/matrix entries and they match

   Level: developer
E*/
typedef enum {PETSC_CUSP_UNALLOCATED,PETSC_CUSP_GPU,PETSC_CUSP_CPU,PETSC_CUSP_BOTH} PetscCUSPFlag;
#endif

#if defined(PETSC_HAVE_VIENNACL)
/*E
    PetscViennaCLFlag - indicates which memory (CPU, GPU, or none contains valid vector

   PETSC_VIENNACL_UNALLOCATED  - no memory contains valid matrix entries; NEVER used for vectors
   PETSC_VIENNACL_GPU - GPU has valid vector/matrix entries
   PETSC_VIENNACL_CPU - CPU has valid vector/matrix entries
   PETSC_VIENNACL_BOTH - Both GPU and CPU have valid vector/matrix entries and they match

   Level: developer
E*/
typedef enum {PETSC_VIENNACL_UNALLOCATED,PETSC_VIENNACL_GPU,PETSC_VIENNACL_CPU,PETSC_VIENNACL_BOTH} PetscViennaCLFlag;
#endif

#endif /* _PETSCHEAD_H */

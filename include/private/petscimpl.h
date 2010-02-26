
/*
    Defines the basic header of all PETSc objects.
*/

#if !defined(_PETSCHEAD_H)
#define _PETSCHEAD_H
#include "petscsys.h"  
PETSC_EXTERN_CXX_BEGIN

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
      destroy()         - Is the routine for destroying the entire PETSc object; 
                          for example,MatDestroy() is the general matrix 
                          destruction routine.
      compose()         - Associates a PETSc object with another PETSc object.
      query()           - Returns a different PETSc object that has been associated
                          with the first object.
      composefunction() - Attaches an additional registered function.
      queryfunction()   - Requests a registered function that has been registered.
      publish()         - Not currently used
*/

typedef struct {
   PetscErrorCode (*getcomm)(PetscObject,MPI_Comm *);
   PetscErrorCode (*view)(PetscObject,PetscViewer);
   PetscErrorCode (*destroy)(PetscObject);
   PetscErrorCode (*compose)(PetscObject,const char[],PetscObject);
   PetscErrorCode (*query)(PetscObject,const char[],PetscObject *);
   PetscErrorCode (*composefunction)(PetscObject,const char[],const char[],void (*)(void));
   PetscErrorCode (*queryfunction)(PetscObject,const char[],void (**)(void));
   PetscErrorCode (*publish)(PetscObject);
} PetscOps;

/*
   All PETSc objects begin with the fields defined in PETSCHEADER.
   The PetscObject is a way of examining these fields regardless of 
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
typedef struct _p_PetscObject {
  PetscCookie    cookie;                                        
  PetscOps       *bops;                                         
  MPI_Comm       comm;                                          
  PetscInt       type;                                          
  PetscLogDouble flops,time,mem;                                
  PetscInt       id;                                            
  PetscInt       refct;                                         
  PetscMPIInt    tag;                                           
  PetscFList     qlist;                                         
  PetscOList     olist;                                         
  char           *class_name;                                   
  char           *type_name;                                    
  PetscObject    parent;                                        
  PetscInt       parentid;                                      
  char*          name;                                          
  char           *prefix;                                       
  PetscInt       tablevel;                                      
  void           *cpp;                                          
  PetscInt       amem;                                          
  PetscInt       state;                                         
  PetscInt       int_idmax,        intstar_idmax;               
  PetscInt       *intcomposedstate,*intstarcomposedstate;       
  PetscInt       *intcomposeddata, **intstarcomposeddata;       
  PetscInt       real_idmax,        realstar_idmax;             
  PetscInt       *realcomposedstate,*realstarcomposedstate;     
  PetscReal      *realcomposeddata, **realstarcomposeddata;     
  PetscInt       scalar_idmax,        scalarstar_idmax;         
  PetscInt       *scalarcomposedstate,*scalarstarcomposedstate; 
  PetscScalar    *scalarcomposeddata, **scalarstarcomposeddata; 
  void           (**fortran_func_pointers)(void);                  /* used by Fortran interface functions to stash user provided Fortran functions */
  void           *python_context;                               
  PetscErrorCode (*python_destroy)(void*);
} _p_PetscObject;

#define PETSCHEADER(ObjectOps) \
  _p_PetscObject hdr;	       \
  ObjectOps      *ops

#define  PETSCFREEDHEADER -1

typedef PetscErrorCode (*PetscObjectFunction)(PetscObject); /* force cast in next macro to NEVER use extern "C" style */
typedef PetscErrorCode (*PetscObjectViewerFunction)(PetscObject,PetscViewer); 

/*@C
    PetscHeaderCreate - Creates a PETSc object

    Input Parameters:
+   tp - the data structure type of the object
.   pops - the data structure type of the objects operations (for example VecOps)
.   cook - the cookie associated with this object
.   t - type (no longer should be used)
.   class_name - string name of class; should be static
.   com - the MPI Communicator
.   des - the destroy routine for this object
-   vie - the view routine for this object

    Output Parameter:
.   h - the newly created object

    Level: developer

.seealso: PetscHeaderDestroy(), PetscCookieRegister()

@*/ 
#define PetscHeaderCreate(h,tp,pops,cook,t,class_name,com,des,vie)	\
  (PetscNew(struct tp,&(h)) ||						\
   PetscNew(PetscOps,&(((PetscObject)(h))->bops)) ||			\
   PetscNew(pops,&((h)->ops)) ||					\
   PetscHeaderCreate_Private((PetscObject)h,cook,t,class_name,com,(PetscObjectFunction)des,(PetscObjectViewerFunction)vie) || \
   PetscLogObjectCreate(h) ||						\
   PetscLogObjectMemory(h, sizeof(struct tp) + sizeof(PetscOps) + sizeof(pops)))

EXTERN PetscErrorCode PetscComposedQuantitiesDestroy(PetscObject obj);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscHeaderCreate_Private(PetscObject,PetscCookie,PetscInt,const char[],MPI_Comm,PetscErrorCode (*)(PetscObject),PetscErrorCode (*)(PetscObject,PetscViewer));

/*@C
    PetscHeaderDestroy - Final step in destroying a PetscObject

    Input Parameters:
.   h - the header created with PetscHeaderCreate()

    Level: developer

.seealso: PetscHeaderCreate()
@*/ 
#define PetscHeaderDestroy(h)			   \
  (PetscLogObjectDestroy((PetscObject)(h)) ||	   \
   PetscComposedQuantitiesDestroy((PetscObject)h) || \
   PetscHeaderDestroy_Private((PetscObject)(h)) || \
   PetscFree((h)->ops) ||			   \
   PetscFree(h))

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscHeaderDestroy_Private(PetscObject);

/* ---------------------------------------------------------------------------------------*/

#if !defined(PETSC_USE_DEBUG)

#define PetscValidHeaderSpecific(h,ck,arg)
#define PetscValidHeader(h,arg)
#define PetscValidPointer(h,arg)
#define PetscValidCharPointer(h,arg)
#define PetscValidIntPointer(h,arg)
#define PetscValidScalarPointer(h,arg)

#elif !defined(PETSC_HAVE_UNALIGNED_POINTERS)
/* 
    Macros to test if a PETSc object is valid and if pointers are
valid

*/
#define PetscValidHeaderSpecific(h,ck,arg)                                            \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Object: Parameter # %d",arg);}          \
   if ((unsigned long)(h) & (unsigned long)3) {			\
    SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object: Parameter # %d",arg);   \
  }                                                                                   \
  if (((PetscObject)(h))->cookie != ck) {                                             \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {                             \
      SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Object already free: Parameter # %d",arg);       \
    } else {                                                                          \
      SETERRQ1(PETSC_ERR_ARG_WRONG,"Wrong type of object: Parameter # %d",arg);       \
    }                                                                                 \
  }} 

#define PetscValidHeader(h,arg)                                                              \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Object: Parameter # %d",arg);}             \
   if ((unsigned long)(h) & (unsigned long)3) {			\
    SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object: Parameter # %d",arg);     \
  } else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {                           \
      SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Object already free: Parameter # %d",arg);         \
  } else if (((PetscObject)(h))->cookie < PETSC_SMALLEST_COOKIE ||                                \
      ((PetscObject)(h))->cookie > PETSC_LARGEST_COOKIE) {                               \
      SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Invalid type of object: Parameter # %d",arg);      \
  }}

#define PetscValidPointer(h,arg)                                                              \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);}             \
   if ((unsigned long)(h) & (unsigned long)3){				\
    SETERRQ1(PETSC_ERR_ARG_BADPTR,"Invalid Pointer: Parameter # %d",arg);                 \
  }}

#define PetscValidCharPointer(h,arg)                                                           \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);}              \
  }

#define PetscValidIntPointer(h,arg)                                                            \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_BADPTR,"Null Pointer: Parameter # %d",arg);}            \
  if ((unsigned long)(h) & (unsigned long)3){				\
    SETERRQ1(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to Int: Parameter # %d",arg);           \
  }}

#if !defined(PETSC_HAVE_DOUBLES_ALIGNED) 
#define PetscValidScalarPointer(h,arg)                                                          \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);}               \
  if ((unsigned long)(h) & (unsigned long)3) {			\
    SETERRQ1(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscScalar: Parameter # %d",arg);    \
  }}
#else
#define PetscValidScalarPointer(h,arg)                                                          \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);}               \
  if ((unsigned long)(h) & (unsigned long)7) {			\
    SETERRQ1(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscScalar: Parameter # %d",arg);    \
  }}
#endif

#else
/*
     Version where pointers don't have any particular alignment
*/
#define PetscValidHeaderSpecific(h,ck,arg)                              \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_NULL,"Null Object");}        \
  if (((PetscObject)(h))->cookie != ck) {                           \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {           \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");       \
    } else {                                                        \
      SETERRQ(PETSC_ERR_ARG_WRONG,"Wrong Object");                \
    }                                                               \
  }} 

#define PetscValidHeader(h,arg)                                         \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_NULL,"Null Object");}        \
  if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {      \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");       \
  } else if (((PetscObject)(h))->cookie < PETSC_SMALLEST_COOKIE ||           \
      ((PetscObject)(h))->cookie > PETSC_LARGEST_COOKIE) {          \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid type of object");            \
  }}

#define PetscValidPointer(h,arg)                                        \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_NULL,"Null Pointer");}        \
  }

#define PetscValidCharPointer(h,arg)                                    \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_NULL,"Null Pointer");}        \
  }

#define PetscValidIntPointer(h,arg)                                     \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_NULL,"Null Pointer");}        \
  }

#if !defined(PETSC_HAVE_DOUBLES_ALIGNED)
#define PetscValidScalarPointer(h,arg)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_NULL,"Null Pointer");}        \
  }
#else
#define PetscValidScalarPointer(h,arg)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_NULL,"Null Pointer");}        \
  }
#endif

#endif
#define PetscValidDoublePointer(h,arg) PetscValidScalarPointer(h,arg)

#if !defined(PETSC_USE_DEBUG)

#define PetscCheckSameType(a,arga,b,argb)
#define PetscValidType(a,arg)
#define PetscCheckSameComm(a,arga,b,argb)
#define PetscCheckSameTypeAndComm(a,arga,b,argb)

#else

/*
    For example, in the dot product between two vectors,
  both vectors must be either Seq or MPI, not one of each 
*/
#define PetscCheckSameType(a,arga,b,argb) \
  if (((PetscObject)a)->type != ((PetscObject)b)->type) SETERRQ2(PETSC_ERR_ARG_NOTSAMETYPE,"Objects not of same type: Argument # %d and %d",arga,argb);
/* 
   Use this macro to check if the type is set
*/
#define PetscValidType(a,arg) \
  if (!((PetscObject)a)->type_name) SETERRQ2(PETSC_ERR_ARG_WRONGSTATE,"%s object's type is not set: Argument # %d",((PetscObject)a)->class_name,arg);
/*
   Sometimes object must live on same communicator to inter-operate
*/
#define PetscCheckSameComm(a,arga,b,argb) \
  {PetscErrorCode _6_ierr,__flag; _6_ierr = MPI_Comm_compare(((PetscObject)a)->comm,((PetscObject)b)->comm,&__flag);\
  CHKERRQ(_6_ierr); \
  if (__flag != MPI_CONGRUENT && __flag != MPI_IDENT) \
  SETERRQ2(PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the two objects: Argument # %d and %d",arga,argb);}

#define PetscCheckSameTypeAndComm(a,arga,b,argb) {\
  PetscCheckSameType(a,arga,b,argb);\
  PetscCheckSameComm(a,arga,b,argb);}

#endif

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectPublishBaseBegin(PetscObject);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectPublishBaseEnd(PetscObject);

/*MC
   PetscObjectStateIncrease - Increases the state of any PetscObject, 
   regardless of the type.

   Synopsis:
   PetscErrorCode PetscObjectStateIncrease(PetscObject obj)

   Not Collective

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
   as VecSet or MatScale already call this routine. It is also called, as a 
   precaution, in VecRestoreArray, MatRestoreRow, MatRestoreArray.

   Level: developer

   seealso: PetscObjectStateQuery(), PetscObjectStateDecrease()

   Concepts: state

M*/
#define PetscObjectStateIncrease(obj) ((obj)->state++,0)

/*MC
   PetscObjectStateDecrease - Decreases the state of any PetscObject, 
   regardless of the type.

   Synopsis:
   PetscErrorCode PetscObjectStateDecrease(PetscObject obj)

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example, 
         PetscObjectStateIncrease((PetscObject)mat);

   Notes: object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for Vec and Mat objects.

   Level: developer

   seealso: PetscObjectStateQuery(), PetscObjectStateIncrease()

   Concepts: state

M*/
#define PetscObjectStateDecrease(obj) ((obj)->state--,0)

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectStateQuery(PetscObject,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectSetState(PetscObject,PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectComposedDataRegister(PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectComposedDataIncreaseInt(PetscObject);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectComposedDataIncreaseIntstar(PetscObject);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectComposedDataIncreaseReal(PetscObject);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectComposedDataIncreaseRealstar(PetscObject);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectComposedDataIncreaseScalar(PetscObject);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscObjectComposedDataIncreaseScalarstar(PetscObject);
EXTERN PetscInt       PETSC_DLLEXPORT globalcurrentstate;
EXTERN PetscInt       PETSC_DLLEXPORT globalmaxstate;
/*MC
   PetscObjectComposedDataSetInt - attach integer data to a PetscObject

   Synopsis:
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
  ((((obj)->int_idmax < globalmaxstate) ? PetscObjectComposedDataIncreaseInt(obj) : 0), \
   (obj)->intcomposeddata[id] = data,(obj)->intcomposedstate[id] = (obj)->state, 0)

/*MC
   PetscObjectComposedDataGetInt - retrieve integer data attached to an object

   Synopsis:
   PetscErrorCode PetscObjectComposedDataGetInt(PetscObject obj,int id,int data,PetscTruth flag)

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
  ((((obj)->intstar_idmax < globalmaxstate) ? PetscObjectComposedDataIncreaseIntstar(obj) : 0), \
  (obj)->intstarcomposeddata[id] = data,(obj)->intstarcomposedstate[id] = (obj)->state, 0)

/*MC
   PetscObjectComposedDataGetIntstar - retrieve integer array data 
   attached to an object

   Synopsis:
   PetscErrorCode PetscObjectComposedDataGetIntstar(PetscObject obj,int id,int *data,PetscTruth flag)

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
  ((((obj)->real_idmax < globalmaxstate) ? PetscObjectComposedDataIncreaseReal(obj) : 0), \
   (obj)->realcomposeddata[id] = data,(obj)->realcomposedstate[id] = (obj)->state, 0)

/*MC
   PetscObjectComposedDataGetReal - retrieve real data attached to an object

   Synopsis:
   PetscErrorCode PetscObjectComposedDataGetReal(PetscObject obj,int id,PetscReal data,PetscTruth flag)

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
  ((((obj)->realstar_idmax < globalmaxstate) ? PetscObjectComposedDataIncreaseRealstar(obj) : 0), \
  (obj)->realstarcomposeddata[id] = data, (obj)->realstarcomposedstate[id] = (obj)->state, 0)

/*MC
   PetscObjectComposedDataGetRealstar - retrieve real array data
   attached to an object

   Synopsis:
   PetscErrorCode PetscObjectComposedDataGetRealstar(PetscObject obj,int id,PetscReal *data,PetscTruth flag)

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
  ((((obj)->scalar_idmax < globalmaxstate) ? PetscObjectComposedDataIncreaseScalar(obj) : 0) \
  (obj)->scalarcomposeddata[id] = data,(obj)->scalarcomposedstate[id] = (obj)->state, 0)
#else
#define PetscObjectComposedDataSetScalar(obj,id,data) \
        PetscObjectComposedDataSetReal(obj,id,data)
#endif
/*MC
   PetscObjectComposedDataGetScalar - retrieve scalar data attached to an object

   Synopsis:
   PetscErrorCode PetscObjectComposedDataGetScalar(PetscObject obj,int id,PetscScalar data,PetscTruth flag)

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
  ((((obj)->scalarstar_idmax < globalmaxstate) ? PetscObjectComposedDataIncreaseScalarstar(obj) : 0), \
   (obj)->scalarstarcomposeddata[id] = data,(obj)->scalarstarcomposedstate[id] = (obj)->state, 0)
#else
#define PetscObjectComposedDataSetScalarstar(obj,id,data) \
        PetscObjectComposedDataSetRealstar(obj,id,data)
#endif
/*MC
   PetscObjectComposedDataGetScalarstar - retrieve scalar array data
   attached to an object

   Synopsis:
   PetscErrorCode PetscObjectComposedDataGetScalarstar(PetscObject obj,int id,PetscScalar *data,PetscTruth flag)

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
#define PetscObjectComposedDataGetScalarstar(obj,id,data,flag)	         \
        PetscObjectComposedDataGetRealstar(obj,id,data,flag)
#endif

/* some vars for logging */
extern PETSC_DLLEXPORT PetscTruth PetscPreLoadingUsed;       /* true if we are or have done preloading */
extern PETSC_DLLEXPORT PetscTruth PetscPreLoadingOn;         /* true if we are currently in a preloading calculation */

extern PETSC_DLLEXPORT PetscMPIInt Petsc_Counter_keyval;
extern PETSC_DLLEXPORT PetscMPIInt Petsc_InnerComm_keyval;
extern PETSC_DLLEXPORT PetscMPIInt Petsc_OuterComm_keyval;

/*
  PETSc communicators have this attribute, see
  PetscCommDuplicate(), PetscCommDestroy(), PetscCommGetNewTag(), PetscObjectGetName()
*/
typedef struct {
  PetscMPIInt tag;              /* next free tag value */
  PetscInt    refcount;         /* number of references, communicator can be freed when this reaches 0 */
  PetscInt    namecount;        /* used to generate the next name, as in Vec_0, Mat_1, ... */
} PetscCommCounter;

PETSC_EXTERN_CXX_END

#endif /* _PETSCHEAD_H */


/*
    Defines the basic header of all PETSc objects.
*/

#if !defined(_PETSCHEAD_H)
#define _PETSCHEAD_H
#include "petsc.h"  
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
      reference()       - Increases the reference count for a PETSc object; when
                          a reference count reaches zero it is destroyed.
      destroy()         - Is the routine for destroying the entire PETSc object; 
                          for example,MatDestroy() is the general matrix 
                          destruction routine.
      compose()         - Associates a PETSc object with another PETSc object.
      query()           - Returns a different PETSc object that has been associated
                          with the first object.
      composefunction() - Attaches an additional registered function.
      queryfunction()   - Requests a registered function that has been registered.
      composelanguage() - associates the object's representation in a different language
      querylanguage()   - obtain the object's representation in a different language
*/

typedef struct {
   PetscErrorCode (*getcomm)(PetscObject,MPI_Comm *);
   PetscErrorCode (*view)(PetscObject,PetscViewer);
   PetscErrorCode (*reference)(PetscObject);
   PetscErrorCode (*destroy)(PetscObject);
   PetscErrorCode (*compose)(PetscObject,const char[],PetscObject);
   PetscErrorCode (*query)(PetscObject,const char[],PetscObject *);
   PetscErrorCode (*composefunction)(PetscObject,const char[],const char[],void (*)(void));
   PetscErrorCode (*queryfunction)(PetscObject,const char[],void (**)(void));
   PetscErrorCode (*composelanguage)(PetscObject,PetscLanguage,void *);
   PetscErrorCode (*querylanguage)(PetscObject,PetscLanguage,void **);
   PetscErrorCode (*publish)(PetscObject);
} PetscOps;

#define PETSCHEADER(ObjectOps)                                  \
  PetscCookie    cookie;                                        \
  PetscOps       *bops;                                         \
  ObjectOps      *ops;                                          \
  MPI_Comm       comm;                                          \
  PetscInt       type;                                          \
  PetscLogDouble flops,time,mem;                                \
  PetscInt       id;                                            \
  PetscInt       refct;                                         \
  PetscMPIInt    tag;                                           \
  PetscFList     qlist;                                         \
  PetscOList     olist;                                         \
  char           *class_name;                                   \
  char           *type_name;                                    \
  PetscObject    parent;                                        \
  PetscInt       parentid;                                      \
  char*          name;                                          \
  char           *prefix;                                       \
  void           *cpp;                                          \
  PetscInt       amem;                                          \
  PetscInt       state;                                         \
  PetscInt       int_idmax,        intstar_idmax;               \
  PetscInt       *intcomposedstate,*intstarcomposedstate;       \
  PetscInt       *intcomposeddata, **intstarcomposeddata;       \
  PetscInt       real_idmax,        realstar_idmax;             \
  PetscInt       *realcomposedstate,*realstarcomposedstate;     \
  PetscReal      *realcomposeddata, **realstarcomposeddata;     \
  PetscInt       scalar_idmax,        scalarstar_idmax;         \
  PetscInt       *scalarcomposedstate,*scalarstarcomposedstate; \
  PetscScalar    *scalarcomposeddata, **scalarstarcomposeddata;	\
  void           (**fortran_func_pointers)(void)

  /*  ... */                               

#define  PETSCFREEDHEADER -1

EXTERN PetscErrorCode PetscHeaderCreate_Private(PetscObject,PetscCookie,PetscInt,const char[],MPI_Comm,PetscErrorCode (*)(PetscObject),PetscErrorCode (*)(PetscObject,PetscViewer));
EXTERN PetscErrorCode PetscHeaderDestroy_Private(PetscObject);

typedef PetscErrorCode (*PetscObjectFunction)(PetscObject); /* force cast in next macro to NEVER use extern "C" style */
typedef PetscErrorCode (*PetscObjectViewerFunction)(PetscObject,PetscViewer); 

/*
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
*/ 
#define PetscHeaderCreate(h,tp,pops,cook,t,class_name,com,des,vie) \
  (PetscNew(struct tp,&(h)) || \
   PetscNew(PetscOps,&((h)->bops)) || \
   PetscNew(pops,&((h)->ops)) || \
   PetscHeaderCreate_Private((PetscObject)h,cook,t,class_name,com,(PetscObjectFunction)des,(PetscObjectViewerFunction)vie) || \
   PetscLogObjectCreate(h))

#define PetscHeaderDestroy(h) \
  (PetscLogObjectDestroy((PetscObject)(h)) || \
   PetscHeaderDestroy_Private((PetscObject)(h)) || \
   PetscFree(h))

/* ---------------------------------------------------------------------------------------*/

#if !defined(PETSC_USE_DEBUG)

#define PetscValidHeaderSpecific(h,ck,arg)
#define PetscValidHeader(h,arg)
#define PetscValidPointer(h,arg)
#define PetscValidCharPointer(h,arg)
#define PetscValidIntPointer(h,arg)
#define PetscValidScalarPointer(h,arg)

#elif !defined(PETSC_HAVE_CRAY90_POINTER)
/* 
    Macros to test if a PETSc object is valid and if pointers are
valid

*/
#define PetscValidHeaderSpecific(h,ck,arg)                                            \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Object: Parameter # %d",arg);}          \
  if ((unsigned long)h & (unsigned long)3) {                                          \
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
  if ((unsigned long)h & (unsigned long)3) {                                             \
    SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object: Parameter # %d",arg);     \
  } else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {                           \
      SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Object already free: Parameter # %d",arg);         \
  } else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||                                \
      ((PetscObject)(h))->cookie > PETSC_LARGEST_COOKIE) {                               \
      SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Invalid type of object: Parameter # %d",arg);      \
  }}

#define PetscValidPointer(h,arg)                                                              \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);}             \
  if ((unsigned long)h & (unsigned long)3){                                               \
    SETERRQ1(PETSC_ERR_ARG_BADPTR,"Invalid Pointer: Parameter # %d",arg);                 \
  }}

#define PetscValidCharPointer(h,arg)                                                           \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);}              \
  }

#define PetscValidIntPointer(h,arg)                                                            \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_BADPTR,"Null Pointer: Parameter # %d",arg);}            \
  if ((unsigned long)h & (unsigned long)3){                                                \
    SETERRQ1(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to Int: Parameter # %d",arg);           \
  }}

#if !defined(PETSC_HAVE_DOUBLES_ALIGNED) || defined (PETSC_HAVE_DOUBLES_ALIGNED)
#define PetscValidScalarPointer(h,arg)                                                          \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);}               \
  if ((unsigned long)h & (unsigned long)3) {                                                \
    SETERRQ1(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscScalar: Parameter # %d",arg);    \
  }}
#else
#define PetscValidScalarPointer(h,arg)                                                          \
  {if (!h) {SETERRQ1(PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);}               \
  if ((unsigned long)h & (unsigned long)7) {                                                \
    SETERRQ1(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscScalar: Parameter # %d",arg);    \
  }}
#endif

#else
/*
     Version for Cray 90 that handles pointers differently
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
  } else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||           \
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
  if ((a)->type != (b)->type) SETERRQ2(PETSC_ERR_ARG_NOTSAMETYPE,"Objects not of same type: Argument # %d and %d",arga,argb);
/* 
   Use this macro to check if the type is set
*/
#define PetscValidType(a,arg) \
  if (!(a)->type_name) SETERRQ1(PETSC_ERR_ARG_WRONGSTATE,"Object Type not set: Argument # %d",arg);
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

/*
   All PETSc objects begin with the fields defined in PETSCHEADER.
   The PetscObject is a way of examining these fields regardless of 
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
struct _p_PetscObject {
  PETSCHEADER(int);
};

EXTERN PetscErrorCode PetscObjectPublishBaseBegin(PetscObject);
EXTERN PetscErrorCode PetscObjectPublishBaseEnd(PetscObject);

EXTERN PetscErrorCode PetscObjectIncreaseState(PetscObject);
EXTERN PetscErrorCode PetscObjectGetState(PetscObject,PetscInt*);
EXTERN PetscErrorCode PetscRegisterComposedData(PetscInt*);
EXTERN PetscErrorCode PetscObjectIncreaseIntComposedData(PetscObject);
EXTERN PetscErrorCode PetscObjectIncreaseIntstarComposedData(PetscObject);
EXTERN PetscErrorCode PetscObjectIncreaseRealComposedData(PetscObject);
EXTERN PetscErrorCode PetscObjectIncreaseRealstarComposedData(PetscObject);
EXTERN PetscErrorCode PetscObjectIncreaseScalarComposedData(PetscObject);
EXTERN PetscErrorCode PetscObjectIncreaseScalarstarComposedData(PetscObject);
EXTERN PetscInt globalcurrentstate,globalmaxstate;
/*MC
   PetscObjectSetIntComposedData - attach integer data to a PetscObject

   Synopsis:
   PetscErrorCode PetscObjectSetIntComposedData(PetscObject obj,int id,int data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   Level: developer
M*/
#define PetscObjectSetIntComposedData(obj,id,data)                  \
0; {PetscErrorCode ierr_;                                                       \
  if ((obj)->int_idmax < globalmaxstate) {                            \
    ierr_ = PetscObjectIncreaseIntComposedData(obj); CHKERRQ(ierr_);  \
  }                                                                   \
  (obj)->intcomposeddata[id] = data;                                  \
  (obj)->intcomposedstate[id] = (obj)->state;                         \
}
/*MC
   PetscObjectGetIntComposedData - retrieve integer data attached to an object

   Synopsis:
   PetscErrorCode PetscObjectGetIntComposedData(PetscObject obj,int id,int *data,PetscTruth *flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectGetIntComposedData(obj,id,data,flag)              \
0; {                                                                 \
  if ((obj)->intcomposedstate) {                                     \
    if ((obj)->intcomposedstate[id] == (obj)->state) {               \
      data = (obj)->intcomposeddata[id];                             \
      flag = PETSC_TRUE;                                             \
    } else {                                                         \
      flag = PETSC_FALSE;                                            \
    }                                                                \
  } else flag = PETSC_FALSE;                                         \
}

/*MC
   PetscObjectSetIntstarComposedData - attach integer array data to a PetscObject

   Synopsis:
   PetscErrorCode PetscObjectSetIntstarComposedData(PetscObject obj,int id,int *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   Level: developer
M*/
#define PetscObjectSetIntstarComposedData(obj,id,data)                    \
0; {PetscErrorCode ierr_;                                                            \
  if ((obj)->intstar_idmax < globalmaxstate) {                            \
    ierr_ = PetscObjectIncreaseIntstarComposedData(obj); CHKERRQ(ierr_);  \
  }                                                                       \
  (obj)->intstarcomposeddata[id] = data;                                  \
  (obj)->intstarcomposedstate[id] = (obj)->state;                         \
}
/*MC
   PetscObjectGetIntstarComposedData - retrieve integer array data 
   attached to an object

   Synopsis:
   PetscErrorCode PetscObjectGetIntstarComposedData(PetscObject obj,int id,int **data,PetscTruth *flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectGetIntstarComposedData(obj,id,data,flag)              \
0; {                                                                     \
  if ((obj)->intstarcomposedstate) {                                     \
    if ((obj)->intstarcomposedstate[id] == (obj)->state) {               \
      data = (obj)->intstarcomposeddata[id];                             \
      flag = PETSC_TRUE;                                                 \
    } else {                                                             \
      flag = PETSC_FALSE;                                                \
    }                                                                    \
  } else flag = PETSC_FALSE;                                             \
}

/*MC
   PetscObjectSetRealComposedData - attach real data to a PetscObject

   Synopsis:
   PetscErrorCode PetscObjectSetRealComposedData(PetscObject obj,int id,PetscReal data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   Level: developer
M*/
#define PetscObjectSetRealComposedData(obj,id,data)                  \
0; {PetscErrorCode ierr_;                                                       \
  if ((obj)->real_idmax < globalmaxstate) {                          \
    ierr_ = PetscObjectIncreaseRealComposedData(obj); CHKERRQ(ierr_); \
  }                                                                  \
  (obj)->realcomposeddata[id] = data;                                \
  (obj)->realcomposedstate[id] = (obj)->state;                       \
}
/*MC
   PetscObjectGetRealComposedData - retrieve real data attached to an object

   Synopsis:
   PetscErrorCode PetscObjectGetRealComposedData(PetscObject obj,int id,PetscReal *data,PetscTruth *flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectGetRealComposedData(obj,id,data,flag)             \
0; {                                                                 \
  if ((obj)->realcomposedstate) {                                    \
    if ((obj)->realcomposedstate[id] == (obj)->state) {              \
      data = (obj)->realcomposeddata[id];                            \
      flag = PETSC_TRUE;                                             \
    } else {                                                         \
      flag = PETSC_FALSE;                                            \
    }                                                                \
  } else {                                                           \
    flag = PETSC_FALSE;                                              \
  }                                                                  \
}

/*MC
   PetscObjectSetRealstarComposedData - attach real array data to a PetscObject

   Synopsis:
   PetscErrorCode PetscObjectSetRealstarComposedData(PetscObject obj,int id,PetscReal *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   Level: developer
M*/
#define PetscObjectSetRealstarComposedData(obj,id,data)                  \
0; {PetscErrorCode ierr_;                                                       \
  if ((obj)->realstar_idmax < globalmaxstate) {                          \
    ierr_ = PetscObjectIncreaseRealstarComposedData(obj); CHKERRQ(ierr_); \
  }                                                                  \
  (obj)->realstarcomposeddata[id] = data;                                \
  (obj)->realstarcomposedstate[id] = (obj)->state;                       \
}
/*MC
   PetscObjectGetRealstarComposedData - retrieve real array data
   attached to an object

   Synopsis:
   PetscErrorCode PetscObjectGetRealstarComposedData(PetscObject obj,int id,PetscReal **data,PetscTruth *flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectGetRealstarComposedData(obj,id,data,flag)        \
0; {                                                                \
  if ((obj)->realstarcomposedstate) {                               \
    if ((obj)->realstarcomposedstate[id] == (obj)->state) {         \
      data = (obj)->realstarcomposeddata[id];                       \
      flag = PETSC_TRUE;                                            \
    } else {                                                        \
      flag = PETSC_FALSE;                                           \
    }                                                               \
  } else {                                                          \
    flag = PETSC_FALSE;                                             \
  }                                                                 \
}

/*MC
   PetscObjectSetScalarComposedData - attach scalar data to a PetscObject 

   Synopsis:
   PetscErrorCode PetscObjectSetScalarComposedData(PetscObject obj,int id,PetscScalar data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectSetScalarComposedData(obj,id,data)                 \
0; {PetscErrorCode ierr_;                                                        \
  if ((obj)->scalar_idmax < globalmaxstate) {                         \
    ierr_ = PetscObjectIncreaseScalarComposedData(obj); CHKERRQ(ierr_);\
  }                                                                   \
  (obj)->scalarcomposeddata[id] = data;                               \
  (obj)->scalarcomposedstate[id] = (obj)->state;                      \
}
#else
#define PetscObjectSetScalarComposedData(obj,id,data) \
        PetscObjectSetRealComposedData(obj,id,data)
#endif
/*MC
   PetscObjectGetScalarComposedData - retrieve scalar data attached to an object

   Synopsis:
   PetscErrorCode PetscObjectGetScalarComposedData(PetscObject obj,int id,PetscScalar *data,PetscTruth *flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectGetScalarComposedData(obj,id,data,flag)           \
0; {                                                                 \
  if ((obj)->scalarcomposedstate) {				     \
    if ((obj)->scalarcomposedstate[id] == (obj)->state) {            \
      data = (obj)->scalarcomposeddata[id];                          \
      flag = PETSC_TRUE;                                             \
    } else {                                                         \
      flag = PETSC_FALSE;                                            \
    }                                                                \
  } else flag = PETSC_FALSE;                                         \
}
#else
#define PetscObjectGetScalarComposedData(obj,id,data,flag)	     \
        PetscObjectGetRealComposedData(obj,id,data,flag)
#endif

/*MC
   PetscObjectSetScalarstarComposedData - attach scalar array data to a PetscObject 

   Synopsis:
   PetscErrorCode PetscObjectSetScalarstarComposedData(PetscObject obj,int id,PetscScalar *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectSetScalarstarComposedData(obj,id,data)                   \
0; {PetscErrorCode ierr_;                                                              \
  if ((obj)->scalarstar_idmax < globalmaxstate) {                           \
    ierr_ = PetscObjectIncreaseScalarstarComposedData(obj); CHKERRQ(ierr_); \
  }                                                                         \
  (obj)->scalarstarcomposeddata[id] = data;                                 \
  (obj)->scalarstarcomposedstate[id] = (obj)->state;                        \
}
#else
#define PetscObjectSetScalarstarComposedData(obj,id,data) \
        PetscObjectSetRealstarComposedData(obj,id,data)
#endif
/*MC
   PetscObjectGetScalarstarComposedData - retrieve scalar array data
   attached to an object

   Synopsis:
   PetscErrorCode PetscObjectGetScalarstarComposedData(PetscObject obj,int id,PetscScalar **data,PetscTruth *flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   Notes:

   This routine does not return an error code; any errors are handled
   internally.

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectGetScalarstarComposedData(obj,id,data,flag)           \
0; {                                                                     \
  if ((obj)->scalarstarcomposedstate) {                                  \
    if ((obj)->scalarstarcomposedstate[id] == (obj)->state) {            \
      data = (obj)->scalarstarcomposeddata[id];                          \
      flag = PETSC_TRUE;                                                 \
    } else {                                                             \
      flag = PETSC_FALSE;                                                \
    }                                                                    \
  } else flag = PETSC_FALSE;                                             \
}
#else
#define PetscObjectGetScalarstarComposedData(obj,id,data,flag)	         \
        PetscObjectGetRealstarComposedData(obj,id,data,flag)
#endif

PETSC_EXTERN_CXX_END
#endif /* _PETSCHEAD_H */

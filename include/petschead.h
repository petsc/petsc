/* $Id: petschead.h,v 1.86 2001/09/07 20:13:16 bsmith Exp $ */

/*
    Defines the basic header of all PETSc objects.
*/

#if !defined(_PETSCHEAD_H)
#define _PETSCHEAD_H
#include "petsc.h"  
PETSC_EXTERN_CXX_BEGIN


EXTERN int PetscRegisterCookie(int *);

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
   int (*getcomm)(PetscObject,MPI_Comm *);
   int (*view)(PetscObject,PetscViewer);
   int (*reference)(PetscObject);
   int (*destroy)(PetscObject);
   int (*compose)(PetscObject,const char[],PetscObject);
   int (*query)(PetscObject,const char[],PetscObject *);
   int (*composefunction)(PetscObject,const char[],const char[],void (*)(void));
   int (*queryfunction)(PetscObject,const char[],void (**)(void));
   int (*composelanguage)(PetscObject,PetscLanguage,void *);
   int (*querylanguage)(PetscObject,PetscLanguage,void **);
   int (*publish)(PetscObject);
} PetscOps;

#define PETSCHEADER(ObjectOps)                            \
  int            cookie;                                  \
  PetscOps       *bops;                                   \
  ObjectOps      *ops;                                    \
  MPI_Comm       comm;                                    \
  int            type;                                    \
  PetscLogDouble flops,time,mem;                          \
  int            id;                                      \
  int            refct;                                   \
  int            tag;                                     \
  PetscFList     qlist;                                   \
  PetscOList     olist;                                   \
  char           *class_name;                             \
  char           *type_name;                              \
  ParameterDict  dict;                                    \
  PetscObject    parent;                                  \
  int            parentid;                                \
  char*          name;                                    \
  char           *prefix;                                 \
  void           *cpp;                                    \
  int            amem;                                    \
  int            state;                                   \
  int            int_idmax,real_idmax,scalar_idmax;       \
  int            *intcomposeddata,*intcomposedstate,      \
                 *realcomposedstate,*scalarcomposedstate; \
  PetscReal      *realcomposeddata;                       \
  PetscScalar    *scalarcomposeddata;                     \
  void           (**fortran_func_pointers)(void);       

  /*  ... */                               

#define  PETSCFREEDHEADER -1

EXTERN int PetscHeaderCreate_Private(PetscObject,int,int,const char[],MPI_Comm,int (*)(PetscObject),int (*)(PetscObject,PetscViewer));
EXTERN int PetscHeaderDestroy_Private(PetscObject);

typedef int (*PetscObjectFunction)(PetscObject); /* force cast in next macro to NEVER use extern "C" style */
typedef int (*PetscObjectViewerFunction)(PetscObject,PetscViewer); 

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
#define PetscHeaderCreate(h,tp,pops,cook,t,class_name,com,des,vie)                      \
  { int _ierr;                                                                          \
    _ierr = PetscNew(struct tp,&(h));CHKERRQ(_ierr);                                      \
    _ierr = PetscMemzero(h,sizeof(struct tp));CHKERRQ(_ierr);                           \
    _ierr = PetscNew(PetscOps,&((h)->bops));CHKERRQ(_ierr);                               \
    _ierr = PetscMemzero((h)->bops,sizeof(PetscOps));CHKERRQ(_ierr);                    \
    _ierr = PetscNew(pops,&((h)->ops));CHKERRQ(_ierr);                                    \
    _ierr = PetscMemzero((h)->ops,sizeof(pops));CHKERRQ(_ierr);                         \
    _ierr = PetscHeaderCreate_Private((PetscObject)h,cook,t,class_name,com,             \
                                 (PetscObjectFunction)des,                             \
                                 (PetscObjectViewerFunction)vie);CHKERRQ(_ierr);       \
  }

#define PetscHeaderDestroy(h)                                             \
  { int _ierr;                                                            \
    _ierr = PetscHeaderDestroy_Private((PetscObject)(h));CHKERRQ(_ierr);\
  }                 

/* ---------------------------------------------------------------------------------------*/

#if !defined(PETSC_HAVE_CRAY90_POINTER)
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
  {int _6_ierr,__flag; _6_ierr = MPI_Comm_compare(((PetscObject)a)->comm,((PetscObject)b)->comm,&__flag);\
  CHKERRQ(_6_ierr); \
  if (__flag != MPI_CONGRUENT && __flag != MPI_IDENT) \
  SETERRQ2(PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the two objects: Argument # %d and %d",arga,argb);}

#define PetscCheckSameTypeAndComm(a,arga,b,argb) {\
  PetscCheckSameType(a,arga,b,argb);\
  PetscCheckSameComm(a,arga,b,argb);}

/*
   All PETSc objects begin with the fields defined in PETSCHEADER.
   The PetscObject is a way of examining these fields regardless of 
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
struct _p_PetscObject {
  PETSCHEADER(int)
};

EXTERN int PetscObjectPublishBaseBegin(PetscObject);
EXTERN int PetscObjectPublishBaseEnd(PetscObject);

EXTERN int PetscObjectIncreaseState(PetscObject);
EXTERN int PetscObjectGetState(PetscObject obj,int*);
EXTERN int PetscRegisterComposedData(int *id);
EXTERN int PetscObjectIncreaseIntComposedData(PetscObject obj);
EXTERN int PetscObjectIncreaseRealComposedData(PetscObject obj);
EXTERN int PetscObjectIncreaseScalarComposedData(PetscObject obj);
EXTERN int globalcurrentstate,globalmaxstate;
/*MC
   PetscObjectSetIntComposedData - attach integer data to a PetscObject

   Synopsis:
   PetscObjectSetIntComposedData(PetscObject obj,int id,int data)

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
#define PetscObjectSetIntComposedData(obj,id,data)                   \
0; {int ierr_;                                                       \
  if ((obj)->int_idmax < globalmaxstate) {                           \
    ierr_ = PetscObjectIncreaseIntComposedData(obj); CHKERRQ(ierr_);  \
  }                                                                  \
  (obj)->intcomposeddata[id] = data;                                 \
  (obj)->intcomposedstate[id] = (obj)->state;                        \
}
/*MC
   PetscObjectGetIntComposedData - retrieve integer data attached to an object

   Synopsis:
   PetscObjectGetIntComposedData(PetscObject obj,int id,int *data,PetscTruth *flag)

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
  if ((int)((obj)->intcomposedstate)) {                              \
    if ((obj)->intcomposedstate[id] == (obj)->state) {               \
      data = (obj)->intcomposeddata[id];                             \
      flag = PETSC_TRUE;                                             \
    } else {                                                         \
      flag = PETSC_FALSE;                                            \
    }                                                                \
  } else flag = PETSC_FALSE;                                         \
}
/*MC
   PetscObjectSetRealComposedData - attach real data to a PetscObject

   Synopsis:
   PetscObjectSetRealComposedData(PetscObject obj,int id,PetscReal data)

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
0; {int ierr_;                                                       \
  if ((obj)->real_idmax < globalmaxstate) {                          \
    ierr_ = PetscObjectIncreaseRealComposedData(obj); CHKERRQ(ierr_); \
  }                                                                  \
  (obj)->realcomposeddata[id] = data;                                \
  (obj)->realcomposedstate[id] = (obj)->state;                       \
}
/*MC
   PetscObjectGetRealComposedData - retrieve real data attached to an object

   Synopsis:
   PetscObjectGetRealComposedData(PetscObject obj,int id,PetscReal *data,PetscTruth *flag)

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
  if (((obj)->realcomposedstate)) {                                  \
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
   PetscObjectSetScalarComposedData - attach scalar data to a PetscObject 

   Synopsis:
   PetscObjectSetScalarComposedData(PetscObject obj,int id,PetscScalar data)

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
#define PetscObjectSetScalarComposedData(obj,id,data)                 \
0; {int ierr_;                                                        \
  if ((obj)->scalar_idmax < globalmaxstate) {                         \
    ierr_ = PetscObjectIncreaseScalarComposedData(obj); CHKERRQ(ierr_);\
  }                                                                   \
  (obj)->scalarcomposeddata[id] = data;                               \
  (obj)->scalarcomposedstate[id] = (obj)->state;                      \
}
/*MC
   PetscObjectGetScalarComposedData - retrieve scalar data attached to an object

   Synopsis:
   PetscObjectGetScalarComposedData(PetscObject obj,int id,PetscScalar *data,PetscTruth *flag)

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
#define PetscObjectGetScalarComposedData(obj,id,data,flag)           \
0; {                                                                 \
  if ((int)((obj)->scalarcomposedstate)) {                           \
    if ((obj)->scalarcomposedstate[id] == (obj)->state) {            \
      data = (obj)->scalarcomposeddata[id];                          \
      flag = PETSC_TRUE;                                             \
    } else {                                                         \
      flag = PETSC_FALSE;                                            \
    }                                                                \
  } else flag = PETSC_FALSE;                                         \
}

PETSC_EXTERN_CXX_END
#endif /* _PETSCHEAD_H */

/* $Id: petschead.h,v 1.86 2001/09/07 20:13:16 bsmith Exp $ */

/*
    Defines the basic header of all PETSc objects.
*/

#if !defined(_PETSCHEAD_H)
#define _PETSCHEAD_H
#include "petsc.h"  
PETSC_EXTERN_CXX_BEGIN

EXTERN int PetscCommDuplicate_Private(MPI_Comm,MPI_Comm*,int*);
EXTERN int PetscCommDestroy_Private(MPI_Comm*);

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
  char           *serialize_name;                         \
  ParameterDict  dict;                                    \
  PetscObject    parent;                                  \
  int            parentid;                                \
  char*          name;                                    \
  char           *prefix;                                 \
  void           *cpp;                                    \
  int            amem;                                    \
  void           (**fortran_func_pointers)(void);       

  /*  ... */                               

#define  PETSCFREEDHEADER -1

EXTERN int PetscHeaderCreate_Private(PetscObject,int,int,char *,MPI_Comm,int (*)(PetscObject),int (*)(PetscObject,PetscViewer));
EXTERN int PetscHeaderDestroy_Private(PetscObject);

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
                                 (int (*)(PetscObject))des,                             \
                                 (int (*)(PetscObject,PetscViewer))vie);CHKERRQ(_ierr); \
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
#define PetscValidHeaderSpecific(h,ck)                              \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null Object");}        \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object");   \
  }                                                                 \
  if (((PetscObject)(h))->cookie != ck) {                           \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {           \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");       \
    } else {                                                        \
      SETERRQ(PETSC_ERR_ARG_WRONG,"Wrong Object");                \
    }                                                               \
  }} 

#define PetscValidHeader(h)                                         \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null Object");}        \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object");   \
  } else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {      \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");       \
  } else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||           \
      ((PetscObject)(h))->cookie > PETSC_LARGEST_COOKIE) {          \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Object");            \
  }}

#define PetscValidPointer(h)                                        \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  if ((unsigned long)h & (unsigned long)3){                         \
    SETERRQ(PETSC_ERR_ARG_BADPTR,"Invalid Pointer");              \
  }}

#define PetscValidCharPointer(h)                                    \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  }

#define PetscValidIntPointer(h)                                     \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  if ((unsigned long)h & (unsigned long)3){                         \
    SETERRQ(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to Int");       \
  }}

#if !defined(PETSC_HAVE_DOUBLES_ALIGNED) || defined (PETSC_HAVE_DOUBLES_ALIGNED)
#define PetscValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscScalar");    \
  }}
#else
#define PetscValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  if ((unsigned long)h & (unsigned long)7) {                        \
    SETERRQ(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscScalar");    \
  }}
#endif

#else
/*
     Version for Cray 90 that handles pointers differently
*/
#define PetscValidHeaderSpecific(h,ck)                              \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null Object");}        \
  if (((PetscObject)(h))->cookie != ck) {                           \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {           \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");       \
    } else {                                                        \
      SETERRQ(PETSC_ERR_ARG_WRONG,"Wrong Object");                \
    }                                                               \
  }} 

#define PetscValidHeader(h)                                         \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null Object");}        \
  if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {      \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");       \
  } else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||           \
      ((PetscObject)(h))->cookie > PETSC_LARGEST_COOKIE) {          \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Object");            \
  }}

#define PetscValidPointer(h)                                        \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  }

#define PetscValidCharPointer(h)                                    \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  }

#define PetscValidIntPointer(h)                                     \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  }

#if !defined(PETSC_HAVE_DOUBLES_ALIGNED)
#define PetscValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  }
#else
#define PetscValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}        \
  }
#endif

#endif
#define PetscValidDoublePointer(h) PetscValidScalarPointer(h)

/*
    For example, in the dot product between two vectors,
  both vectors must be either Seq or MPI, not one of each 
*/
#define PetscCheckSameType(a,b) \
  if ((a)->type != (b)->type) SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,"Objects not of same type");
/* 
   Use this macro to check if the type is set
*/
#define PetscValidType(a) \
  if (!(a)->type_name) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Object Type not set");
/*
   Sometimes object must live on same communicator to inter-operate
*/
#define PetscCheckSameComm(a,b) \
  {int _6_ierr,__flag; _6_ierr = MPI_Comm_compare(((PetscObject)a)->comm,((PetscObject)b)->comm,&__flag);\
  CHKERRQ(_6_ierr); \
  if (__flag != MPI_CONGRUENT && __flag != MPI_IDENT) \
  SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the two objects");}

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

PETSC_EXTERN_CXX_END
#endif /* _PETSCHEAD_H */

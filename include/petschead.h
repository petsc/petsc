/* $Id: petschead.h,v 1.63 1998/05/20 18:17:29 bsmith Exp bsmith $ */

/*
    Defines the basic header of all PETSc objects.
*/

#if !defined(_PHEAD_H)
#define _PHEAD_H
#include "petsc.h"  

extern int PetscCommDup_Private(MPI_Comm,MPI_Comm*,int*);
extern int PetscCommFree_Private(MPI_Comm*);

extern int PetscRegisterCookie(int *);

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
                          for example, MatDestroy() is the general matrix 
                          destruction routine.
      query()           - Returns a different PETSc object that has been associated
                          with the first object.
      compose()         - Associates a PETSc object with another PETSc object.
      composefunction() - Attaches an additional registered function.
      reference()       - Increases the reference count for a PETSc object; when
                          a reference count reaches zero it is destroyed.
      queryfunction()   - Requests a registered function that has been registered.
*/

typedef struct {
   int (*getcomm)(PetscObject,MPI_Comm *);
   int (*view)(PetscObject,Viewer);
   int (*reference)(PetscObject);
   int (*destroy)(PetscObject);
   int (*compose)(PetscObject,char*,PetscObject);
   int (*query)(PetscObject,char *,PetscObject *);
   int (*composefunction)(PetscObject,char *,char *,void *);
   int (*queryfunction)(PetscObject,char *, void **);
   int (*composelanguage)(PetscObject,PetscLanguage,void *);
   int (*querylanguage)(PetscObject,PetscLanguage,void **);
} PetscOps;

#define PETSCHEADER(ObjectOps)                         \
  int         cookie;                                  \
  PetscOps    *bops;                                   \
  ObjectOps   *ops;                                    \
  MPI_Comm    comm;                                    \
  int         type;                                    \
  PLogDouble  flops,time,mem;                          \
  int         id;                                      \
  int         refct;                                   \
  int         tag;                                     \
  DLList      qlist;                                   \
  OList       olist;                                   \
  char        *type_name;                              \
  PetscObject parent;                                  \
  char*       name;                                    \
  char        *prefix;                                 \
  void        *cpp;                                    \
  void**      fortran_func_pointers;       

  /*  ... */                               

#define  PETSCFREEDHEADER -1

extern int PetscHeaderCreate_Private(PetscObject,int,int,MPI_Comm,int (*)(PetscObject),int (*)(PetscObject,Viewer));
extern int PetscHeaderDestroy_Private(PetscObject);

/*
    PetscHeaderCreate - Creates a PETSc object

    Input Parameters:
+   tp - the data structure type of the object
.   pops - the data structure type of the objects operations (for example _VecOps)
.   cook - the cookie associated with this object
.   t - type (no longer should be used)
.   com - the MPI Communicator
.   des - the destroy routine for this object
-   vie - the view routine for this object

    Output Parameter:
.   h - the newly created object
*/ 
#define PetscHeaderCreate(h,tp,pops,cook,t,com,des,vie)                                     \
  { int _ierr;                                                                              \
    h       = PetscNew(struct tp);CHKPTRQ((h));                                             \
    PetscMemzero(h,sizeof(struct tp));                                                      \
    (h)->bops = PetscNew(PetscOps);CHKPTRQ(((h)->bops));                                    \
    PetscMemzero((h)->bops,sizeof(sizeof(PetscOps)));                                       \
    (h)->ops  = PetscNew(pops);CHKPTRQ(((h)->ops));                                         \
    _ierr = PetscHeaderCreate_Private((PetscObject)h,cook,t,com,                            \
                                       (int (*)(PetscObject))des,                           \
                                       (int (*)(PetscObject,Viewer))vie); CHKERRQ(_ierr);   \
  }

#define PetscHeaderDestroy(h)                                             \
  { int _ierr;                                                            \
    _ierr = PetscHeaderDestroy_Private((PetscObject) (h)); CHKERRQ(_ierr);\
  }                 

/* ---------------------------------------------------------------------------------------*/

/* 
     PetscLow and PetscHigh are a way of checking whether an address is 
  out of range.  These are set in src/sys/src/memory/tr.c

     The macro USE_TRMALLOC_RANGE must be turned on to enable this 
  feature. We seem to have some problems with this on the DEC alpha
  hence it is currently turned off.

*/
extern void *PetscLow,*PetscHigh;

#if defined(USE_TRMALLOC_RANGE)
#define PetscValidHeaderSpecific(h,ck)                                        \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null Object");}                  \
  if ((unsigned long)h & (unsigned long)3) {                                  \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Invalid Pointer to Object");             \
  }                                                                           \
  if (PetscLow > (void *) h || PetscHigh < (void *)h){                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Invalid Pointer to Object:out of range");\
  }                                                                           \
  if (((PetscObject)(h))->cookie != ck) {                                     \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {                     \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Object already free");                 \
    } else {                                                                    \
      SETERRQ(PETSC_ERR_ARG_WRONG,0,"Wrong Object");                          \
    }                                                                         \
  }}
#define PetscValidHeader(h)                                                   \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null Object");}                  \
  if ((unsigned long)h & (unsigned long)3) {                                  \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Invalid Pointer to Object");             \
  } else if (PetscLow > (void *) h || PetscHigh < (void *)h){                 \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Invalid Pointer to Object:out of range");\
  } else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {                \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Object already free");                 \
  } else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||                     \
      ((PetscObject)(h))->cookie > LARGEST_PETSC_COOKIE) {                    \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Invalid Object");                      \
  }}
#else
#define PetscValidHeaderSpecific(h,ck)                              \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null Object");}        \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Invalid Pointer to Object");   \
  }                                                                 \
  if (((PetscObject)(h))->cookie != ck) {                           \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {           \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Object already free");       \
    } else {                                                        \
      SETERRQ(PETSC_ERR_ARG_WRONG,0,"Wrong Object");                \
    }                                                               \
  }} 
#define PetscValidHeader(h)                                         \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null Object");}        \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Invalid Pointer to Object");   \
  } else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {      \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Object already free");       \
  } else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||           \
      ((PetscObject)(h))->cookie > LARGEST_PETSC_COOKIE) {          \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Invalid Object");            \
  }}
#endif

#define PetscValidIntPointer(h)                                     \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Null Pointer");}        \
  if ((unsigned long)h & (unsigned long)3){                         \
    SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Invalid Pointer to Int");       \
  }}
#define PetscValidPointer(h)                                        \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Null Pointer");}        \
  if ((unsigned long)h & (unsigned long)3){                         \
    SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Invalid Pointer");              \
  }}

/*
   Some machines do not double align doubles 
*/
#if !defined(HAVE_DOUBLE_ALIGN)
#define PetscValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Null Pointer");}        \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Invalid Pointer to Scalar");    \
  }}
#else
#define PetscValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Null Pointer");}        \
  if ((unsigned long)h & (unsigned long)7) {                        \
    SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Invalid Pointer to Scalar");    \
  }}
#endif

/*
    For example, in the dot product between two vectors,
  both vectors must be either Seq or MPI, not one of each 
*/
#define PetscCheckSameType(a,b) \
  if ((a)->type != (b)->type) SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,0,"Objects not of same type");

/*
   All PETSc objects begin with the fields defined in PETSCHEADER.
   The PetscObject is a way of examining these fields regardless of 
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
struct _p_PetscObject {
  PETSCHEADER(int)
};

extern int PetscObjectSetOptionsPrefix(PetscObject,char*);
extern int PetscObjectAppendOptionsPrefix(PetscObject,char*);
extern int PetscObjectGetOptionsPrefix(PetscObject,char**);

#endif



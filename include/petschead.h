
/*
    Defines the basic format of all data types. 
*/

#if !defined(_PETSCIMPL)
#define _PETSCIMPL
#include "petsc.h"  

#include <stdio.h>


/*
     All Major PETSc Data structures have a common core; this 
   is defined below by PETSCHEADER. 

     CREATEHEADER should be used whenever you create a PETSc structure.

     CHKSAME checks if you PETSc structures are of same type.

*/

#define PETSCHEADER                   \
  int    cookie;                      \
  int    type;                        \
  int    (*destroy)(PetscObject);     \
  int    (*view)(PetscObject,Viewer); \
  /*  ... */                          \

#define CREATEHEADER(h,tp)                           \
      {h = (struct tp *) NEW(struct tp);           \
       CHKPTR((h));                                  \
       (h)->cookie = 0; (h)->type = 0;               \
       (h)->destroy = (int (*)(PetscObject)) 0; \
       (h)->view = (int (*)(PetscObject,Viewer)) 0;}

#define FREEDHEADER -1

#define VALIDHEADER(h,ck)                  \
  {if (!h) {SETERR(1,"Null Object");}      \
  if ((h)->cookie != ck) {                 \
    if ((h)->cookie == FREEDHEADER) {      \
      SETERR(1,"Object already free");     \
    }                                      \
    else {                                 \
      SETERR(2,"Invalid or Wrong Object"); \
    }                                      \
  }}

#define CHKSAME(a,b) \
  if ((a)->type != (b)->type) SETERR(3,"Objects not of same type");

#define CHKTYPE(a,b) (((a)->type & (b)) ? 1 : 0)

struct _PetscObject {
  PETSCHEADER
};


#endif

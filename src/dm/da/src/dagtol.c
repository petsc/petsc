#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dagtol.c,v 1.17 1999/03/17 23:25:10 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAGlobalToLocalBegin"
/*@
   DAGlobalToLocalBegin - Maps values from the global vector to the local
   patch; the ghost points are included. Must be followed by 
   DAGlobalToLocalEnd() to complete the exchange.

   Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local values

   Level: beginner

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToLocalEnd(), DALocalToGlobal(), DACreate2d()
@*/
int DAGlobalToLocalBegin(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(l,VEC_COOKIE);
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  ierr = VecScatterBegin(g,l,mode,SCATTER_FORWARD,da->gtol); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGlobalToLocalEnd"
/*@
   DAGlobalToLocalEnd - Maps values from the global vector to the local
   patch; the ghost points are included. Must be preceeded by 
   DAGlobalToLocalBegin().

   Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local values

   Level: beginner

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, end

.seealso: DAGlobalToLocalBegin(), DALocalToGlobal(), DACreate2d()
@*/
int DAGlobalToLocalEnd(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(l,VEC_COOKIE);
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTER_FORWARD,da->gtol); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGlobalToNatural_Create"
/*
   DAGlobalToNatural_Create - Create the global to natural scatter object

   Collective on DA

   Input Parameter:
.  da - the distributed array context

   Level: developer

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
*/
int DAGlobalToNatural_Create(DA da)
{
  int ierr,m,start;
  IS  from,to;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (!da->natural) {
    SETERRQ(1,1,"Natural layout vector not yet created; cannot scatter into it");
  }

  /* create the scatter context */
  ierr = VecGetLocalSize(da->natural,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(da->natural,&start,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(da->comm,m,start,1,&to);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(da->ao,to);CHKERRQ(ierr);
  ierr = ISCreateStride(da->comm,m,start,1,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(da->global,from,da->natural,to,&da->gton);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGlobalToNaturalBegin"
/*@
   DAGlobalToNaturalBegin - Maps values from the global vector to a global vector
   in the "natural" grid ordering. Must be followed by 
   DAGlobalToNaturalEnd() to complete the exchange.

   Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the natural ordering values

   Level: advanced

   Notes:
   The global and natrual vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
@*/
int DAGlobalToNaturalBegin(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(l,VEC_COOKIE);
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  if (!da->gton) {
    /* create the scatter context */
    ierr = DAGlobalToNatural_Create(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(g,l,mode,SCATTER_FORWARD,da->gton); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGlobalToNaturalEnd"
/*@
   DAGlobalToNaturalEnd - Maps values from the global vector to a global vector
   in the natural ordering. Must be preceeded by DAGlobalToNaturalBegin().

   Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the global values in the natural ordering

   Level: advanced

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, end

.seealso: DAGlobalToNaturalBegin(), DALocalToGlobal(), DACreate2d(),
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
@*/
int DAGlobalToNaturalEnd(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(l,VEC_COOKIE);
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTER_FORWARD,da->gton); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DANaturalToGlobalBegin"
/*@
   DANaturalToGlobalBegin - Maps values from a global vector in the "natural" ordering 
   to a global vector in the PETSc DA grid ordering. Must be followed by 
   DANaturalToGlobalEnd() to complete the exchange.

   Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector in a natural ordering
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the values in the DA ordering

   Level: advanced

   Notes:
   The global and natrual vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToNaturalEnd(), DAGlobalToNaturalBegin(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
@*/
int DANaturalToGlobalBegin(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(l,VEC_COOKIE);
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  if (!da->gton) {
    /* create the scatter context */
    ierr = DAGlobalToNatural_Create(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(g,l,mode,SCATTER_REVERSE,da->gton); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DANaturalToGlobalEnd"
/*@
   DANaturalToGlobalEnd - Maps values from the natural ordering global vector 
   to a global vector in the PETSc DA ordering. Must be preceeded by DANaturalToGlobalBegin().

   Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector in a natural ordering
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the global values in the PETSc DA ordering

   Level: intermediate

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, end

.seealso: DAGlobalToNaturalBegin(), DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(),
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
@*/
int DANaturalToGlobalEnd(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(l,VEC_COOKIE);
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTER_REVERSE,da->gton); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAGlobalToLocalBegin"
/*@
   DAGlobalToLocalBegin - Maps values from the global vector to the local
   patch; the ghost points are included. Must be followed by 
   DAGlobalToLocalEnd() to complete the exchange.

   Neighbor-wise Collective on DA

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

.seealso: DAGlobalToLocalEnd(), DALocalToGlobal(), DACreate2d(), 
          DALocalToLocalBegin(), DALocalToLocalEnd(),
          DALocalToGlobalBegin(), DALocalToGlobalEnd()
          

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGlobalToLocalBegin(DA da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  PetscValidHeaderSpecific(l,VEC_CLASSID,4);
  ierr = VecScatterBegin(dd->gtol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DALocalToGlobalBegin"
/*@
   DALocalToGlobalBegin - Adds values from the local (ghosted) vector
   into the global (nonghosted) vector.

   Neighbor-wise Collective on DA

   Input Parameters:
+  da - the distributed array context
-  l  - the local values

   Output Parameter:
.  g - the global vector

   Level: beginner

   Notes:
   Use DALocalToGlobal() to discard the ghost point values

   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToLocalEnd(), DALocalToGlobal(), DACreate2d(), 
          DALocalToLocalBegin(), DALocalToLocalEnd(), DALocalToGlobalEnd()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DALocalToGlobalBegin(DA da,Vec l,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  ierr = VecScatterBegin(dd->gtol,l,g,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DALocalToGlobalEnd"
/*@
   DALocalToGlobalEnd - Adds values from the local (ghosted) vector
   into the global (nonghosted) vector.

   Neighbor-wise Collective on DA

   Input Parameters:
+  da - the distributed array context
-  l  - the local values

   Output Parameter:
.  g - the global vector

   Level: beginner

   Notes:
   Use DALocalToGlobal() to discard the ghost point values

   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToLocalEnd(), DALocalToGlobal(), DACreate2d(), 
          DALocalToLocalBegin(), DALocalToLocalEnd(), DALocalToGlobalBegin()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DALocalToGlobalEnd(DA da,Vec l,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  ierr = VecScatterEnd(dd->gtol,l,g,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGlobalToLocalEnd"
/*@
   DAGlobalToLocalEnd - Maps values from the global vector to the local
   patch; the ghost points are included. Must be preceeded by 
   DAGlobalToLocalBegin().

   Neighbor-wise Collective on DA

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

.seealso: DAGlobalToLocalBegin(), DALocalToGlobal(), DACreate2d(),
     DALocalToLocalBegin(), DALocalToLocalEnd(), DALocalToGlobalBegin(), DALocalToGlobalEnd()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGlobalToLocalEnd(DA da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  PetscValidHeaderSpecific(l,VEC_CLASSID,4);
  ierr = VecScatterEnd(dd->gtol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode DAGetNatural_Private(DA,PetscInt*,IS*);
#undef __FUNCT__  
#define __FUNCT__ "DAGlobalToNatural_Create"
/*
   DAGlobalToNatural_Create - Create the global to natural scatter object

   Collective on DA

   Input Parameter:
.  da - the distributed array context

   Level: developer

   Notes: This is an internal routine called by DAGlobalToNatural() to 
     create the scatter context.

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToNaturalBegin(), DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
*/
PetscErrorCode DAGlobalToNatural_Create(DA da)
{
  PetscErrorCode ierr;
  PetscInt       m,start,Nlocal;
  IS             from,to;
  Vec            global;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (!dd->natural) {
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ORDER,"Natural layout vector not yet created; cannot scatter into it");
  }

  /* create the scatter context */
  ierr = VecGetLocalSize(dd->natural,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(dd->natural,&start,PETSC_NULL);CHKERRQ(ierr);

  ierr = DAGetNatural_Private(da,&Nlocal,&to);CHKERRQ(ierr);
  if (Nlocal != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal error: Nlocal %D local vector size %D",Nlocal,m);
  ierr = ISCreateStride(((PetscObject)da)->comm,m,start,1,&from);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)da)->comm,dd->Nlocal,PETSC_DETERMINE,0,&global);
  ierr = VecSetBlockSize(global,dd->w);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,dd->natural,to,&dd->gton);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGlobalToNaturalBegin"
/*@
   DAGlobalToNaturalBegin - Maps values from the global vector to a global vector
   in the "natural" grid ordering. Must be followed by 
   DAGlobalToNaturalEnd() to complete the exchange.

   Neighbor-wise Collective on DA

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

   You must call DACreateNaturalVector() before using this routine

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector(),
          DALocalToGlobalBegin(), DALocalToGlobalEnd()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGlobalToNaturalBegin(DA da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  if (!dd->gton) {
    /* create the scatter context */
    ierr = DAGlobalToNatural_Create(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(dd->gton,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGlobalToNaturalEnd"
/*@
   DAGlobalToNaturalEnd - Maps values from the global vector to a global vector
   in the natural ordering. Must be preceeded by DAGlobalToNaturalBegin().

   Neighbor-wise Collective on DA

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
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector(),
          DALocalToGlobalBegin(), DALocalToGlobalEnd()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGlobalToNaturalEnd(DA da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  ierr = VecScatterEnd(dd->gton,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DANaturalToGlobalBegin"
/*@
   DANaturalToGlobalBegin - Maps values from a global vector in the "natural" ordering 
   to a global vector in the PETSc DA grid ordering. Must be followed by 
   DANaturalToGlobalEnd() to complete the exchange.

   Neighbor-wise Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector in a natural ordering
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the values in the DA ordering

   Level: advanced

   Notes:
   The global and natural vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToNaturalEnd(), DAGlobalToNaturalBegin(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector(),
          DALocalToGlobalBegin(), DALocalToGlobalEnd()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DANaturalToGlobalBegin(DA da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  if (!dd->gton) {
    /* create the scatter context */
    ierr = DAGlobalToNatural_Create(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(dd->gton,g,l,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DANaturalToGlobalEnd"
/*@
   DANaturalToGlobalEnd - Maps values from the natural ordering global vector 
   to a global vector in the PETSc DA ordering. Must be preceeded by DANaturalToGlobalBegin().

   Neighbor-wise Collective on DA

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
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector(),
          DALocalToGlobalBegin(), DALocalToGlobalEnd()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DANaturalToGlobalEnd(DA da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  ierr = VecScatterEnd(dd->gton,g,l,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


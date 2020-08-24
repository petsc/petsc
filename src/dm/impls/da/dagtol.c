/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

PetscErrorCode  DMGlobalToLocalBegin_DA(DM da,Vec g,InsertMode mode,Vec l)
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

PetscErrorCode  DMGlobalToLocalEnd_DA(DM da,Vec g,InsertMode mode,Vec l)
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

PetscErrorCode  DMLocalToGlobalBegin_DA(DM da,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  if (mode == ADD_VALUES) {
    ierr = VecScatterBegin(dd->gtol,l,g,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (mode == INSERT_VALUES) {
    if (dd->bx != DM_BOUNDARY_GHOSTED && dd->bx != DM_BOUNDARY_NONE && dd->s > 0 && dd->m == 1) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Available only for boundary none or with parallelism in x direction");
    if (dd->bx != DM_BOUNDARY_GHOSTED && dd->by != DM_BOUNDARY_NONE && dd->s > 0 && dd->n == 1) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Available only for boundary none or with parallelism in y direction");
    if (dd->bx != DM_BOUNDARY_GHOSTED && dd->bz != DM_BOUNDARY_NONE && dd->s > 0 && dd->p == 1) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Available only for boundary none or with parallelism in z direction");
    ierr = VecScatterBegin(dd->gtol,l,g,INSERT_VALUES,SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Not yet implemented");
  PetscFunctionReturn(0);
}

PetscErrorCode  DMLocalToGlobalEnd_DA(DM da,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  if (mode == ADD_VALUES) {
    ierr = VecScatterEnd(dd->gtol,l,g,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (mode == INSERT_VALUES) {
    ierr = VecScatterEnd(dd->gtol,l,g,INSERT_VALUES,SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Not yet implemented");
  PetscFunctionReturn(0);
}

extern PetscErrorCode DMDAGetNatural_Private(DM,PetscInt*,IS*);
/*
   DMDAGlobalToNatural_Create - Create the global to natural scatter object

   Collective on da

   Input Parameter:
.  da - the distributed array context

   Level: developer

   Notes:
    This is an internal routine called by DMDAGlobalToNatural() to
     create the scatter context.

.seealso: DMDAGlobalToNaturalBegin(), DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()
*/
PetscErrorCode DMDAGlobalToNatural_Create(DM da)
{
  PetscErrorCode ierr;
  PetscInt       m,start,Nlocal;
  IS             from,to;
  Vec            global;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (!dd->natural) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ORDER,"Natural layout vector not yet created; cannot scatter into it");

  /* create the scatter context */
  ierr = VecGetLocalSize(dd->natural,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(dd->natural,&start,NULL);CHKERRQ(ierr);

  ierr = DMDAGetNatural_Private(da,&Nlocal,&to);CHKERRQ(ierr);
  if (Nlocal != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal error: Nlocal %D local vector size %D",Nlocal,m);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)da),m,start,1,&from);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)da),dd->w,dd->Nlocal,PETSC_DETERMINE,NULL,&global);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,dd->natural,to,&dd->gton);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMDAGlobalToNaturalBegin - Maps values from the global vector to a global vector
   in the "natural" grid ordering. Must be followed by
   DMDAGlobalToNaturalEnd() to complete the exchange.

   Neighbor-wise Collective on da

   Input Parameters:
+  da - the distributed array context
.  g - the global vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the natural ordering values

   Level: advanced

   Notes:
   The global and natrual vectors used here need not be the same as those
   obtained from DMCreateGlobalVector() and DMDACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

   You must call DMDACreateNaturalVector() before using this routine

.seealso: DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()

@*/
PetscErrorCode  DMDAGlobalToNaturalBegin(DM da,Vec g,InsertMode mode,Vec n)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  PetscValidHeaderSpecific(n,VEC_CLASSID,4);
  if (!dd->gton) {
    /* create the scatter context */
    ierr = DMDAGlobalToNatural_Create(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(dd->gton,g,n,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMDAGlobalToNaturalEnd - Maps values from the global vector to a global vector
   in the natural ordering. Must be preceeded by DMDAGlobalToNaturalBegin().

   Neighbor-wise Collective on da

   Input Parameters:
+  da - the distributed array context
.  g - the global vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the global values in the natural ordering

   Level: advanced

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DMCreateGlobalVector() and DMDACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

.seealso: DMDAGlobalToNaturalBegin(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()

@*/
PetscErrorCode  DMDAGlobalToNaturalEnd(DM da,Vec g,InsertMode mode,Vec n)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  PetscValidHeaderSpecific(n,VEC_CLASSID,4);
  ierr = VecScatterEnd(dd->gton,g,n,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMDANaturalToGlobalBegin - Maps values from a global vector in the "natural" ordering
   to a global vector in the PETSc DMDA grid ordering. Must be followed by
   DMDANaturalToGlobalEnd() to complete the exchange.

   Neighbor-wise Collective on da

   Input Parameters:
+  da - the distributed array context
.  g - the global vector in a natural ordering
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the values in the DMDA ordering

   Level: advanced

   Notes:
   The global and natural vectors used here need not be the same as those
   obtained from DMCreateGlobalVector() and DMDACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

.seealso: DMDAGlobalToNaturalEnd(), DMDAGlobalToNaturalBegin(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()

@*/
PetscErrorCode  DMDANaturalToGlobalBegin(DM da,Vec n,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(n,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  if (!dd->gton) {
    /* create the scatter context */
    ierr = DMDAGlobalToNatural_Create(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(dd->gton,n,g,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMDANaturalToGlobalEnd - Maps values from the natural ordering global vector
   to a global vector in the PETSc DMDA ordering. Must be preceeded by DMDANaturalToGlobalBegin().

   Neighbor-wise Collective on da

   Input Parameters:
+  da - the distributed array context
.  g - the global vector in a natural ordering
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the global values in the PETSc DMDA ordering

   Level: advanced

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DMCreateGlobalVector() and DMDACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

.seealso: DMDAGlobalToNaturalBegin(), DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()

@*/
PetscErrorCode  DMDANaturalToGlobalEnd(DM da,Vec n,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(n,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  ierr = VecScatterEnd(dd->gton,n,g,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

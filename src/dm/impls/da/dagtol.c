
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_DA"
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


#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalEnd_DA"
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

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalBegin_DA"
PetscErrorCode  DMLocalToGlobalBegin_DA(DM da,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  if (mode == ADD_VALUES) {
    ierr = VecScatterBegin(dd->gtol,l,g,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (mode == INSERT_VALUES) {
    ierr = VecScatterBegin(dd->ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"Not yet implemented");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalEnd_DA"
PetscErrorCode  DMLocalToGlobalEnd_DA(DM da,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  if (mode == ADD_VALUES) {
    ierr = VecScatterEnd(dd->gtol,l,g,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (mode == INSERT_VALUES) {
    ierr = VecScatterEnd(dd->ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"Not yet implemented");
  PetscFunctionReturn(0);
}

extern PetscErrorCode DMDAGetNatural_Private(DM,PetscInt*,IS*);
#undef __FUNCT__
#define __FUNCT__ "DMDAGlobalToNatural_Create"
/*
   DMDAGlobalToNatural_Create - Create the global to natural scatter object

   Collective on DMDA

   Input Parameter:
.  da - the distributed array context

   Level: developer

   Notes: This is an internal routine called by DMDAGlobalToNatural() to
     create the scatter context.

.keywords: distributed array, global to local, begin

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
  if (!dd->natural) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ORDER,"Natural layout vector not yet created; cannot scatter into it");

  /* create the scatter context */
  ierr = VecGetLocalSize(dd->natural,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(dd->natural,&start,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMDAGetNatural_Private(da,&Nlocal,&to);CHKERRQ(ierr);
  if (Nlocal != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal error: Nlocal %D local vector size %D",Nlocal,m);
  ierr = ISCreateStride(((PetscObject)da)->comm,m,start,1,&from);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)da)->comm,dd->w,dd->Nlocal,PETSC_DETERMINE,0,&global);
  ierr = VecScatterCreate(global,from,dd->natural,to,&dd->gton);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGlobalToNaturalBegin"
/*@
   DMDAGlobalToNaturalBegin - Maps values from the global vector to a global vector
   in the "natural" grid ordering. Must be followed by
   DMDAGlobalToNaturalEnd() to complete the exchange.

   Neighbor-wise Collective on DMDA

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

.keywords: distributed array, global to local, begin

.seealso: DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()

@*/
PetscErrorCode  DMDAGlobalToNaturalBegin(DM da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  if (!dd->gton) {
    /* create the scatter context */
    ierr = DMDAGlobalToNatural_Create(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(dd->gton,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGlobalToNaturalEnd"
/*@
   DMDAGlobalToNaturalEnd - Maps values from the global vector to a global vector
   in the natural ordering. Must be preceeded by DMDAGlobalToNaturalBegin().

   Neighbor-wise Collective on DMDA

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

.keywords: distributed array, global to local, end

.seealso: DMDAGlobalToNaturalBegin(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()

@*/
PetscErrorCode  DMDAGlobalToNaturalEnd(DM da,Vec g,InsertMode mode,Vec l)
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
#define __FUNCT__ "DMDANaturalToGlobalBegin"
/*@
   DMDANaturalToGlobalBegin - Maps values from a global vector in the "natural" ordering
   to a global vector in the PETSc DMDA grid ordering. Must be followed by
   DMDANaturalToGlobalEnd() to complete the exchange.

   Neighbor-wise Collective on DMDA

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

.keywords: distributed array, global to local, begin

.seealso: DMDAGlobalToNaturalEnd(), DMDAGlobalToNaturalBegin(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()

@*/
PetscErrorCode  DMDANaturalToGlobalBegin(DM da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(l,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  if (!dd->gton) {
    /* create the scatter context */
    ierr = DMDAGlobalToNatural_Create(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(dd->gton,g,l,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDANaturalToGlobalEnd"
/*@
   DMDANaturalToGlobalEnd - Maps values from the natural ordering global vector
   to a global vector in the PETSc DMDA ordering. Must be preceeded by DMDANaturalToGlobalBegin().

   Neighbor-wise Collective on DMDA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector in a natural ordering
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the global values in the PETSc DMDA ordering

   Level: intermediate

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DMCreateGlobalVector() and DMDACreateNaturalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

.keywords: distributed array, global to local, end

.seealso: DMDAGlobalToNaturalBegin(), DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()

@*/
PetscErrorCode  DMDANaturalToGlobalEnd(DM da,Vec g,InsertMode mode,Vec l)
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


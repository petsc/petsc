#include <petsc-private/dmimpl.h> /*I "petscdm.h" I*/

#undef __FUNCT__
#define __FUNCT__ "DMGetLocalVector"
/*@
   DMGetLocalVector - Gets a Seq PETSc vector that
   may be used with the DMXXX routines. This vector has spaces for the ghost values.

   Not Collective

   Input Parameter:
.  dm - the distributed array

   Output Parameter:
.  g - the local vector

   Level: beginner

   Note:
   The vector values are NOT initialized and may have garbage in them, so you may need
   to zero them.

   The output parameter, g, is a regular PETSc vector that should be returned with
   DMRestoreLocalVector() DO NOT call VecDestroy() on it.

   VecStride*() operations can be useful when using DM with dof > 1

.keywords: distributed array, create, local, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMCreateLocalVector(), DMRestoreLocalVector(),
          VecStrideMax(), VecStrideMin(), VecStrideNorm()
@*/
PetscErrorCode  DMGetLocalVector(DM dm,Vec* g)
{
  PetscErrorCode ierr,i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->localin[i]) {
      *g             = dm->localin[i];
      dm->localin[i] = PETSC_NULL;
      goto alldone;
    }
  }
  ierr = DMCreateLocalVector(dm,g);CHKERRQ(ierr);

  alldone:
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (!dm->localout[i]) {
      dm->localout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestoreLocalVector"
/*@
   DMRestoreLocalVector - Returns a Seq PETSc vector that
     obtained from DMGetLocalVector(). Do not use with vector obtained via
     DMCreateLocalVector().

   Not Collective

   Input Parameter:
+  dm - the distributed array
-  g - the local vector

   Level: beginner

.keywords: distributed array, create, local, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMCreateLocalVector(), DMGetLocalVector()
@*/
PetscErrorCode  DMRestoreLocalVector(DM dm,Vec* g)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (j=0; j<DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->localout[j]) {
      dm->localout[j] = PETSC_NULL;
      for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
        if (!dm->localin[i]) {
          dm->localin[i] = *g;
          goto alldone;
        }
      }
    }
  }
  ierr = VecDestroy(g);CHKERRQ(ierr);
  alldone:
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetGlobalVector"
/*@
   DMGetGlobalVector - Gets a MPI PETSc vector that
   may be used with the DMXXX routines.

   Collective on DM

   Input Parameter:
.  dm - the distributed array

   Output Parameter:
.  g - the global vector

   Level: beginner

   Note:
   The vector values are NOT initialized and may have garbage in them, so you may need
   to zero them.

   The output parameter, g, is a regular PETSc vector that should be returned with
   DMRestoreGlobalVector() DO NOT call VecDestroy() on it.

   VecStride*() operations can be useful when using DM with dof > 1

.keywords: distributed array, create, Global, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMCreateLocalVector(), DMRestoreLocalVector()
          VecStrideMax(), VecStrideMin(), VecStrideNorm()

@*/
PetscErrorCode  DMGetGlobalVector(DM dm,Vec* g)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->globalin[i]) {
      *g             = dm->globalin[i];
      dm->globalin[i] = PETSC_NULL;
      goto alldone;
    }
  }
  ierr = DMCreateGlobalVector(dm,g);CHKERRQ(ierr);

  alldone:
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (!dm->globalout[i]) {
      dm->globalout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMClearGlobalVectors"
/*@
   DMClearGlobalVectors - Destroys all the global vectors that have been stashed in this DM

   Collective on DM

   Input Parameter:
.  dm - the distributed array

   Level: developer

.keywords: distributed array, create, Global, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMCreateLocalVector(), DMRestoreLocalVector()
          VecStrideMax(), VecStrideMin(), VecStrideNorm()

@*/
PetscErrorCode  DMClearGlobalVectors(DM dm)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->globalout[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Clearing DM of global vectors that has a global vector obtained with DMGetGlobalVector()");
    ierr = VecDestroy(&dm->globalin[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestoreGlobalVector"
/*@
   DMRestoreGlobalVector - Returns a Seq PETSc vector that
     obtained from DMGetGlobalVector(). Do not use with vector obtained via
     DMCreateGlobalVector().

   Not Collective

   Input Parameter:
+  dm - the distributed array
-  g - the global vector

   Level: beginner

.keywords: distributed array, create, global, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToGlobalBegin(),
          DMGlobalToGlobalEnd(), DMGlobalToGlobal(), DMCreateLocalVector(), DMGetGlobalVector()
@*/
PetscErrorCode  DMRestoreGlobalVector(DM dm,Vec* g)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (j=0; j<DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->globalout[j]) {
      dm->globalout[j] = PETSC_NULL;
      for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
        if (!dm->globalin[i]) {
          dm->globalin[i] = *g;
          goto alldone;
        }
      }
    }
  }
  ierr = VecDestroy(g);CHKERRQ(ierr);
  alldone:
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetNamedGlobalVector"
/*@C
   DMGetNamedGlobalVector - get access to a named, persistent global vector

   Collective on DM

   Input Arguments:
+  dm - DM to hold named vectors
-  name - unique name for Vec

   Output Arguments:
.  X - named Vec

   Level: developer

   Note: If a Vec with the given name does not exist, it is created.

.seealso: DMRestoreNamedGlobalVector()
@*/
PetscErrorCode DMGetNamedGlobalVector(DM dm,const char *name,Vec *X)
{
  PetscErrorCode ierr;
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  for (link=dm->namedglobal; link; link=link->next) {
    PetscBool match;
    ierr = PetscStrcmp(name,link->name,&match);CHKERRQ(ierr);
    if (match) {
      if (link->status != DMVEC_STATUS_IN) SETERRQ1(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' already checked out",name);
      goto found;
    }
  }

  /* Create the Vec */
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&link->name);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&link->X);CHKERRQ(ierr);
  link->next = dm->namedglobal;
  dm->namedglobal = link;

  found:
  *X = link->X;
  link->status = DMVEC_STATUS_OUT;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestoreNamedGlobalVector"
/*@C
   DMRestoreNamedGlobalVector - restore access to a named, persistent global vector

   Collective on DM

   Input Arguments:
+  dm - DM on which the vector was gotten
.  name - name under which the vector was gotten
-  X - Vec to restore

   Output Arguments:

   Level: developer

.seealso: DMGetNamedGlobalVector()
@*/
PetscErrorCode DMRestoreNamedGlobalVector(DM dm,const char *name,Vec *X)
{
  PetscErrorCode ierr;
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  PetscValidHeaderSpecific(*X,VEC_CLASSID,3);
  for (link=dm->namedglobal; link; link=link->next) {
    PetscBool match;
    ierr = PetscStrcmp(name,link->name,&match);CHKERRQ(ierr);
    if (match) {
      if (link->status != DMVEC_STATUS_OUT) SETERRQ1(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' was not checked out",name);
      if (link->X != *X) SETERRQ1(((PetscObject)dm)->comm,PETSC_ERR_ARG_INCOMP,"Attempt to restore Vec name '%s', but Vec does not match the cache",name);
      link->status = DMVEC_STATUS_IN;
      *X = PETSC_NULL;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(((PetscObject)dm)->comm,PETSC_ERR_ARG_INCOMP,"Could not find Vec name '%s' to restore",name);
  PetscFunctionReturn(0);
}

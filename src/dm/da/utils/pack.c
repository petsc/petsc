#define PETSCDM_DLL
 
#include "petscda.h"     /*I      "petscda.h"     I*/
#include "petscmat.h"    /*I      "petscmat.h"    I*/

/*
   rstart is where an array/subvector starts in the global parallel vector, so arrays
   rstarts are meaningless (and set to the previous one) except on processor 0
*/

typedef enum {VECPACK_ARRAY, VECPACK_DA, VECPACK_VECSCATTER} VecPackLinkType;

struct VecPackLink {
  DA                 da;
  PetscInt           n,rstart;      /* rstart is relative to this processor */
  VecPackLinkType    type;
  struct VecPackLink *next;
};

typedef struct _VecPackOps *VecPackOps;
struct _VecPackOps {
  PetscErrorCode (*view)(VecPack,PetscViewer);
  PetscErrorCode (*createglobalvector)(VecPack,Vec*);
  PetscErrorCode (*getcoloring)(VecPack,ISColoringType,ISColoring*);
  PetscErrorCode (*getmatrix)(VecPack,MatType,Mat*);
  PetscErrorCode (*getinterpolation)(VecPack,VecPack,Mat*,Vec*);
  PetscErrorCode (*refine)(VecPack,MPI_Comm,VecPack*);
};

struct _p_VecPack {
  PETSCHEADER(struct _VecPackOps);
  PetscMPIInt        rank;
  PetscInt           n,N,rstart;   /* rstart is relative to all processors */
  Vec                globalvector;
  PetscInt           nDA,nredundant;
  struct VecPackLink *next;
};

#undef __FUNCT__  
#define __FUNCT__ "VecPackCreate"
/*@C
    VecPackCreate - Creates a vector packer, used to generate "composite"
      vectors made up of several subvectors.

    Collective on MPI_Comm

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   packer - the packer object

    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackScatter(),
         VecPackGather(), VecPackCreateGlobalVector(), VecPackGetGlobalIndices(), VecPackGetAccess()
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackCreate(MPI_Comm comm,VecPack *packer)
{
  PetscErrorCode ierr;
  VecPack        p;

  PetscFunctionBegin;
  PetscValidPointer(packer,2);
  *packer = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(p,_p_VecPack,struct _VecPackOps,DA_COOKIE,0,"VecPack",comm,VecPackDestroy,0);CHKERRQ(ierr);
  p->n            = 0;
  p->next         = PETSC_NULL;
  p->comm         = comm;
  p->globalvector = PETSC_NULL;
  p->nredundant   = 0;
  p->nDA          = 0;
  ierr            = MPI_Comm_rank(comm,&p->rank);CHKERRQ(ierr);

  p->ops->createglobalvector = VecPackCreateGlobalVector;
  p->ops->refine             = VecPackRefine;
  p->ops->getinterpolation   = VecPackGetInterpolation;
  p->ops->getmatrix          = VecPackGetMatrix;
  *packer = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackDestroy"
/*@C
    VecPackDestroy - Destroys a vector packer.

    Collective on VecPack

    Input Parameter:
.   packer - the packer object

    Level: advanced

.seealso VecPackCreate(), VecPackAddArray(), VecPackAddDA(), VecPackScatter(),
         VecPackGather(), VecPackCreateGlobalVector(), VecPackGetGlobalIndices(), VecPackGetAccess()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackDestroy(VecPack packer)
{
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next,*prev;

  PetscFunctionBegin;
  if (--packer->refct > 0) PetscFunctionReturn(0);
  while (next) {
    prev = next;
    next = next->next;
    if (prev->type == VECPACK_DA) {
      ierr = DADestroy(prev->da);CHKERRQ(ierr);
    }
    ierr = PetscFree(prev);CHKERRQ(ierr);
  }
  if (packer->globalvector) {
    ierr = VecDestroy(packer->globalvector);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(packer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetAccess_Array"
PetscErrorCode VecPackGetAccess_Array(VecPack packer,struct VecPackLink *mine,Vec vec,PetscScalar **array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;

  PetscFunctionBegin;
  if (array) {
    if (!packer->rank) {
      ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
      *array  = varray + mine->rstart;
      ierr    = VecRestoreArray(vec,&varray);CHKERRQ(ierr);
    } else {
      *array = 0;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetAccess_DA"
PetscErrorCode VecPackGetAccess_DA(VecPack packer,struct VecPackLink *mine,Vec vec,Vec *global)
{
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (global) {
    ierr    = DAGetGlobalVector(mine->da,global);CHKERRQ(ierr);
    ierr    = VecGetArray(vec,&array);CHKERRQ(ierr);
    ierr    = VecPlaceArray(*global,array+mine->rstart);CHKERRQ(ierr);
    ierr    = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreAccess_Array"
PetscErrorCode VecPackRestoreAccess_Array(VecPack packer,struct VecPackLink *mine,Vec vec,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreAccess_DA"
PetscErrorCode VecPackRestoreAccess_DA(VecPack packer,struct VecPackLink *mine,Vec vec,Vec *global)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (global) {
    ierr = VecResetArray(*global);CHKERRQ(ierr);
    ierr = DARestoreGlobalVector(mine->da,global);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackScatter_Array"
PetscErrorCode VecPackScatter_Array(VecPack packer,struct VecPackLink *mine,Vec vec,PetscScalar *array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;

  PetscFunctionBegin;

  if (!packer->rank) {
    ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
    ierr    = PetscMemcpy(array,varray+mine->rstart,mine->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr    = VecRestoreArray(vec,&varray);CHKERRQ(ierr);
  }
  ierr    = MPI_Bcast(array,mine->n,MPIU_SCALAR,0,packer->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackScatter_DA"
PetscErrorCode VecPackScatter_DA(VecPack packer,struct VecPackLink *mine,Vec vec,Vec local)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec            global;

  PetscFunctionBegin;
  ierr = DAGetGlobalVector(mine->da,&global);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(global,array+mine->rstart);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(mine->da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(mine->da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  ierr = VecResetArray(global);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(mine->da,&global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGather_Array"
PetscErrorCode VecPackGather_Array(VecPack packer,struct VecPackLink *mine,Vec vec,PetscScalar *array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;

  PetscFunctionBegin;
  if (!packer->rank) {
    ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
    if (varray+mine->rstart == array) SETERRQ(PETSC_ERR_ARG_WRONG,"You need not VecPackGather() into objects obtained via VecPackGetAccess()");
    ierr    = PetscMemcpy(varray+mine->rstart,array,mine->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr    = VecRestoreArray(vec,&varray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGather_DA"
PetscErrorCode VecPackGather_DA(VecPack packer,struct VecPackLink *mine,Vec vec,Vec local)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec            global;

  PetscFunctionBegin;
  ierr = DAGetGlobalVector(mine->da,&global);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(global,array+mine->rstart);CHKERRQ(ierr);
  ierr = DALocalToGlobal(mine->da,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  ierr = VecResetArray(global);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(mine->da,&global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

#include <stdarg.h>

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetAccess"
/*@C
    VecPackGetAccess - Allows one to access the individual packed vectors in their global
       representation.

    Collective on VecPack

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    ... - the individual sequential or parallel objects (arrays or vectors)
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackScatter(),
         VecPackRestoreAccess(), VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackGetAccess(VecPack packer,Vec gvec,...)
{
  va_list            Argp;
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (!packer->globalvector) {
    SETERRQ(PETSC_ERR_ORDER,"Must first create global vector with VecPackCreateGlobalVector()");
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr  = VecPackGetAccess_Array(packer,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      Vec *vec;
      vec  = va_arg(Argp, Vec*);
      ierr = VecPackGetAccess_DA(packer,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreAccess"
/*@C
    VecPackRestoreAccess - Allows one to access the individual packed vectors in their global
       representation.

    Collective on VecPack

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    ... - the individual sequential or parallel objects (arrays or vectors)
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackScatter(),
         VecPackRestoreAccess()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackRestoreAccess(VecPack packer,Vec gvec,...)
{
  va_list            Argp;
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (!packer->globalvector) {
    SETERRQ(PETSC_ERR_ORDER,"Must first create global vector with VecPackCreateGlobalVector()");
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr  = VecPackRestoreAccess_Array(packer,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      Vec *vec;
      vec  = va_arg(Argp, Vec*);
      ierr = VecPackRestoreAccess_DA(packer,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackScatter"
/*@C
    VecPackScatter - Scatters from a global packed vector into its individual local vectors

    Collective on VecPack

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackScatter(VecPack packer,Vec gvec,...)
{
  va_list            Argp;
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (!packer->globalvector) {
    SETERRQ(PETSC_ERR_ORDER,"Must first create global vector with VecPackCreateGlobalVector()");
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      PetscScalar *array;
      array = va_arg(Argp, PetscScalar*);
      ierr = VecPackScatter_Array(packer,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      Vec vec;
      vec = va_arg(Argp, Vec);
      PetscValidHeaderSpecific(vec,VEC_COOKIE,3);
      ierr = VecPackScatter_DA(packer,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGather"
/*@C
    VecPackGather - Gathers into a global packed vector from its individual local vectors

    Collective on VecPack

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackScatter(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackGather(VecPack packer,Vec gvec,...)
{
  va_list            Argp;
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (!packer->globalvector) {
    SETERRQ(PETSC_ERR_ORDER,"Must first create global vector with VecPackCreateGlobalVector()");
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      PetscScalar *array;
      array = va_arg(Argp, PetscScalar*);
      ierr  = VecPackGather_Array(packer,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      Vec vec;
      vec = va_arg(Argp, Vec);
      PetscValidHeaderSpecific(vec,VEC_COOKIE,3);
      ierr = VecPackGather_DA(packer,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackAddArray"
/*@C
    VecPackAddArray - adds an "redundant" array to a VecPack. The array values will 
       be stored in part of the array on processor 0.

    Collective on VecPack

    Input Parameter:
+    packer - the packer object
-    n - the length of the array
 
    Level: advanced

.seealso VecPackDestroy(), VecPackGather(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackScatter(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackAddArray(VecPack packer,PetscInt n)
{
  struct VecPackLink *mine,*next = packer->next;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (packer->globalvector) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot add an array once you have called VecPackCreateGlobalVector()");
  }

  /* create new link */
  ierr                = PetscNew(struct VecPackLink,&mine);CHKERRQ(ierr);
  mine->n             = n;
  mine->da            = PETSC_NULL;
  mine->type          = VECPACK_ARRAY;
  mine->next          = PETSC_NULL;
  if (!packer->rank) packer->n += n;

  /* add to end of list */
  if (!next) {
    packer->next = mine;
  } else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  packer->nredundant++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackAddDA"
/*@C
    VecPackAddDA - adds a DA vector to a VecPack

    Collective on VecPack

    Input Parameter:
+    packer - the packer object
-    da - the DA object
 
    Level: advanced

.seealso VecPackDestroy(), VecPackGather(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackScatter(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackAddDA(VecPack packer,DA da)
{
  PetscErrorCode     ierr;
  PetscInt           n;
  struct VecPackLink *mine,*next = packer->next;
  Vec                global;

  PetscFunctionBegin;
  if (packer->globalvector) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot add a DA once you have called VecPackCreateGlobalVector()");
  }

  /* create new link */
  ierr = PetscNew(struct VecPackLink,&mine);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)da);CHKERRQ(ierr);
  ierr = DAGetGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = VecGetLocalSize(global,&n);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&global);CHKERRQ(ierr);
  mine->n      = n;
  mine->da     = da;  
  mine->type   = VECPACK_DA;
  mine->next   = PETSC_NULL;
  packer->n   += n;

  /* add to end of list */
  if (!next) {
    packer->next = mine;
  } else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  packer->nDA++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackCreateGlobalVector"
/*@C
    VecPackCreateGlobalVector - Creates a vector of the correct size to be gathered into 
        by the packer.

    Collective on VecPack

    Input Parameter:
.    packer - the packer object

    Output Parameters:
.   gvec - the global vector

    Level: advanced

    Notes: Once this has been created you cannot add additional arrays or vectors to be packed.

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackScatter(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackCreateGlobalVector(VecPack packer,Vec *gvec)
{
  PetscErrorCode     ierr;
  PetscInt           nprev = 0;
  PetscMPIInt        rank;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (packer->globalvector) {
    ierr = VecDuplicate(packer->globalvector,gvec);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPI(packer->comm,packer->n,PETSC_DETERMINE,gvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*gvec);CHKERRQ(ierr);
    packer->globalvector = *gvec;

    ierr = VecGetSize(*gvec,&packer->N);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(*gvec,&packer->rstart,PETSC_NULL);CHKERRQ(ierr);
    
    /* now set the rstart for each linked array/vector */
    ierr = MPI_Comm_rank(packer->comm,&rank);CHKERRQ(ierr);
    while (next) {
      next->rstart = nprev; 
      if (!rank || next->type != VECPACK_ARRAY) nprev += next->n;
      next = next->next;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetGlobalIndices"
/*@C
    VecPackGetGlobalIndices - Gets the global indices for all the entries in the packed
      vectors.

    Collective on VecPack

    Input Parameter:
.    packer - the packer object

    Output Parameters:
.    idx - the individual indices for each packed vector/array. Note that this includes
           all the ghost points that individual ghosted DA's may have.
 
    Level: advanced

    Notes:
       The idx parameters should be freed by the calling routine with PetscFree()

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetAccess(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackGetGlobalIndices(VecPack packer,...)
{
  va_list            Argp;
  PetscErrorCode     ierr;
  PetscInt           i,**idx,n;
  struct VecPackLink *next = packer->next;
  Vec                global,dglobal;
  PF                 pf;
  PetscScalar        *array;

  PetscFunctionBegin;
  ierr = VecPackCreateGlobalVector(packer,&global);CHKERRQ(ierr);

  /* put 0 to N-1 into the global vector */
  ierr = PFCreate(PETSC_COMM_WORLD,1,1,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFIDENTITY,PETSC_NULL);CHKERRQ(ierr);
  ierr = PFApplyVec(pf,PETSC_NULL,global);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  va_start(Argp,packer);
  while (next) {
    idx = va_arg(Argp, PetscInt**);

    if (next->type == VECPACK_ARRAY) {
      
      ierr = PetscMalloc(next->n*sizeof(PetscInt),idx);CHKERRQ(ierr);
      if (!packer->rank) {
        ierr   = VecGetArray(global,&array);CHKERRQ(ierr);
        array += next->rstart;
        for (i=0; i<next->n; i++) (*idx)[i] = (PetscInt)PetscRealPart(array[i]);
        array -= next->rstart;
        ierr   = VecRestoreArray(global,&array);CHKERRQ(ierr);
      }
      ierr = MPI_Bcast(*idx,next->n,MPIU_INT,0,packer->comm);CHKERRQ(ierr);

    } else if (next->type == VECPACK_DA) {
      Vec local;

      ierr   = DACreateLocalVector(next->da,&local);CHKERRQ(ierr);
      ierr   = VecGetArray(global,&array);CHKERRQ(ierr);
      array += next->rstart;
      ierr   = DAGetGlobalVector(next->da,&dglobal);CHKERRQ(ierr);
      ierr   = VecPlaceArray(dglobal,array);CHKERRQ(ierr);
      ierr   = DAGlobalToLocalBegin(next->da,dglobal,INSERT_VALUES,local);CHKERRQ(ierr);
      ierr   = DAGlobalToLocalEnd(next->da,dglobal,INSERT_VALUES,local);CHKERRQ(ierr);
      array -= next->rstart;
      ierr   = VecRestoreArray(global,&array);CHKERRQ(ierr);
      ierr   = VecResetArray(dglobal);CHKERRQ(ierr);
      ierr   = DARestoreGlobalVector(next->da,&dglobal);CHKERRQ(ierr);

      ierr   = VecGetArray(local,&array);CHKERRQ(ierr);
      ierr   = VecGetSize(local,&n);CHKERRQ(ierr);
      ierr   = PetscMalloc(n*sizeof(PetscInt),idx);CHKERRQ(ierr);
      for (i=0; i<n; i++) (*idx)[i] = (PetscInt)PetscRealPart(array[i]);
      ierr    = VecRestoreArray(local,&array);CHKERRQ(ierr);
      ierr    = VecDestroy(local);CHKERRQ(ierr);

    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecPackGetLocalVectors_Array"
PetscErrorCode VecPackGetLocalVectors_Array(VecPack packer,struct VecPackLink *mine,PetscScalar **array)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscMalloc(mine->n*sizeof(PetscScalar),array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetLocalVectors_DA"
PetscErrorCode VecPackGetLocalVectors_DA(VecPack packer,struct VecPackLink *mine,Vec *local)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetLocalVector(mine->da,local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreLocalVectors_Array"
PetscErrorCode VecPackRestoreLocalVectors_Array(VecPack packer,struct VecPackLink *mine,PetscScalar **array)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(*array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreLocalVectors_DA"
PetscErrorCode VecPackRestoreLocalVectors_DA(VecPack packer,struct VecPackLink *mine,Vec *local)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DARestoreLocalVector(mine->da,local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetLocalVectors"
/*@C
    VecPackGetLocalVectors - Gets local vectors and arrays for each part of a VecPack.'
       Use VecPakcRestoreLocalVectors() to return them.

    Collective on VecPack

    Input Parameter:
.    packer - the packer object
 
    Output Parameter:
.    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(), 
         VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackGetLocalVectors(VecPack packer,...)
{
  va_list            Argp;
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;

  /* loop over packed objects, handling one at at time */
  va_start(Argp,packer);
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr = VecPackGetLocalVectors_Array(packer,next,array);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      Vec *vec;
      vec = va_arg(Argp, Vec*);
      ierr = VecPackGetLocalVectors_DA(packer,next,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreLocalVectors"
/*@C
    VecPackRestoreLocalVectors - Restores local vectors and arrays for each part of a VecPack.'
       Use VecPakcRestoreLocalVectors() to return them.

    Collective on VecPack

    Input Parameter:
.    packer - the packer object
 
    Output Parameter:
.    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(), 
         VecPackGetLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackRestoreLocalVectors(VecPack packer,...)
{
  va_list            Argp;
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;

  /* loop over packed objects, handling one at at time */
  va_start(Argp,packer);
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr = VecPackRestoreLocalVectors_Array(packer,next,array);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      Vec *vec;
      vec = va_arg(Argp, Vec*);
      ierr = VecPackRestoreLocalVectors_DA(packer,next,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecPackGetEntries_Array"
PetscErrorCode VecPackGetEntries_Array(VecPack packer,struct VecPackLink *mine,PetscInt *n)
{
  PetscFunctionBegin;
  if (n) *n = mine->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetEntries_DA"
PetscErrorCode VecPackGetEntries_DA(VecPack packer,struct VecPackLink *mine,DA *da)
{
  PetscFunctionBegin;
  if (da) *da = mine->da;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetEntries"
/*@C
    VecPackGetEntries - Gets the DA, redundant size, etc for each entry in a VecPack.
       Use VecPackRestoreEntries() to return them.

    Collective on VecPack

    Input Parameter:
.    packer - the packer object
 
    Output Parameter:
.    ... - the individual entries, DAs or integer sizes)
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(), 
         VecPackRestoreLocalVectors(), VecPackGetLocalVectors(), VecPackRestoreEntries(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackGetEntries(VecPack packer,...)
{
  va_list            Argp;
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;

  /* loop over packed objects, handling one at at time */
  va_start(Argp,packer);
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      PetscInt *n;
      n = va_arg(Argp, PetscInt*);
      ierr = VecPackGetEntries_Array(packer,next,n);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      DA *da;
      da = va_arg(Argp, DA*);
      ierr = VecPackGetEntries_DA(packer,next,da);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRefine"
/*@C
    VecPackRefine - Refines a VecPack by refining all of its DAs

    Collective on VecPack

    Input Parameters:
+    packer - the packer object
-    comm - communicator to contain the new DM object, usually PETSC_NULL

    Output Parameter:
.    fine - new packer
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackRefine(VecPack packer,MPI_Comm comm,VecPack *fine)
{
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;
  DA                 da;

  PetscFunctionBegin;
  ierr = VecPackCreate(comm,fine);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      ierr = VecPackAddArray(*fine,next->n);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      ierr = DARefine(next->da,comm,&da);CHKERRQ(ierr);
      ierr = VecPackAddDA(*fine,da);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)da);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#include "petscmat.h"

struct MatPackLink {
  Mat                A;
  struct MatPackLink *next;
};

struct MatPack {
  VecPack            right,left;
  struct MatPackLink *next;
};

#undef __FUNCT__  
#define __FUNCT__ "MatMultBoth_Shell_Pack"
PetscErrorCode MatMultBoth_Shell_Pack(Mat A,Vec x,Vec y,PetscTruth add)
{
  struct MatPack     *mpack;
  struct VecPackLink *xnext,*ynext;
  struct MatPackLink *anext;
  PetscScalar        *xarray,*yarray;
  PetscErrorCode     ierr;
  PetscInt           i;
  Vec                xglobal,yglobal;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&mpack);CHKERRQ(ierr);
  xnext = mpack->right->next;
  ynext = mpack->left->next;
  anext = mpack->next;

  while (xnext) {
    if (xnext->type == VECPACK_ARRAY) {
      if (!mpack->right->rank) {
        ierr    = VecGetArray(x,&xarray);CHKERRQ(ierr);
        ierr    = VecGetArray(y,&yarray);CHKERRQ(ierr);
        if (add) {
          for (i=0; i<xnext->n; i++) {
            yarray[ynext->rstart+i] += xarray[xnext->rstart+i];
          }
        } else {
          ierr    = PetscMemcpy(yarray+ynext->rstart,xarray+xnext->rstart,xnext->n*sizeof(PetscScalar));CHKERRQ(ierr);
        }
        ierr    = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
        ierr    = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
      }
    } else if (xnext->type == VECPACK_DA) {
      ierr  = VecGetArray(x,&xarray);CHKERRQ(ierr);
      ierr  = VecGetArray(y,&yarray);CHKERRQ(ierr);
      ierr  = DAGetGlobalVector(xnext->da,&xglobal);CHKERRQ(ierr);
      ierr  = DAGetGlobalVector(ynext->da,&yglobal);CHKERRQ(ierr);
      ierr  = VecPlaceArray(xglobal,xarray+xnext->rstart);CHKERRQ(ierr);
      ierr  = VecPlaceArray(yglobal,yarray+ynext->rstart);CHKERRQ(ierr);
      if (add) {
        ierr  = MatMultAdd(anext->A,xglobal,yglobal,yglobal);CHKERRQ(ierr);
      } else {
        ierr  = MatMult(anext->A,xglobal,yglobal);CHKERRQ(ierr);
      }
      ierr  = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
      ierr  = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
      ierr  = VecResetArray(xglobal);CHKERRQ(ierr);
      ierr  = VecResetArray(yglobal);CHKERRQ(ierr);
      ierr  = DARestoreGlobalVector(xnext->da,&xglobal);CHKERRQ(ierr);
      ierr  = DARestoreGlobalVector(ynext->da,&yglobal);CHKERRQ(ierr);
      anext = anext->next;
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    xnext = xnext->next;
    ynext = ynext->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_Shell_Pack"
PetscErrorCode MatMultAdd_Shell_Pack(Mat A,Vec x,Vec y,Vec z)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (z != y) SETERRQ(PETSC_ERR_SUP,"Handles y == z only");
  ierr = MatMultBoth_Shell_Pack(A,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Shell_Pack"
PetscErrorCode MatMult_Shell_Pack(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMultBoth_Shell_Pack(A,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Shell_Pack"
PetscErrorCode MatMultTranspose_Shell_Pack(Mat A,Vec x,Vec y)
{
  struct MatPack     *mpack;
  struct VecPackLink *xnext,*ynext;
  struct MatPackLink *anext;
  PetscScalar        *xarray,*yarray;
  PetscErrorCode     ierr;
  Vec                xglobal,yglobal;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(A,(void**)&mpack);CHKERRQ(ierr);
  xnext = mpack->left->next;
  ynext = mpack->right->next;
  anext = mpack->next;

  while (xnext) {
    if (xnext->type == VECPACK_ARRAY) {
      if (!mpack->right->rank) {
        ierr    = VecGetArray(x,&xarray);CHKERRQ(ierr);
        ierr    = VecGetArray(y,&yarray);CHKERRQ(ierr);
        ierr    = PetscMemcpy(yarray+ynext->rstart,xarray+xnext->rstart,xnext->n*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr    = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
        ierr    = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
      }
    } else if (xnext->type == VECPACK_DA) {
      ierr  = VecGetArray(x,&xarray);CHKERRQ(ierr);
      ierr  = VecGetArray(y,&yarray);CHKERRQ(ierr);
      ierr  = DAGetGlobalVector(xnext->da,&xglobal);CHKERRQ(ierr);
      ierr  = DAGetGlobalVector(ynext->da,&yglobal);CHKERRQ(ierr);
      ierr  = VecPlaceArray(xglobal,xarray+xnext->rstart);CHKERRQ(ierr);
      ierr  = VecPlaceArray(yglobal,yarray+ynext->rstart);CHKERRQ(ierr);
      ierr  = MatMultTranspose(anext->A,xglobal,yglobal);CHKERRQ(ierr);
      ierr  = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
      ierr  = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
      ierr  = VecResetArray(xglobal);CHKERRQ(ierr);
      ierr  = VecResetArray(yglobal);CHKERRQ(ierr);
      ierr  = DARestoreGlobalVector(xnext->da,&xglobal);CHKERRQ(ierr);
      ierr  = DARestoreGlobalVector(ynext->da,&yglobal);CHKERRQ(ierr);
      anext = anext->next;
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    xnext = xnext->next;
    ynext = ynext->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Shell_Pack"
PetscErrorCode MatDestroy_Shell_Pack(Mat A)
{
  struct MatPack     *mpack;
  struct MatPackLink *anext,*oldanext;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(A,(void**)&mpack);CHKERRQ(ierr);
  anext = mpack->next;

  while (anext) {
    ierr     = MatDestroy(anext->A);CHKERRQ(ierr);
    oldanext = anext;
    anext    = anext->next;
    ierr     = PetscFree(oldanext);CHKERRQ(ierr);
  }
  ierr = PetscFree(mpack);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetInterpolation"
/*@C
    VecPackGetInterpolation - GetInterpolations a VecPack by refining all of its DAs

    Collective on VecPack

    Input Parameters:
+    coarse - coarse grid packer
-    fine - fine grid packer

    Output Parameter:
+    A - interpolation matrix
-    v - scaling vector
 
    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess(),
         VecPackGetLocalVectors(), VecPackRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackGetInterpolation(VecPack coarse,VecPack fine,Mat *A,Vec *v)
{
  PetscErrorCode     ierr;
  PetscInt           m,n,M,N;
  struct VecPackLink *nextc  = coarse->next;
  struct VecPackLink *nextf = fine->next;
  struct MatPackLink *nextmat,*pnextmat = 0;
  struct MatPack     *mpack;
  Vec                gcoarse,gfine;

  PetscFunctionBegin;
  /* use global vectors only for determining matrix layout */
  ierr = VecPackCreateGlobalVector(coarse,&gcoarse);CHKERRQ(ierr);
  ierr = VecPackCreateGlobalVector(fine,&gfine);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gcoarse,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gfine,&m);CHKERRQ(ierr);
  ierr = VecGetSize(gcoarse,&N);CHKERRQ(ierr);
  ierr = VecGetSize(gfine,&M);CHKERRQ(ierr);
  ierr = VecDestroy(gcoarse);CHKERRQ(ierr);
  ierr = VecDestroy(gfine);CHKERRQ(ierr);

  ierr         = PetscNew(struct MatPack,&mpack);CHKERRQ(ierr);
  mpack->right = coarse;
  mpack->left  = fine;
  ierr  = MatCreate(fine->comm,A);CHKERRQ(ierr);
  ierr  = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr  = MatSetType(*A,MATSHELL);CHKERRQ(ierr);
  ierr  = MatShellSetContext(*A,mpack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))MatMult_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_DESTROY,(void(*)(void))MatDestroy_Shell_Pack);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (nextc) {
    if (nextc->type != nextf->type) SETERRQ(PETSC_ERR_ARG_INCOMP,"Two VecPack have different layout");

    if (nextc->type == VECPACK_ARRAY) {
      ;
    } else if (nextc->type == VECPACK_DA) {
      ierr          = PetscNew(struct MatPackLink,&nextmat);CHKERRQ(ierr);
      nextmat->next = 0;
      if (pnextmat) {
        pnextmat->next = nextmat;
        pnextmat       = nextmat;
      } else {
        pnextmat    = nextmat;
        mpack->next = nextmat;
      }
      ierr = DAGetInterpolation(nextc->da,nextf->da,&nextmat->A,PETSC_NULL);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    nextc = nextc->next;
    nextf = nextf->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetMatrix" 
/*@C
    VecPackGetMatrix - Creates a matrix with the correct parallel layout and nonzero structure required for 
      computing the Jacobian on a function defined using the stencils set in the DA's and coupling in the array variables

    Collective on DA

    Input Parameter:
+   da - the distributed array
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ

    Output Parameters:
.   J  - matrix with the correct nonzero structure
        (obviously without the correct Jacobian values)

    Level: advanced

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself. 


.seealso DAGetMatrix(), VecPackCreate()

@*/
PetscErrorCode PETSCDM_DLLEXPORT VecPackGetMatrix(VecPack packer, MatType mtype,Mat *J)
{
  PetscErrorCode     ierr;
  struct VecPackLink *next = packer->next;
  Vec                gvec;
  PetscInt           m,*dnz,*onz,i,j,mA;
  Mat                Atmp;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  /* use global vector to determine layout needed for matrix */
  ierr = VecPackCreateGlobalVector(packer,&gvec);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gvec,&m);CHKERRQ(ierr);
  ierr = VecDestroy(gvec);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(packer->comm,&rank);CHKERRQ(ierr);
  ierr = MatCreate(packer->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,m,m,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J,mtype);CHKERRQ(ierr);

  ierr = MatPreallocateInitialize(packer->comm,m,m,dnz,onz);CHKERRQ(ierr);
  /* loop over packed objects, handling one at at time */
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      for (j=0; j<next->n; j++) {
	if (rank) {
          for (i=0; i<m; i++) {
            ierr = MatPreallocateSet(j,1,&i,dnz,onz);CHKERRQ(ierr);
          }
	} else {
          for (i=next->n; i<m; i++) {
            ierr = MatPreallocateSet(i,1,&j,dnz,onz);CHKERRQ(ierr);
          }
	}
      }
    } else if (next->type == VECPACK_DA) {
      PetscInt       nc,rstart;
      const PetscInt *cols;

      ierr = DAGetMatrix(next->da,mtype,&Atmp);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(Atmp,&rstart,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetLocalSize(Atmp,&mA,PETSC_NULL);CHKERRQ(ierr);
      for (i=0; i<mA; i++) {
        ierr = MatGetRow(Atmp,rstart+i,&nc,&cols,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatPreallocateSet(packer->rstart+next->rstart+i,nc,cols,dnz,onz);CHKERRQ(ierr);
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  ierr = MatMPIAIJSetPreallocation(*J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J,0,dnz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



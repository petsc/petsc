 
#include "petscda.h"     /*I      "petscda.h"     I*/
#include "petscmat.h"    /*I      "petscmat.h"    I*/

/*
   rstart is where an array/subvector starts in the global parallel vector, so arrays
   rstarts are meaningless (and set to the previous one) except on processor 0
*/

typedef enum {VECPACK_ARRAY, VECPACK_DA, VECPACK_VECSCATTER} VecPackLinkType;

struct VecPackLink {
  DA                 da;
  int                n,rstart;      /* rstart is relative to this processor */
  VecPackLinkType    type;
  struct VecPackLink *next;
};

typedef struct _VecPackOps *VecPackOps;
struct _VecPackOps {
  int  (*view)(VecPack,PetscViewer);
  int  (*createglobalvector)(VecPack,Vec*);
  int  (*getcoloring)(VecPack,ISColoringType,ISColoring*);
  int  (*getmatrix)(VecPack,MatType,Mat*);
  int  (*getinterpolation)(VecPack,VecPack,Mat*,Vec*);
  int  (*refine)(VecPack,MPI_Comm,VecPack*);
};

struct _p_VecPack {
  PETSCHEADER(struct _VecPackOps)
  int                rank;
  int                n,N,rstart;   /* rstart is relative to all processors */
  Vec                globalvector;
  int                nDA,nredundant;
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

@*/
int VecPackCreate(MPI_Comm comm,VecPack *packer)
{
  int     ierr;
  VecPack p;

  PetscFunctionBegin;
  PetscValidPointer(packer,2);
  *packer = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  PetscHeaderCreate(p,_p_VecPack,struct _VecPackOps,DA_COOKIE,0,"VecPack",comm,VecPackDestroy,0);
  PetscLogObjectCreate(p);
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
int VecPackDestroy(VecPack packer)
{
  int                ierr;
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
  PetscHeaderDestroy(packer);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetAccess_Array"
int VecPackGetAccess_Array(VecPack packer,struct VecPackLink *mine,Vec vec,PetscScalar **array)
{
  int    ierr;
  PetscScalar *varray;

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
int VecPackGetAccess_DA(VecPack packer,struct VecPackLink *mine,Vec vec,Vec *global)
{
  int    ierr;
  PetscScalar *array;

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
int VecPackRestoreAccess_Array(VecPack packer,struct VecPackLink *mine,Vec vec,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreAccess_DA"
int VecPackRestoreAccess_DA(VecPack packer,struct VecPackLink *mine,Vec vec,Vec *global)
{
  int    ierr;

  PetscFunctionBegin;
  if (global) {
    ierr = VecResetArray(*global);CHKERRQ(ierr);
    ierr = DARestoreGlobalVector(mine->da,global);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackScatter_Array"
int VecPackScatter_Array(VecPack packer,struct VecPackLink *mine,Vec vec,PetscScalar *array)
{
  int    ierr;
  PetscScalar *varray;

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
int VecPackScatter_DA(VecPack packer,struct VecPackLink *mine,Vec vec,Vec local)
{
  int    ierr;
  PetscScalar *array;
  Vec    global;

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
int VecPackGather_Array(VecPack packer,struct VecPackLink *mine,Vec vec,PetscScalar *array)
{
  int    ierr;
  PetscScalar *varray;

  PetscFunctionBegin;
  if (!packer->rank) {
    ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
    if (varray+mine->rstart == array) SETERRQ(1,"You need not VecPackGather() into objects obtained via VecPackGetAccess()");
    ierr    = PetscMemcpy(varray+mine->rstart,array,mine->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr    = VecRestoreArray(vec,&varray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGather_DA"
int VecPackGather_DA(VecPack packer,struct VecPackLink *mine,Vec vec,Vec local)
{
  int    ierr;
  PetscScalar *array;
  Vec    global;

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
         VecPackRestoreAccess()

@*/
int VecPackGetAccess(VecPack packer,Vec gvec,...)
{
  va_list            Argp;
  int                ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (!packer->globalvector) {
    SETERRQ(1,"Must first create global vector with VecPackCreateGlobalVector()");
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
      SETERRQ(1,"Cannot handle that object type yet");
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
int VecPackRestoreAccess(VecPack packer,Vec gvec,...)
{
  va_list            Argp;
  int                ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (!packer->globalvector) {
    SETERRQ(1,"Must first create global vector with VecPackCreateGlobalVector()");
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
      SETERRQ(1,"Cannot handle that object type yet");
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
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess()

@*/
int VecPackScatter(VecPack packer,Vec gvec,...)
{
  va_list            Argp;
  int                ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (!packer->globalvector) {
    SETERRQ(1,"Must first create global vector with VecPackCreateGlobalVector()");
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
      SETERRQ(1,"Cannot handle that object type yet");
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
         VecPackScatter(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess()

@*/
int VecPackGather(VecPack packer,Vec gvec,...)
{
  va_list            Argp;
  int                ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;
  if (!packer->globalvector) {
    SETERRQ(1,"Must first create global vector with VecPackCreateGlobalVector()");
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
      SETERRQ(1,"Cannot handle that object type yet");
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
         VecPackScatter(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess()

@*/
int VecPackAddArray(VecPack packer,int n)
{
  struct VecPackLink *mine,*next = packer->next;
  int ierr;

  PetscFunctionBegin;
  if (packer->globalvector) {
    SETERRQ(1,"Cannot add an array once you have called VecPackCreateGlobalVector()");
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
         VecPackScatter(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess()

@*/
int VecPackAddDA(VecPack packer,DA da)
{
  int                ierr,n;
  struct VecPackLink *mine,*next = packer->next;
  Vec                global;

  PetscFunctionBegin;
  if (packer->globalvector) {
    SETERRQ(1,"Cannot add a DA once you have called VecPackCreateGlobalVector()");
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
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess()

@*/
int VecPackCreateGlobalVector(VecPack packer,Vec *gvec)
{
  int                ierr,nprev = 0,rank;
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
.    idx - the individual indices for each packed vector/array
 
    Level: advanced

    Notes:
       The idx parameters should be freed by the calling routine with PetscFree()

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackCreateGlobalVector(),
         VecPackGather(), VecPackCreate(), VecPackGetAccess()

@*/
int VecPackGetGlobalIndices(VecPack packer,...)
{
  va_list            Argp;
  int                ierr,i,**idx,n;
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
    idx = va_arg(Argp, int**);

    if (next->type == VECPACK_ARRAY) {
      
      ierr = PetscMalloc(next->n*sizeof(int),idx);CHKERRQ(ierr);
      if (!packer->rank) {
        ierr   = VecGetArray(global,&array);CHKERRQ(ierr);
        array += next->rstart;
        for (i=0; i<next->n; i++) (*idx)[i] = (int)PetscRealPart(array[i]);
        array -= next->rstart;
        ierr   = VecRestoreArray(global,&array);CHKERRQ(ierr);
      }
      ierr = MPI_Bcast(*idx,next->n,MPI_INT,0,packer->comm);CHKERRQ(ierr);

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
      ierr   = PetscMalloc(n*sizeof(int),idx);CHKERRQ(ierr);
      for (i=0; i<n; i++) (*idx)[i] = (int)PetscRealPart(array[i]);
      ierr    = VecRestoreArray(local,&array);CHKERRQ(ierr);
      ierr    = VecDestroy(local);CHKERRQ(ierr);

    } else {
      SETERRQ(1,"Cannot handle that object type yet");
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
int VecPackGetLocalVectors_Array(VecPack packer,struct VecPackLink *mine,PetscScalar **array)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(mine->n*sizeof(PetscScalar),array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetLocalVectors_DA"
int VecPackGetLocalVectors_DA(VecPack packer,struct VecPackLink *mine,Vec *local)
{
  int    ierr;
  PetscFunctionBegin;
  ierr = DAGetLocalVector(mine->da,local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreLocalVectors_Array"
int VecPackRestoreLocalVectors_Array(VecPack packer,struct VecPackLink *mine,PetscScalar **array)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscFree(*array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackRestoreLocalVectors_DA"
int VecPackRestoreLocalVectors_DA(VecPack packer,struct VecPackLink *mine,Vec *local)
{
  int    ierr;
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
int VecPackGetLocalVectors(VecPack packer,...)
{
  va_list            Argp;
  int                ierr;
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
      SETERRQ(1,"Cannot handle that object type yet");
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
int VecPackRestoreLocalVectors(VecPack packer,...)
{
  va_list            Argp;
  int                ierr;
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
      SETERRQ(1,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecPackGetEntries_Array"
int VecPackGetEntries_Array(VecPack packer,struct VecPackLink *mine,int *n)
{
  PetscFunctionBegin;
  if (n) *n = mine->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPackGetEntries_DA"
int VecPackGetEntries_DA(VecPack packer,struct VecPackLink *mine,DA *da)
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
         VecPackRestoreLocalVectors(), VecPackGetLocalVectors(), VecPackRestoreEntries()

@*/
int VecPackGetEntries(VecPack packer,...)
{
  va_list            Argp;
  int                ierr;
  struct VecPackLink *next = packer->next;

  PetscFunctionBegin;

  /* loop over packed objects, handling one at at time */
  va_start(Argp,packer);
  while (next) {
    if (next->type == VECPACK_ARRAY) {
      int *n;
      n = va_arg(Argp, int*);
      ierr = VecPackGetEntries_Array(packer,next,n);CHKERRQ(ierr);
    } else if (next->type == VECPACK_DA) {
      DA *da;
      da = va_arg(Argp, DA*);
      ierr = VecPackGetEntries_DA(packer,next,da);CHKERRQ(ierr);
    } else {
      SETERRQ(1,"Cannot handle that object type yet");
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
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess()

@*/
int VecPackRefine(VecPack packer,MPI_Comm comm,VecPack *fine)
{
  int                ierr;
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
      SETERRQ(1,"Cannot handle that object type yet");
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
int MatMultBoth_Shell_Pack(Mat A,Vec x,Vec y,PetscTruth add)
{
  struct MatPack     *mpack;
  struct VecPackLink *xnext,*ynext;
  struct MatPackLink *anext;
  PetscScalar        *xarray,*yarray;
  int                ierr,i;
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
      SETERRQ(1,"Cannot handle that object type yet");
    }
    xnext = xnext->next;
    ynext = ynext->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_Shell_Pack"
int MatMultAdd_Shell_Pack(Mat A,Vec x,Vec y,Vec z)
{
  int ierr;
  PetscFunctionBegin;
  if (z != y) SETERRQ(1,"Handles y == z only");
  ierr = MatMultBoth_Shell_Pack(A,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Shell_Pack"
int MatMult_Shell_Pack(Mat A,Vec x,Vec y)
{
  int ierr;
  PetscFunctionBegin;
  ierr = MatMultBoth_Shell_Pack(A,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Shell_Pack"
int MatMultTranspose_Shell_Pack(Mat A,Vec x,Vec y)
{
  struct MatPack     *mpack;
  struct VecPackLink *xnext,*ynext;
  struct MatPackLink *anext;
  PetscScalar        *xarray,*yarray;
  int                ierr;
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
      SETERRQ(1,"Cannot handle that object type yet");
    }
    xnext = xnext->next;
    ynext = ynext->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Shell_Pack"
int MatDestroy_Shell_Pack(Mat A)
{
  struct MatPack     *mpack;
  struct MatPackLink *anext,*oldanext;
  int                ierr;

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
         VecPackGather(), VecPackCreate(), VecPackGetGlobalIndices(), VecPackGetAccess()

@*/
int VecPackGetInterpolation(VecPack coarse,VecPack fine,Mat *A,Vec *v)
{
  int                ierr,m,n,M,N;
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
  ierr  = MatCreate(fine->comm,m,n,M,N,A);CHKERRQ(ierr);
  ierr  = MatSetType(*A,MATSHELL);CHKERRQ(ierr);
  ierr  = MatShellSetContext(*A,mpack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))MatMult_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_DESTROY,(void(*)(void))MatDestroy_Shell_Pack);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (nextc) {
    if (nextc->type != nextf->type) SETERRQ(1,"Two VecPack have different layout");

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
      SETERRQ(1,"Cannot handle that object type yet");
    }
    nextc = nextc->next;
    nextf = nextf->next;
  }
  PetscFunctionReturn(0);
}





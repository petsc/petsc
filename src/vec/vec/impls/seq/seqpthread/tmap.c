#include <petsc-private/vecimpl.h>  /*I   "petscvec.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutCreate"
/*
     PetscThreadsLayoutCreate - Allocates PetsThreadscLayout space and sets the map contents to the default.


   Input Parameters:
.    map - pointer to the map

   Level: developer

.seealso: PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetLocalSizes(), PetscThreadsLayout, 
          PetscThreadsLayoutDestroy(), PetscThreadsLayoutSetUp()
*/
PetscErrorCode PetscThreadsLayoutCreate(PetscThreadsLayout *tmap)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscThreadsLayout,tmap);CHKERRQ(ierr);
  (*tmap)->nthreads = -1;
  (*tmap)->N        = -1;
  (*tmap)->trstarts =  0;
  (*tmap)->affinity =  0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutDestroy"
/*
     PetscThreadsLayoutDestroy - Frees a map object and frees its range if that exists.

   Input Parameters:
.    map - the PetscThreadsLayout

   Level: developer

      The PetscThreadsLayout object and methods are intended to be used in the PETSc threaded Vec and Mat implementions; it is 
      recommended they not be used in user codes unless you really gain something in their use.

.seealso: PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetLocalSizes(), PetscThreadsLayout, 
          PetscThreadsLayoutSetUp()
*/
PetscErrorCode PetscThreadsLayoutDestroy(PetscThreadsLayout *tmap)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if(!*tmap) PetscFunctionReturn(0);
  ierr = PetscFree((*tmap)->trstarts);CHKERRQ(ierr);
  ierr = PetscFree((*tmap)->affinity);CHKERRQ(ierr);
  ierr = PetscFree((*tmap));CHKERRQ(ierr);
  *tmap = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetUp"
/*
     PetscThreadsLayoutSetUp - given a map where you have set the thread count, either global size or
           local sizes sets up the map so that it may be used.

   Input Parameters:
.    map - pointer to the map

   Level: developer

   Notes: Typical calling sequence
      PetscThreadsLayoutCreate(PetscThreadsLayout *);
      PetscThreadsLayoutSetNThreads(PetscThreadsLayout,nthreads);
      PetscThreadsLayoutSetSize(PetscThreadsLayout,N) or PetscThreadsLayoutSetLocalSizes(PetscThreadsLayout, *n); or both
      PetscThreadsLayoutSetUp(PetscThreadsLayout);

       If the local sizes, global size are already set and row offset exists then this does nothing.

.seealso: PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetLocalSizes(), PetscThreadsLayout, 
          PetscThreadsLayoutDestroy()
*/
PetscErrorCode PetscThreadsLayoutSetUp(PetscThreadsLayout tmap)
{
  PetscErrorCode     ierr;
  PetscInt           t,rstart=0,n,Q,R;
  PetscBool          S;
  
  PetscFunctionBegin;
  if(!tmap->nthreads) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Number of threads not set yet");
  if((tmap->N >= 0) && (tmap->trstarts)) PetscFunctionReturn(0);
  ierr = PetscMalloc((tmap->nthreads+1)*sizeof(PetscInt),&tmap->trstarts);CHKERRQ(ierr);

  Q = tmap->N/tmap->nthreads;
  R = tmap->N - Q*tmap->nthreads;
  for(t=0;t < tmap->nthreads;t++) {
    tmap->trstarts[t] = rstart;
    S               = (PetscBool)(t<R);
    n               = S?Q+1:Q;
    rstart         += n;
  }
  tmap->trstarts[tmap->nthreads] = rstart;
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutDuplicate"
/*

    PetscThreadsLayoutDuplicate - creates a new PetscThreadsLayout with the same information as a given one. If the PetscThreadsLayout already exists it is destroyed first.

     Collective on PetscThreadsLayout

    Input Parameter:
.     in - input PetscThreadsLayout to be copied

    Output Parameter:
.     out - the copy

   Level: developer

    Notes: PetscThreadsLayoutSetUp() does not need to be called on the resulting PetscThreadsLayout

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutDestroy(), PetscThreadsLayoutSetUp()
*/
PetscErrorCode PetscThreadsLayoutDuplicate(PetscThreadsLayout in,PetscThreadsLayout *out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscThreadsLayoutDestroy(out);CHKERRQ(ierr);
  ierr = PetscThreadsLayoutCreate(out);CHKERRQ(ierr);
  ierr = PetscMemcpy(*out,in,sizeof(struct _n_PetscThreadsLayout));CHKERRQ(ierr);

  ierr = PetscMalloc(in->nthreads*sizeof(PetscInt),&(*out)->trstarts);CHKERRQ(ierr);
  ierr = PetscMemcpy((*out)->trstarts,in->trstarts,in->nthreads*sizeof(PetscInt));CHKERRQ(ierr);
  
  ierr = PetscMalloc(in->nthreads*sizeof(PetscInt),&(*out)->affinity);CHKERRQ(ierr);
  ierr = PetscMemcpy((*out)->affinity,in->affinity,in->nthreads*sizeof(PetscInt));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetLocalSizes"
/*
     PetscThreadsLayoutSetLocalSizes - Sets the local size for each thread 

   Input Parameters:
+    map - pointer to the map
-    n - local sizes

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutDestroy(), PetscThreadsLayoutGetLocalSizes()

*/
PetscErrorCode PetscThreadsLayoutSetLocalSizes(PetscThreadsLayout tmap,PetscInt n[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if(!tmap->nthreads) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Number of threads not set yet");
  if (tmap->trstarts) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already set local sizes");
  ierr = PetscMalloc((tmap->nthreads+1)*sizeof(PetscInt),&tmap->trstarts);CHKERRQ(ierr);
  tmap->trstarts[0] = 0;
  for(i=1;i < tmap->nthreads+1;i++) tmap->trstarts[i] += n[i-1];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutGetLocalSizes"
/*
     PetscThreadsLayoutGetLocalSizes - Gets the local size for each thread 

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    n - array to hold the local sizes (must be allocated)

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutDestroy(), PetscThreadsLayoutSetLocalSizes()
*/
PetscErrorCode PetscThreadsLayoutGetLocalSizes(PetscThreadsLayout tmap,PetscInt *n[])
{
  PetscInt i;
  PetscInt *tn=*n;
  PetscFunctionBegin;
  for(i=0;i < tmap->nthreads;i++) tn[i] = tmap->trstarts[i+1] - tmap->trstarts[i];
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetSize"
/*
     PetscThreadsLayoutSetSize - Sets the global size for PetscThreadsLayout object

   Input Parameters:
+    map - pointer to the map
-    n -   global size

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetSize()
*/
PetscErrorCode PetscThreadsLayoutSetSize(PetscThreadsLayout tmap,PetscInt N)
{
  PetscFunctionBegin;
  tmap->N = N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutGetSize"
/*
     PetscThreadsLayoutGetSize - Gets the global size 

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    n - global size

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutSetSize(), PetscThreadsLayoutGetLocalSizes()
*/
PetscErrorCode PetscThreadsLayoutGetSize(PetscThreadsLayout tmap,PetscInt *N)
{
  PetscFunctionBegin;
  *N = tmap->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetNThreads"
/*
     PetscThreadsLayoutSetNThreads - Sets the thread count for PetscThreadsLayout object

   Input Parameters:
+    map - pointer to the map
-    nthreads -   number of threads to be used with the map

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetSize()
*/
PetscErrorCode PetscThreadsLayoutSetNThreads(PetscThreadsLayout tmap,PetscInt nthreads)
{
  PetscFunctionBegin;
  tmap->nthreads = nthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetThreadAffinities"
/*
     PetscThreadsLayoutSetLocalSizes - Sets the core affinities for PetscThreadsLayout object

   Input Parameters:
+    map - pointer to the map
-    affinities - core affinities for PetscThreadsLayout 

   Level: developer

.seealso: PetscThreadsLayoutGetThreadAffinities()

*/
PetscErrorCode PetscThreadsLayoutSetThreadAffinities(PetscThreadsLayout tmap, PetscInt affinities[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(tmap->nthreads*sizeof(PetscInt),&tmap->affinity);CHKERRQ(ierr);
  ierr = PetscMemcpy(tmap->affinity,affinities,tmap->nthreads*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutGetThreadAffinities"
/*
     PetscThreadsLayoutGetThreadAffinities - Gets the core affinities of threads

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    affinity - core affinities of threads

   Level: developer

.seealso: PetscThreadsLayoutSetThreadAffinities()
*/
PetscErrorCode PetscThreadsLayoutGetThreadAffinities(PetscThreadsLayout tmap,const PetscInt *affinity[])
{
  PetscFunctionBegin;
  *affinity = tmap->affinity;
  PetscFunctionReturn(0);
}



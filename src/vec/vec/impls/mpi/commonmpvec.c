
#include <../src/vec/vec/impls/mpi/pvecimpl.h> /*I  "petscvec.h"   I*/

/*
  This is used in VecGhostGetLocalForm and VecGhostRestoreLocalForm to ensure
  that the state is updated if either vector has changed since the last time
  one of these functions was called.  It could apply to any PetscObject, but
  VecGhost is quite different from other objects in that two separate vectors
  look at the same memory.

  In principle, we could only propagate state to the local vector on
  GetLocalForm and to the global vector on RestoreLocalForm, but this version is
  more conservative (i.e. robust against misuse) and simpler.

  Note that this function is correct and changes nothing if both arguments are the
  same, which is the case in serial.
*/
static PetscErrorCode VecGhostStateSync_Private(Vec g, Vec l)
{
  PetscObjectState gstate, lstate;

  PetscFunctionBegin;
  PetscCall(PetscObjectStateGet((PetscObject)g, &gstate));
  PetscCall(PetscObjectStateGet((PetscObject)l, &lstate));
  PetscCall(PetscObjectStateSet((PetscObject)g, PetscMax(gstate, lstate)));
  PetscCall(PetscObjectStateSet((PetscObject)l, PetscMax(gstate, lstate)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    VecGhostGetLocalForm - Obtains the local ghosted representation of
    a parallel vector (obtained with `VecCreateGhost()`, `VecCreateGhostWithArray()` or `VecCreateSeq()`).

    Logically Collective

    Input Parameter:
.   g - the global vector

    Output Parameter:
.   l - the local (ghosted) representation,`NULL` if `g` is not ghosted

    Level: advanced

    Notes:
    This routine does not actually update the ghost values, but rather it
    returns a sequential vector that includes the locations for the ghost
    values and their current values. The returned vector and the original
    vector passed in share the same array that contains the actual vector data.

    To update the ghost values from the locations on the other processes one must call
    `VecGhostUpdateBegin()` and `VecGhostUpdateEnd()` before accessing the ghost values. Thus normal
    usage is
.vb
     VecGhostUpdateBegin(x,INSERT_VALUES,SCATTER_FORWARD);
     VecGhostUpdateEnd(x,INSERT_VALUES,SCATTER_FORWARD);
     VecGhostGetLocalForm(x,&xlocal);
     VecGetArray(xlocal,&xvalues);
        // access the non-ghost values in locations xvalues[0:n-1] and ghost values in locations xvalues[n:n+nghost];
     VecRestoreArray(xlocal,&xvalues);
     VecGhostRestoreLocalForm(x,&xlocal);
.ve

    One should call `VecGhostRestoreLocalForm()` or `VecDestroy()` once one is
    finished using the object.

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecCreateGhost()`, `VecGhostRestoreLocalForm()`, `VecCreateGhostWithArray()`
@*/
PetscErrorCode VecGhostGetLocalForm(Vec g, Vec *l)
{
  PetscBool isseq, ismpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g, VEC_CLASSID, 1);
  PetscValidPointer(l, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)g, VECSEQ, &isseq));
  PetscCall(PetscObjectTypeCompare((PetscObject)g, VECMPI, &ismpi));
  if (ismpi) {
    Vec_MPI *v = (Vec_MPI *)g->data;
    *l         = v->localrep;
  } else if (isseq) {
    *l = g;
  } else {
    *l = NULL;
  }
  if (*l) {
    PetscCall(VecGhostStateSync_Private(g, *l));
    PetscCall(PetscObjectReference((PetscObject)*l));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    VecGhostIsLocalForm - Checks if a given vector is the local form of a global vector

    Not Collective

    Input Parameters:
+   g - the global vector
-   l - the local vector

    Output Parameter:
.   flg - `PETSC_TRUE` if `l` is the local form

    Level: advanced

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecCreateGhost()`, `VecGhostRestoreLocalForm()`, `VecCreateGhostWithArray()`, `VecGhostGetLocalForm()`
@*/
PetscErrorCode VecGhostIsLocalForm(Vec g, Vec l, PetscBool *flg)
{
  PetscBool isseq, ismpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(l, VEC_CLASSID, 2);

  *flg = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare((PetscObject)g, VECSEQ, &isseq));
  PetscCall(PetscObjectTypeCompare((PetscObject)g, VECMPI, &ismpi));
  if (ismpi) {
    Vec_MPI *v = (Vec_MPI *)g->data;
    if (l == v->localrep) *flg = PETSC_TRUE;
  } else if (isseq) {
    if (l == g) *flg = PETSC_TRUE;
  } else SETERRQ(PetscObjectComm((PetscObject)g), PETSC_ERR_ARG_WRONG, "Global vector is not ghosted");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    VecGhostRestoreLocalForm - Restores the local ghosted representation of
    a parallel vector obtained with `VecGhostGetLocalForm()`.

    Logically Collective

    Input Parameters:
+   g - the global vector
-   l - the local (ghosted) representation

    Level: advanced

    Note:
    This routine does not actually update the ghost values, but rather it
    returns a sequential vector that includes the locations for the ghost values
    and their current values.

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecCreateGhost()`, `VecGhostGetLocalForm()`, `VecCreateGhostWithArray()`
@*/
PetscErrorCode VecGhostRestoreLocalForm(Vec g, Vec *l)
{
  PetscFunctionBegin;
  if (*l) {
    PetscCall(VecGhostStateSync_Private(g, *l));
    PetscCall(PetscObjectDereference((PetscObject)*l));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecGhostUpdateBegin - Begins the vector scatter to update the vector from
   local representation to global or global representation to local.

   Neighbor-wise Collective

   Input Parameters:
+  g - the vector (obtained with `VecCreateGhost()` or `VecDuplicate()`)
.  insertmode - one of `ADD_VALUES`, `MAX_VALUES`, `MIN_VALUES` or `INSERT_VALUES`
-  scattermode - one of `SCATTER_FORWARD` or `SCATTER_REVERSE`

   Level: advanced

   Notes:
   Use the following to update the ghost regions with correct values from the owning process
.vb
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Use the following to accumulate the ghost region values onto the owning processors
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
.ve

   To accumulate the ghost region values onto the owning processors and then update
   the ghost regions correctly, call the latter followed by the former, i.e.,
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecCreateGhost()`, `VecGhostUpdateEnd()`, `VecGhostGetLocalForm()`,
          `VecGhostRestoreLocalForm()`, `VecCreateGhostWithArray()`
@*/
PetscErrorCode VecGhostUpdateBegin(Vec g, InsertMode insertmode, ScatterMode scattermode)
{
  Vec_MPI  *v;
  PetscBool ismpi, isseq;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g, VEC_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)g, VECMPI, &ismpi));
  PetscCall(PetscObjectTypeCompare((PetscObject)g, VECSEQ, &isseq));
  if (ismpi) {
    v = (Vec_MPI *)g->data;
    PetscCheck(v->localrep, PetscObjectComm((PetscObject)g), PETSC_ERR_ARG_WRONG, "Vector is not ghosted");
    if (!v->localupdate) PetscFunctionReturn(PETSC_SUCCESS);
    if (scattermode == SCATTER_REVERSE) {
      PetscCall(VecScatterBegin(v->localupdate, v->localrep, g, insertmode, scattermode));
    } else {
      PetscCall(VecScatterBegin(v->localupdate, g, v->localrep, insertmode, scattermode));
    }
  } else if (isseq) {
    /* Do nothing */
  } else SETERRQ(PetscObjectComm((PetscObject)g), PETSC_ERR_ARG_WRONG, "Vector is not ghosted");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecGhostUpdateEnd - End the vector scatter to update the vector from
   local representation to global or global representation to local.

   Neighbor-wise Collective

   Input Parameters:
+  g - the vector (obtained with `VecCreateGhost()` or `VecDuplicate()`)
.  insertmode - one of `ADD_VALUES`, `MAX_VALUES`, `MIN_VALUES` or `INSERT_VALUES`
-  scattermode - one of `SCATTER_FORWARD` or `SCATTER_REVERSE`

   Level: advanced

   Notes:
   Use the following to update the ghost regions with correct values from the owning process
.vb
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Use the following to accumulate the ghost region values onto the owning processors
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
.ve

   To accumulate the ghost region values onto the owning processors and then update
   the ghost regions correctly, call the later followed by the former, i.e.,
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecCreateGhost()`, `VecGhostUpdateBegin()`, `VecGhostGetLocalForm()`,
          `VecGhostRestoreLocalForm()`, `VecCreateGhostWithArray()`
@*/
PetscErrorCode VecGhostUpdateEnd(Vec g, InsertMode insertmode, ScatterMode scattermode)
{
  Vec_MPI  *v;
  PetscBool ismpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g, VEC_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)g, VECMPI, &ismpi));
  if (ismpi) {
    v = (Vec_MPI *)g->data;
    PetscCheck(v->localrep, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Vector is not ghosted");
    if (!v->localupdate) PetscFunctionReturn(PETSC_SUCCESS);
    if (scattermode == SCATTER_REVERSE) {
      PetscCall(VecScatterEnd(v->localupdate, v->localrep, g, insertmode, scattermode));
    } else {
      PetscCall(VecScatterEnd(v->localupdate, g, v->localrep, insertmode, scattermode));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

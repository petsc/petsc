
#include <petsc-private/characteristicimpl.h> /*I "petsccharacteristic.h" I*/

PetscClassId CHARACTERISTIC_CLASSID;
PetscLogEvent  CHARACTERISTIC_SetUp, CHARACTERISTIC_Solve, CHARACTERISTIC_QueueSetup, CHARACTERISTIC_DAUpdate;
PetscLogEvent  CHARACTERISTIC_HalfTimeLocal, CHARACTERISTIC_HalfTimeRemote, CHARACTERISTIC_HalfTimeExchange;
PetscLogEvent  CHARACTERISTIC_FullTimeLocal, CHARACTERISTIC_FullTimeRemote, CHARACTERISTIC_FullTimeExchange;
PetscBool   CharacteristicRegisterAllCalled = PETSC_FALSE;
/*
   Contains the list of registered characteristic routines
*/
PetscFList  CharacteristicList = PETSC_NULL;

PetscErrorCode DMDAGetNeighborsRank(DM, PetscMPIInt []);
PetscInt DMDAGetNeighborRelative(DM, PassiveReal, PassiveReal);
PetscErrorCode DMDAMapToPeriodicDomain(DM, PetscScalar [] ); 

PetscErrorCode CharacteristicHeapSort(Characteristic, Queue, PetscInt);
PetscErrorCode CharacteristicSiftDown(Characteristic, Queue, PetscInt, PetscInt);

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicView"
PetscErrorCode CharacteristicView(Characteristic c, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)c)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(c, 1, viewer, 2);

  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (!iascii) {
    if (c->ops->view) {
      ierr = (*c->ops->view)(c, viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicDestroy"
PetscErrorCode CharacteristicDestroy(Characteristic *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*c, CHARACTERISTIC_CLASSID, 1);
  if (--((PetscObject)(*c))->refct > 0) PetscFunctionReturn(0);

  if ((*c)->ops->destroy) {
    ierr = (*(*c)->ops->destroy)((*c));CHKERRQ(ierr);
  }
  ierr = MPI_Type_free(&(*c)->itemType);CHKERRQ(ierr);
  ierr = PetscFree((*c)->queue);CHKERRQ(ierr);
  ierr = PetscFree((*c)->queueLocal);CHKERRQ(ierr);
  ierr = PetscFree((*c)->queueRemote);CHKERRQ(ierr);
  ierr = PetscFree((*c)->neighbors);CHKERRQ(ierr);
  ierr = PetscFree((*c)->needCount);CHKERRQ(ierr);
  ierr = PetscFree((*c)->localOffsets);CHKERRQ(ierr);
  ierr = PetscFree((*c)->fillCount);CHKERRQ(ierr);
  ierr = PetscFree((*c)->remoteOffsets);CHKERRQ(ierr);
  ierr = PetscFree((*c)->request);CHKERRQ(ierr);
  ierr = PetscFree((*c)->status);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CharacteristicCreate"
PetscErrorCode CharacteristicCreate(MPI_Comm comm, Characteristic *c)
{
  Characteristic newC;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(c, 2);
  *c = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = CharacteristicInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(newC, _p_Characteristic, struct _CharacteristicOps, CHARACTERISTIC_CLASSID, -1, "Characteristic", "Characteristic", "SemiLagrange", comm, CharacteristicDestroy, CharacteristicView);CHKERRQ(ierr);
  ierr = PetscLogObjectCreate(newC);CHKERRQ(ierr);
  *c = newC;

  newC->structured      = PETSC_TRUE;
  newC->numIds          = 0;
  newC->velocityDA      = PETSC_NULL;
  newC->velocity        = PETSC_NULL;
  newC->velocityOld     = PETSC_NULL;
  newC->numVelocityComp = 0;
  newC->velocityComp    = PETSC_NULL;
  newC->velocityInterp  = PETSC_NULL;
  newC->velocityInterpLocal = PETSC_NULL;
  newC->velocityCtx     = PETSC_NULL;
  newC->fieldDA         = PETSC_NULL;
  newC->field           = PETSC_NULL;
  newC->numFieldComp    = 0;
  newC->fieldComp       = PETSC_NULL;
  newC->fieldInterp     = PETSC_NULL;
  newC->fieldInterpLocal    = PETSC_NULL;
  newC->fieldCtx        = PETSC_NULL;
  newC->itemType        = PETSC_NULL;
  newC->queue           = PETSC_NULL;
  newC->queueSize       = 0;
  newC->queueMax        = 0;
  newC->queueLocal      = PETSC_NULL;
  newC->queueLocalSize  = 0;
  newC->queueLocalMax   = 0;
  newC->queueRemote     = PETSC_NULL;
  newC->queueRemoteSize = 0;
  newC->queueRemoteMax  = 0;
  newC->numNeighbors    = 0;
  newC->neighbors       = PETSC_NULL;
  newC->needCount       = PETSC_NULL;
  newC->localOffsets    = PETSC_NULL;
  newC->fillCount       = PETSC_NULL;
  newC->remoteOffsets   = PETSC_NULL;
  newC->request         = PETSC_NULL;
  newC->status          = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSetType"
/*@C
   CharacteristicSetType - Builds Characteristic for a particular solver. 

   Logically Collective on Characteristic

   Input Parameters:
+  c    - the method of characteristics context
-  type - a known method

   Options Database Key:
.  -characteristic_type <method> - Sets the method; use -help for a list 
    of available methods

   Notes:  
   See "include/petsccharacteristic.h" for available methods

  Normally, it is best to use the CharacteristicSetFromOptions() command and
  then set the Characteristic type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different Krylov methods.
  The CharacteristicSetType() routine is provided for those situations where it
  is necessary to set the iterative solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of iterative solver changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate method.  In other words, this routine is
  not for beginners.

  Level: intermediate

.keywords: Characteristic, set, method

.seealso: CharacteristicType

@*/
PetscErrorCode CharacteristicSetType(Characteristic c, const CharacteristicType type)
{
  PetscErrorCode ierr, (*r)(Characteristic);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);
  PetscValidCharPointer(type, 2);

  ierr = PetscObjectTypeCompare((PetscObject) c, type, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (c->data) {
    /* destroy the old private Characteristic context */
    ierr = (*c->ops->destroy)(c);CHKERRQ(ierr);
    c->ops->destroy = PETSC_NULL;
    c->data = 0;
  }

  ierr =  PetscFListFind(CharacteristicList, ((PetscObject)c)->comm,type,PETSC_TRUE, (void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Characteristic type given: %s", type);
  c->setupcalled = 0;
  ierr = (*r)(c);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) c, type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSetUp"
/*@
   CharacteristicSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Collective on Characteristic

   Input Parameter:
.  ksp   - iterative context obtained from CharacteristicCreate()

   Level: developer

.keywords: Characteristic, setup

.seealso: CharacteristicCreate(), CharacteristicSolve(), CharacteristicDestroy()
@*/
PetscErrorCode CharacteristicSetUp(Characteristic c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);

  if (!((PetscObject)c)->type_name){
    ierr = CharacteristicSetType(c, CHARACTERISTICDA);CHKERRQ(ierr);
  }

  if (c->setupcalled == 2) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(CHARACTERISTIC_SetUp,c,0,0,0);CHKERRQ(ierr);
  if (!c->setupcalled) {
    ierr = (*c->ops->setup)(c);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_SetUp,c,0,0,0);CHKERRQ(ierr);
  c->setupcalled = 2;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicRegister"
/*@C
  CharacteristicRegister - See CharacteristicRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode CharacteristicRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(Characteristic))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&CharacteristicList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSetVelocityInterpolation"
PetscErrorCode CharacteristicSetVelocityInterpolation(Characteristic c, DM da, Vec v, Vec vOld, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(Vec, PetscReal [], PetscInt, PetscInt [], PetscScalar [], void *), void *ctx)
{
  PetscFunctionBegin;
  c->velocityDA      = da;
  c->velocity        = v;
  c->velocityOld     = vOld;
  c->numVelocityComp = numComponents;
  c->velocityComp    = components;
  c->velocityInterp  = interp;
  c->velocityCtx     = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSetVelocityInterpolationLocal"
PetscErrorCode CharacteristicSetVelocityInterpolationLocal(Characteristic c, DM da, Vec v, Vec vOld, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(void *, PetscReal [], PetscInt, PetscInt [], PetscScalar [], void *), void *ctx)
{
  PetscFunctionBegin;
  c->velocityDA          = da;
  c->velocity            = v;
  c->velocityOld         = vOld;
  c->numVelocityComp     = numComponents;
  c->velocityComp        = components;
  c->velocityInterpLocal = interp;
  c->velocityCtx         = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSetFieldInterpolation"
PetscErrorCode CharacteristicSetFieldInterpolation(Characteristic c, DM da, Vec v, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(Vec, PetscReal [], PetscInt, PetscInt [], PetscScalar [], void *), void *ctx)
{
  PetscFunctionBegin;
#if 0
  if (numComponents > 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Fields with more than 2 components are not supported. Send mail to petsc-maint@mcs.anl.gov.");
#endif
  c->fieldDA      = da;
  c->field        = v;
  c->numFieldComp = numComponents;
  c->fieldComp    = components;
  c->fieldInterp  = interp;
  c->fieldCtx     = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSetFieldInterpolationLocal"
PetscErrorCode CharacteristicSetFieldInterpolationLocal(Characteristic c, DM da, Vec v, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(void *, PetscReal [], PetscInt, PetscInt [], PetscScalar [], void *), void *ctx)
{
  PetscFunctionBegin;
#if 0
  if (numComponents > 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Fields with more than 2 components are not supported. Send mail to petsc-maint@mcs.anl.gov.");
#endif
  c->fieldDA          = da;
  c->field            = v;
  c->numFieldComp     = numComponents;
  c->fieldComp        = components;
  c->fieldInterpLocal = interp;
  c->fieldCtx         = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSolve"
PetscErrorCode CharacteristicSolve(Characteristic c, PetscReal dt, Vec solution)
{
  CharacteristicPointDA2D Qi;
  DM                      da = c->velocityDA;
  Vec                     velocityLocal, velocityLocalOld;
  Vec                     fieldLocal;
  DMDALocalInfo           info;
  PetscScalar             **solArray;
  void                    *velocityArray;
  void                    *velocityArrayOld;
  void                    *fieldArray;
  PassiveScalar           *interpIndices;
  PassiveScalar           *velocityValues, *velocityValuesOld;
  PassiveScalar           *fieldValues;
  PetscMPIInt             rank;
  PetscInt                dim;
  PetscMPIInt             neighbors[9];
  PetscInt                dof;
  PetscInt                gx, gy;
  PetscInt                n, is, ie, js, je, comp;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  c->queueSize = 0;
  ierr = MPI_Comm_rank(((PetscObject)c)->comm, &rank);CHKERRQ(ierr);
  ierr = DMDAGetNeighborsRank(da, neighbors);CHKERRQ(ierr);
  ierr = CharacteristicSetNeighbors(c, 9, neighbors);CHKERRQ(ierr);
  ierr = CharacteristicSetUp(c);CHKERRQ(ierr);
  /* global and local grid info */
  ierr = DMDAGetInfo(da, &dim, &gx, &gy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
  is   = info.xs;          ie   = info.xs+info.xm;
  js   = info.ys;          je   = info.ys+info.ym;
  /* Allocation */
  ierr = PetscMalloc(dim*sizeof(PetscScalar),                &interpIndices);CHKERRQ(ierr);
  ierr = PetscMalloc(c->numVelocityComp*sizeof(PetscScalar), &velocityValues);CHKERRQ(ierr);
  ierr = PetscMalloc(c->numVelocityComp*sizeof(PetscScalar), &velocityValuesOld);CHKERRQ(ierr);
  ierr = PetscMalloc(c->numFieldComp*sizeof(PetscScalar),    &fieldValues);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(CHARACTERISTIC_Solve,0,0,0,0);CHKERRQ(ierr);

  /* -----------------------------------------------------------------------
     PART 1, AT t-dt/2
     -----------------------------------------------------------------------*/
  ierr = PetscLogEventBegin(CHARACTERISTIC_QueueSetup,0,0,0,0);CHKERRQ(ierr);
  /* GET POSITION AT HALF TIME IN THE PAST */
  if (c->velocityInterpLocal) {
    ierr = DMGetLocalVector(c->velocityDA, &velocityLocal);CHKERRQ(ierr);
    ierr = DMGetLocalVector(c->velocityDA, &velocityLocalOld);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(c->velocityDA, c->velocity, INSERT_VALUES, velocityLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(c->velocityDA, c->velocity, INSERT_VALUES, velocityLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(c->velocityDA, c->velocityOld, INSERT_VALUES, velocityLocalOld);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(c->velocityDA, c->velocityOld, INSERT_VALUES, velocityLocalOld);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(c->velocityDA, velocityLocal,    &velocityArray);   CHKERRQ(ierr);
    ierr = DMDAVecGetArray(c->velocityDA, velocityLocalOld, &velocityArrayOld);CHKERRQ(ierr);
  }
  ierr = PetscInfo(PETSC_NULL, "Calculating position at t_{n - 1/2}\n");CHKERRQ(ierr);
  for(Qi.j = js; Qi.j < je; Qi.j++) {
    for(Qi.i = is; Qi.i < ie; Qi.i++) {
      interpIndices[0] = Qi.i;
      interpIndices[1] = Qi.j;
      if (c->velocityInterpLocal) {
        c->velocityInterpLocal(velocityArray, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx);
      } else {
        c->velocityInterp(c->velocity, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx);
      }
      Qi.x = Qi.i - velocityValues[0]*dt/2.0;
      Qi.y = Qi.j - velocityValues[1]*dt/2.0;

      /* Determine whether the position at t - dt/2 is local */
      Qi.proc = DMDAGetNeighborRelative(da, Qi.x, Qi.y);

      /* Check for Periodic boundaries and move all periodic points back onto the domain */
      ierr = DMDAMapCoordsToPeriodicDomain(da,&(Qi.x),&(Qi.y));CHKERRQ(ierr);
      ierr = CharacteristicAddPoint(c, &Qi);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_QueueSetup,0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange,0,0,0,0);CHKERRQ(ierr);
  ierr = CharacteristicSendCoordinatesBegin(c);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange,0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(CHARACTERISTIC_HalfTimeLocal,0,0,0,0);CHKERRQ(ierr);
  /* Calculate velocity at t_n+1/2 (local values) */
  ierr = PetscInfo(PETSC_NULL, "Calculating local velocities at t_{n - 1/2}\n");CHKERRQ(ierr);
  for(n = 0; n < c->queueSize; n++) {
    Qi = c->queue[n];
    if (c->neighbors[Qi.proc] == rank) {
      interpIndices[0] = Qi.x;
      interpIndices[1] = Qi.y;
      if (c->velocityInterpLocal) {
        c->velocityInterpLocal(velocityArray,    interpIndices, c->numVelocityComp, c->velocityComp, velocityValues,    c->velocityCtx);
        c->velocityInterpLocal(velocityArrayOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx);
      } else {
        c->velocityInterp(c->velocity,    interpIndices, c->numVelocityComp, c->velocityComp, velocityValues,    c->velocityCtx);
        c->velocityInterp(c->velocityOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx);
      }
      Qi.x = 0.5*(velocityValues[0] + velocityValuesOld[0]);
      Qi.y = 0.5*(velocityValues[1] + velocityValuesOld[1]);
    }
    c->queue[n] = Qi;
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_HalfTimeLocal,0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange,0,0,0,0);CHKERRQ(ierr);
  ierr = CharacteristicSendCoordinatesEnd(c);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange,0,0,0,0);CHKERRQ(ierr);


  /* Calculate velocity at t_n+1/2 (fill remote requests) */
  ierr = PetscLogEventBegin(CHARACTERISTIC_HalfTimeRemote,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscInfo1(PETSC_NULL, "Calculating %d remote velocities at t_{n - 1/2}\n", c->queueRemoteSize);CHKERRQ(ierr);
  for(n = 0; n < c->queueRemoteSize; n++) {
    Qi = c->queueRemote[n];
    interpIndices[0] = Qi.x;
    interpIndices[1] = Qi.y;
    if (c->velocityInterpLocal) {
      c->velocityInterpLocal(velocityArray,    interpIndices, c->numVelocityComp, c->velocityComp, velocityValues,    c->velocityCtx);
      c->velocityInterpLocal(velocityArrayOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx);
    } else {
      c->velocityInterp(c->velocity,    interpIndices, c->numVelocityComp, c->velocityComp, velocityValues,    c->velocityCtx);
      c->velocityInterp(c->velocityOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx);
    }
    Qi.x = 0.5*(velocityValues[0] + velocityValuesOld[0]);
    Qi.y = 0.5*(velocityValues[1] + velocityValuesOld[1]);
    c->queueRemote[n] = Qi;
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_HalfTimeRemote,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange,0,0,0,0);CHKERRQ(ierr);
  ierr = CharacteristicGetValuesBegin(c);CHKERRQ(ierr);
  ierr = CharacteristicGetValuesEnd(c);CHKERRQ(ierr);
  if (c->velocityInterpLocal) {
    ierr = DMDAVecRestoreArray(c->velocityDA, velocityLocal,    &velocityArray);   CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(c->velocityDA, velocityLocalOld, &velocityArrayOld);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(c->velocityDA, &velocityLocal);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(c->velocityDA, &velocityLocalOld);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange,0,0,0,0);CHKERRQ(ierr);

  /* -----------------------------------------------------------------------
     PART 2, AT t-dt
     -----------------------------------------------------------------------*/

  /* GET POSITION AT t_n (local values) */
  ierr = PetscLogEventBegin(CHARACTERISTIC_FullTimeLocal,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscInfo(PETSC_NULL, "Calculating position at t_{n}\n");CHKERRQ(ierr);
  for(n = 0; n < c->queueSize; n++) {
    Qi = c->queue[n];
    Qi.x = Qi.i - Qi.x*dt;
    Qi.y = Qi.j - Qi.y*dt;

    /* Determine whether the position at t-dt is local */
    Qi.proc = DMDAGetNeighborRelative(da, Qi.x, Qi.y);

    /* Check for Periodic boundaries and move all periodic points back onto the domain */
    ierr = DMDAMapCoordsToPeriodicDomain(da,&(Qi.x),&(Qi.y));CHKERRQ(ierr);

    c->queue[n] = Qi;
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_FullTimeLocal,0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange,0,0,0,0);CHKERRQ(ierr);
  ierr = CharacteristicSendCoordinatesBegin(c);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange,0,0,0,0);CHKERRQ(ierr);

  /* GET VALUE AT FULL TIME IN THE PAST (LOCAL REQUESTS) */
  ierr = PetscLogEventBegin(CHARACTERISTIC_FullTimeLocal,0,0,0,0);CHKERRQ(ierr);
  if (c->fieldInterpLocal) {
    ierr = DMGetLocalVector(c->fieldDA, &fieldLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(c->fieldDA, c->field, INSERT_VALUES, fieldLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(c->fieldDA, c->field, INSERT_VALUES, fieldLocal);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(c->fieldDA, fieldLocal, &fieldArray);CHKERRQ(ierr);
  }
  ierr = PetscInfo(PETSC_NULL, "Calculating local field at t_{n}\n");CHKERRQ(ierr);
  for(n = 0; n < c->queueSize; n++) {
    if (c->neighbors[c->queue[n].proc] == rank) {
      interpIndices[0] = c->queue[n].x;
      interpIndices[1] = c->queue[n].y;
      if (c->fieldInterpLocal) {
        c->fieldInterpLocal(fieldArray, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx);
      } else {
        c->fieldInterp(c->field, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx);
      }
      for(comp = 0; comp < c->numFieldComp; comp++) {
        c->queue[n].field[comp] = fieldValues[comp];
      }
    }
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_FullTimeLocal,0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange,0,0,0,0);CHKERRQ(ierr);
  ierr = CharacteristicSendCoordinatesEnd(c);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange,0,0,0,0);CHKERRQ(ierr);

  /* GET VALUE AT FULL TIME IN THE PAST (REMOTE REQUESTS) */
  ierr = PetscLogEventBegin(CHARACTERISTIC_FullTimeRemote,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscInfo1(PETSC_NULL, "Calculating %d remote field points at t_{n}\n", c->queueRemoteSize);CHKERRQ(ierr);
  for(n = 0; n < c->queueRemoteSize; n++) {
    interpIndices[0] = c->queueRemote[n].x;
    interpIndices[1] = c->queueRemote[n].y;

    /* for debugging purposes */
    if (1) { /* hacked bounds test...let's do better */
      PetscScalar im = interpIndices[0]; PetscScalar jm = interpIndices[1];

      if (( im < (PetscScalar) is - 1.) || (im > (PetscScalar) ie) || (jm < (PetscScalar)  js - 1.) || (jm > (PetscScalar) je)) {
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB, "Nonlocal point: (%g,%g)", im, jm);
      }
    }

    if (c->fieldInterpLocal) {
      c->fieldInterpLocal(fieldArray, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx);
    } else {
      c->fieldInterp(c->field, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx);
    }
    for(comp = 0; comp < c->numFieldComp; comp++) {
      c->queueRemote[n].field[comp] = fieldValues[comp];
    }
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_FullTimeRemote,0,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange,0,0,0,0);CHKERRQ(ierr);
  ierr = CharacteristicGetValuesBegin(c);CHKERRQ(ierr);
  ierr = CharacteristicGetValuesEnd(c);CHKERRQ(ierr);
  if (c->fieldInterpLocal) {
    ierr = DMDAVecRestoreArray(c->fieldDA, fieldLocal, &fieldArray);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(c->fieldDA, &fieldLocal);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange,0,0,0,0);CHKERRQ(ierr);

  /* Return field of characteristics at t_n-1 */
  ierr = PetscLogEventBegin(CHARACTERISTIC_DAUpdate,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(c->fieldDA,0,0,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(c->fieldDA, solution, &solArray);CHKERRQ(ierr);
  for(n = 0; n < c->queueSize; n++) {
    Qi = c->queue[n];
    for(comp = 0; comp < c->numFieldComp; comp++) {
      solArray[Qi.j][Qi.i*dof+c->fieldComp[comp]] = Qi.field[comp];
    }
  }
  ierr = DMDAVecRestoreArray(c->fieldDA, solution, &solArray);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(CHARACTERISTIC_DAUpdate,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(CHARACTERISTIC_Solve,0,0,0,0);CHKERRQ(ierr);

  /* Cleanup */
  ierr = PetscFree(interpIndices);CHKERRQ(ierr);
  ierr = PetscFree(velocityValues);CHKERRQ(ierr);
  ierr = PetscFree(velocityValuesOld);CHKERRQ(ierr);
  ierr = PetscFree(fieldValues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSetNeighbors"
PetscErrorCode CharacteristicSetNeighbors(Characteristic c, PetscInt numNeighbors, PetscMPIInt neighbors[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  c->numNeighbors = numNeighbors;
  ierr = PetscFree(c->neighbors);CHKERRQ(ierr);
  ierr = PetscMalloc(numNeighbors * sizeof(PetscMPIInt), &c->neighbors);CHKERRQ(ierr);
  ierr = PetscMemcpy(c->neighbors, neighbors, numNeighbors * sizeof(PetscMPIInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicAddPoint"
PetscErrorCode CharacteristicAddPoint(Characteristic c, CharacteristicPointDA2D *point)
{
  PetscFunctionBegin;
  if (c->queueSize >= c->queueMax) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Exceeeded maximum queue size %d", c->queueMax);
  }
  c->queue[c->queueSize++] = *point;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSendCoordinatesBegin"
int CharacteristicSendCoordinatesBegin(Characteristic c)
{
  PetscMPIInt    rank, tag = 121;
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)c)->comm, &rank);CHKERRQ(ierr);
  ierr = CharacteristicHeapSort(c, c->queue, c->queueSize);CHKERRQ(ierr);
  ierr = PetscMemzero(c->needCount, c->numNeighbors * sizeof(PetscInt));CHKERRQ(ierr);
  for(i = 0;  i < c->queueSize; i++) {
    c->needCount[c->queue[i].proc]++;
  }
  c->fillCount[0] = 0;
  for(n = 1; n < c->numNeighbors; n++) {
    ierr = MPI_Irecv(&(c->fillCount[n]), 1, MPIU_INT, c->neighbors[n], tag, ((PetscObject)c)->comm, &(c->request[n-1]));CHKERRQ(ierr);
  }
  for(n = 1; n < c->numNeighbors; n++) {
    ierr = MPI_Send(&(c->needCount[n]), 1, MPIU_INT, c->neighbors[n], tag, ((PetscObject)c)->comm);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(c->numNeighbors-1, c->request, c->status);CHKERRQ(ierr);
  /* Initialize the remote queue */
  c->queueLocalMax  = c->localOffsets[0]  = 0;
  c->queueRemoteMax = c->remoteOffsets[0] = 0; 
  for(n = 1; n < c->numNeighbors; n++) {
    c->remoteOffsets[n] = c->queueRemoteMax;
    c->queueRemoteMax  += c->fillCount[n];
    c->localOffsets[n]  = c->queueLocalMax;
    c->queueLocalMax   += c->needCount[n];
  }
  /* HACK BEGIN */
  for(n = 1; n < c->numNeighbors; n++) {
    c->localOffsets[n] += c->needCount[0];
  }
  c->needCount[0] = 0;
  /* HACK END */
  if (c->queueRemoteMax) {
    ierr = PetscMalloc(sizeof(CharacteristicPointDA2D) * c->queueRemoteMax, &c->queueRemote);CHKERRQ(ierr);
  } else {
    c->queueRemote = PETSC_NULL;
  }
  c->queueRemoteSize = c->queueRemoteMax;

  /* Send and Receive requests for values at t_n+1/2, giving the coordinates for interpolation */
  for(n = 1; n < c->numNeighbors; n++) {
    ierr = PetscInfo2(PETSC_NULL, "Receiving %d requests for values from proc %d\n", c->fillCount[n], c->neighbors[n]);CHKERRQ(ierr);
    ierr = MPI_Irecv(&(c->queueRemote[c->remoteOffsets[n]]), c->fillCount[n], c->itemType, c->neighbors[n], tag, ((PetscObject)c)->comm, &(c->request[n-1]));CHKERRQ(ierr);
  }
  for(n = 1; n < c->numNeighbors; n++) {
    ierr = PetscInfo2(PETSC_NULL, "Sending %d requests for values from proc %d\n", c->needCount[n], c->neighbors[n]);CHKERRQ(ierr);
    ierr = MPI_Send(&(c->queue[c->localOffsets[n]]), c->needCount[n], c->itemType, c->neighbors[n], tag, ((PetscObject)c)->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicSendCoordinatesEnd"
PetscErrorCode CharacteristicSendCoordinatesEnd(Characteristic c)
{
#if 0
  PetscMPIInt rank;
  PetscInt n;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Waitall(c->numNeighbors-1, c->request, c->status);CHKERRQ(ierr);
#if 0
  ierr = MPI_Comm_rank(((PetscObject)c)->comm, &rank);CHKERRQ(ierr);
  for(n = 0; n < c->queueRemoteSize; n++) {
    if (c->neighbors[c->queueRemote[n].proc] == rank) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "This is fucked up, n = %d proc = %d", n, c->queueRemote[n].proc);
    }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicGetValuesBegin"
PetscErrorCode CharacteristicGetValuesBegin(Characteristic c)
{
  PetscMPIInt    tag = 121;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* SEND AND RECIEVE FILLED REQUESTS for velocities at t_n+1/2 */
  for(n = 1; n < c->numNeighbors; n++) {
    ierr = MPI_Irecv(&(c->queue[c->localOffsets[n]]), c->needCount[n], c->itemType, c->neighbors[n], tag, ((PetscObject)c)->comm, &(c->request[n-1]));CHKERRQ(ierr);
  }
  for(n = 1; n < c->numNeighbors; n++) {
    ierr = MPI_Send(&(c->queueRemote[c->remoteOffsets[n]]), c->fillCount[n], c->itemType, c->neighbors[n], tag, ((PetscObject)c)->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicGetValuesEnd"
PetscErrorCode CharacteristicGetValuesEnd(Characteristic c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Waitall(c->numNeighbors-1, c->request, c->status);CHKERRQ(ierr);
  /* Free queue of requests from other procs */
  ierr = PetscFree(c->queueRemote);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "CharacteristicHeapSort"
/*
  Based on code from http://linux.wku.edu/~lamonml/algor/sort/heap.html
*/
PetscErrorCode CharacteristicHeapSort(Characteristic c, Queue queue, PetscInt size)
/*---------------------------------------------------------------------*/
{
  PetscErrorCode          ierr;
  CharacteristicPointDA2D temp;
  PetscInt                n;
  
  PetscFunctionBegin;
  if (0) { /* Check the order of the queue before sorting */
    PetscInfo(PETSC_NULL, "Before Heap sort\n");
    for (n=0;  n<size; n++) {
      ierr = PetscInfo2(PETSC_NULL,"%d %d\n",n,queue[n].proc);CHKERRQ(ierr);
    }
  }

  /* SORTING PHASE */  
  for (n = (size / 2)-1; n >= 0; n--) {
    ierr = CharacteristicSiftDown(c, queue, n, size-1);CHKERRQ(ierr); /* Rich had size-1 here, Matt had size*/
  }
  for (n = size-1; n >= 1; n--) {
    temp = queue[0];
    queue[0] = queue[n];
    queue[n] = temp;
    ierr = CharacteristicSiftDown(c, queue, 0, n-1);CHKERRQ(ierr);
  }   
  if (0) { /* Check the order of the queue after sorting */
    ierr = PetscInfo(PETSC_NULL, "Avter  Heap sort\n");CHKERRQ(ierr); 
    for (n=0;  n<size; n++) {
      ierr = PetscInfo2(PETSC_NULL,"%d %d\n",n,queue[n].proc);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "CharacteristicSiftDown"
/*
  Based on code from http://linux.wku.edu/~lamonml/algor/sort/heap.html
*/
PetscErrorCode CharacteristicSiftDown(Characteristic c, Queue queue, PetscInt root, PetscInt bottom)
/*---------------------------------------------------------------------*/
{
  PetscBool                done = PETSC_FALSE;
  PetscInt                 maxChild;
  CharacteristicPointDA2D  temp;

  PetscFunctionBegin;
  while ((root*2 <= bottom) && (!done)) {
    if (root*2 == bottom)  maxChild = root * 2;
    else if (queue[root*2].proc > queue[root*2+1].proc)  maxChild = root * 2;
    else  maxChild = root * 2 + 1;

    if (queue[root].proc < queue[maxChild].proc) {
      temp = queue[root];
      queue[root] = queue[maxChild];
      queue[maxChild] = temp;
      root = maxChild;
    } else
      done = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetNeighborsRank"
/* [center, left, top-left, top, top-right, right, bottom-right, bottom, bottom-left] */
PetscErrorCode DMDAGetNeighborsRank(DM da, PetscMPIInt neighbors[])
{
  DMDABoundaryType bx, by;
  PetscBool      IPeriodic = PETSC_FALSE, JPeriodic = PETSC_FALSE;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscInt       **procs,pi,pj,pim,pip,pjm,pjp,PI,PJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) da, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, 0, 0, 0, &PI,&PJ, 0, 0, 0, &bx, &by,0, 0);

  if (bx == DMDA_BOUNDARY_PERIODIC) {
    IPeriodic = PETSC_TRUE;
  }
  if (by == DMDA_BOUNDARY_PERIODIC) {
    JPeriodic = PETSC_TRUE;
  }

  neighbors[0] = rank;
  rank = 0;
  ierr = PetscMalloc(sizeof(PetscInt*)*PJ,&procs);CHKERRQ(ierr);
  for (pj=0;pj<PJ;pj++) {
    ierr = PetscMalloc(sizeof(PetscInt)*PI,&(procs[pj]));CHKERRQ(ierr);
    for (pi=0;pi<PI;pi++) {
      procs[pj][pi] = rank;
      rank++;
    }
  }  
    
  pi = neighbors[0] % PI;
  pj = neighbors[0] / PI;
  pim = pi-1;  if (pim<0) pim=PI-1;
  pip = (pi+1)%PI;
  pjm = pj-1;  if (pjm<0) pjm=PJ-1;
  pjp = (pj+1)%PJ;

  neighbors[1] = procs[pj] [pim];
  neighbors[2] = procs[pjp][pim];
  neighbors[3] = procs[pjp][pi];
  neighbors[4] = procs[pjp][pip];
  neighbors[5] = procs[pj] [pip];
  neighbors[6] = procs[pjm][pip];
  neighbors[7] = procs[pjm][pi];
  neighbors[8] = procs[pjm][pim];

  if (!IPeriodic) {
    if (pi==0)    neighbors[1]=neighbors[2]=neighbors[8]=neighbors[0];
    if (pi==PI-1) neighbors[4]=neighbors[5]=neighbors[6]=neighbors[0];
  }

  if (!JPeriodic) {
    if (pj==0)    neighbors[6]=neighbors[7]=neighbors[8]=neighbors[0];
    if (pj==PJ-1) neighbors[2]=neighbors[3]=neighbors[4]=neighbors[0];
  }

  for(pj = 0; pj < PJ; pj++) {
    ierr = PetscFree(procs[pj]);CHKERRQ(ierr);
  }
  ierr = PetscFree(procs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetNeighborRelative"
/*
  SUBDOMAIN NEIGHBORHOOD PROCESS MAP: 
    2 | 3 | 4
    __|___|__
    1 | 0 | 5   
    __|___|__
    8 | 7 | 6
      |   |
*/
PetscInt DMDAGetNeighborRelative(DM da, PassiveReal ir, PassiveReal jr)
{
  DMDALocalInfo  info;
  PassiveReal    is,ie,js,je;
  PetscErrorCode ierr;
  
  ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
  is = (PassiveReal) info.xs - 0.5; ie = (PassiveReal) info.xs + info.xm - 0.5; 
  js = (PassiveReal) info.ys - 0.5; je = (PassiveReal) info.ys + info.ym - 0.5;
  
  if (ir >= is && ir <= ie) { /* center column */
    if (jr >= js && jr <= je) {
      return 0; 
    } else if (jr < js) {
      return 7;
    } else {
      return 3;
    }
  } else if (ir < is) {     /* left column */
    if (jr >= js && jr <= je) {
      return 1;
    } else if (jr < js) {
      return 8;
    } else {
      return 2;
    }
  } else {                  /* right column */
    if (jr >= js && jr <= je) {
      return 5;
    } else if (jr < js) {
      return 6;
    } else {
      return 4;
    }
  } 
}

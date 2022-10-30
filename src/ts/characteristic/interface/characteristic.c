
#include <petsc/private/characteristicimpl.h> /*I "petsccharacteristic.h" I*/
#include <petscdmda.h>
#include <petscviewer.h>

PetscClassId  CHARACTERISTIC_CLASSID;
PetscLogEvent CHARACTERISTIC_SetUp, CHARACTERISTIC_Solve, CHARACTERISTIC_QueueSetup, CHARACTERISTIC_DAUpdate;
PetscLogEvent CHARACTERISTIC_HalfTimeLocal, CHARACTERISTIC_HalfTimeRemote, CHARACTERISTIC_HalfTimeExchange;
PetscLogEvent CHARACTERISTIC_FullTimeLocal, CHARACTERISTIC_FullTimeRemote, CHARACTERISTIC_FullTimeExchange;
/*
   Contains the list of registered characteristic routines
*/
PetscFunctionList CharacteristicList              = NULL;
PetscBool         CharacteristicRegisterAllCalled = PETSC_FALSE;

PetscErrorCode DMDAGetNeighborsRank(DM, PetscMPIInt[]);
PetscInt       DMDAGetNeighborRelative(DM, PetscReal, PetscReal);
PetscErrorCode DMDAMapToPeriodicDomain(DM, PetscScalar[]);

PetscErrorCode CharacteristicHeapSort(Characteristic, Queue, PetscInt);
PetscErrorCode CharacteristicSiftDown(Characteristic, Queue, PetscInt, PetscInt);

PetscErrorCode CharacteristicView(Characteristic c, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)c), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(c, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (!iascii) PetscTryTypeMethod(c, view, viewer);
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicDestroy(Characteristic *c)
{
  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*c, CHARACTERISTIC_CLASSID, 1);
  if (--((PetscObject)(*c))->refct > 0) PetscFunctionReturn(0);

  if ((*c)->ops->destroy) PetscCall((*(*c)->ops->destroy)((*c)));
  PetscCallMPI(MPI_Type_free(&(*c)->itemType));
  PetscCall(PetscFree((*c)->queue));
  PetscCall(PetscFree((*c)->queueLocal));
  PetscCall(PetscFree((*c)->queueRemote));
  PetscCall(PetscFree((*c)->neighbors));
  PetscCall(PetscFree((*c)->needCount));
  PetscCall(PetscFree((*c)->localOffsets));
  PetscCall(PetscFree((*c)->fillCount));
  PetscCall(PetscFree((*c)->remoteOffsets));
  PetscCall(PetscFree((*c)->request));
  PetscCall(PetscFree((*c)->status));
  PetscCall(PetscHeaderDestroy(c));
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicCreate(MPI_Comm comm, Characteristic *c)
{
  Characteristic newC;

  PetscFunctionBegin;
  PetscValidPointer(c, 2);
  *c = NULL;
  PetscCall(CharacteristicInitializePackage());

  PetscCall(PetscHeaderCreate(newC, CHARACTERISTIC_CLASSID, "Characteristic", "Characteristic", "Characteristic", comm, CharacteristicDestroy, CharacteristicView));
  *c = newC;

  newC->structured          = PETSC_TRUE;
  newC->numIds              = 0;
  newC->velocityDA          = NULL;
  newC->velocity            = NULL;
  newC->velocityOld         = NULL;
  newC->numVelocityComp     = 0;
  newC->velocityComp        = NULL;
  newC->velocityInterp      = NULL;
  newC->velocityInterpLocal = NULL;
  newC->velocityCtx         = NULL;
  newC->fieldDA             = NULL;
  newC->field               = NULL;
  newC->numFieldComp        = 0;
  newC->fieldComp           = NULL;
  newC->fieldInterp         = NULL;
  newC->fieldInterpLocal    = NULL;
  newC->fieldCtx            = NULL;
  newC->itemType            = 0;
  newC->queue               = NULL;
  newC->queueSize           = 0;
  newC->queueMax            = 0;
  newC->queueLocal          = NULL;
  newC->queueLocalSize      = 0;
  newC->queueLocalMax       = 0;
  newC->queueRemote         = NULL;
  newC->queueRemoteSize     = 0;
  newC->queueRemoteMax      = 0;
  newC->numNeighbors        = 0;
  newC->neighbors           = NULL;
  newC->needCount           = NULL;
  newC->localOffsets        = NULL;
  newC->fillCount           = NULL;
  newC->remoteOffsets       = NULL;
  newC->request             = NULL;
  newC->status              = NULL;
  PetscFunctionReturn(0);
}

/*@C
   CharacteristicSetType - Builds Characteristic for a particular solver.

   Logically Collective on c

   Input Parameters:
+  c    - the method of characteristics context
-  type - a known method

   Options Database Key:
.  -characteristic_type <method> - Sets the method; use -help for a list
    of available methods

  Level: intermediate

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

.seealso: [](chapter_ts), `CharacteristicType`
@*/
PetscErrorCode CharacteristicSetType(Characteristic c, CharacteristicType type)
{
  PetscBool match;
  PetscErrorCode (*r)(Characteristic);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);
  PetscValidCharPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)c, type, &match));
  if (match) PetscFunctionReturn(0);

  if (c->data) {
    /* destroy the old private Characteristic context */
    PetscUseTypeMethod(c, destroy);
    c->ops->destroy = NULL;
    c->data         = NULL;
  }

  PetscCall(PetscFunctionListFind(CharacteristicList, type, &r));
  PetscCheck(r, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Characteristic type given: %s", type);
  c->setupcalled = 0;
  PetscCall((*r)(c));
  PetscCall(PetscObjectChangeTypeName((PetscObject)c, type));
  PetscFunctionReturn(0);
}

/*@
   CharacteristicSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Collective on c

   Input Parameter:
.  ksp   - iterative context obtained from CharacteristicCreate()

   Level: developer

.seealso: [](chapter_ts), `CharacteristicCreate()`, `CharacteristicSolve()`, `CharacteristicDestroy()`
@*/
PetscErrorCode CharacteristicSetUp(Characteristic c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);

  if (!((PetscObject)c)->type_name) PetscCall(CharacteristicSetType(c, CHARACTERISTICDA));

  if (c->setupcalled == 2) PetscFunctionReturn(0);

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_SetUp, c, NULL, NULL, NULL));
  if (!c->setupcalled) PetscUseTypeMethod(c, setup);
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_SetUp, c, NULL, NULL, NULL));
  c->setupcalled = 2;
  PetscFunctionReturn(0);
}

/*@C
  CharacteristicRegister -  Adds a solver to the method of characteristics package.

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
-  routine_create - routine to create method context

  Level: advanced

  Sample usage:
.vb
    CharacteristicRegister("my_char", MyCharCreate);
.ve

  Then, your Characteristic type can be chosen with the procedural interface via
.vb
    CharacteristicCreate(MPI_Comm, Characteristic* &char);
    CharacteristicSetType(char,"my_char");
.ve
   or at runtime via the option
.vb
    -characteristic_type my_char
.ve

   Notes:
   CharacteristicRegister() may be called multiple times to add several user-defined solvers.

.seealso: [](chapter_ts), `CharacteristicRegisterAll()`, `CharacteristicRegisterDestroy()`
@*/
PetscErrorCode CharacteristicRegister(const char sname[], PetscErrorCode (*function)(Characteristic))
{
  PetscFunctionBegin;
  PetscCall(CharacteristicInitializePackage());
  PetscCall(PetscFunctionListAdd(&CharacteristicList, sname, function));
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicSetVelocityInterpolation(Characteristic c, DM da, Vec v, Vec vOld, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(Vec, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), void *ctx)
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

PetscErrorCode CharacteristicSetVelocityInterpolationLocal(Characteristic c, DM da, Vec v, Vec vOld, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(void *, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), void *ctx)
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

PetscErrorCode CharacteristicSetFieldInterpolation(Characteristic c, DM da, Vec v, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(Vec, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), void *ctx)
{
  PetscFunctionBegin;
#if 0
  PetscCheck(numComponents <= 2,PETSC_COMM_SELF,PETSC_ERR_SUP, "Fields with more than 2 components are not supported. Send mail to petsc-maint@mcs.anl.gov.");
#endif
  c->fieldDA      = da;
  c->field        = v;
  c->numFieldComp = numComponents;
  c->fieldComp    = components;
  c->fieldInterp  = interp;
  c->fieldCtx     = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicSetFieldInterpolationLocal(Characteristic c, DM da, Vec v, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(void *, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), void *ctx)
{
  PetscFunctionBegin;
#if 0
  PetscCheck(numComponents <= 2,PETSC_COMM_SELF,PETSC_ERR_SUP, "Fields with more than 2 components are not supported. Send mail to petsc-maint@mcs.anl.gov.");
#endif
  c->fieldDA          = da;
  c->field            = v;
  c->numFieldComp     = numComponents;
  c->fieldComp        = components;
  c->fieldInterpLocal = interp;
  c->fieldCtx         = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicSolve(Characteristic c, PetscReal dt, Vec solution)
{
  CharacteristicPointDA2D Qi;
  DM                      da = c->velocityDA;
  Vec                     velocityLocal, velocityLocalOld;
  Vec                     fieldLocal;
  DMDALocalInfo           info;
  PetscScalar           **solArray;
  void                   *velocityArray;
  void                   *velocityArrayOld;
  void                   *fieldArray;
  PetscScalar            *interpIndices;
  PetscScalar            *velocityValues, *velocityValuesOld;
  PetscScalar            *fieldValues;
  PetscMPIInt             rank;
  PetscInt                dim;
  PetscMPIInt             neighbors[9];
  PetscInt                dof;
  PetscInt                gx, gy;
  PetscInt                n, is, ie, js, je, comp;

  PetscFunctionBegin;
  c->queueSize = 0;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)c), &rank));
  PetscCall(DMDAGetNeighborsRank(da, neighbors));
  PetscCall(CharacteristicSetNeighbors(c, 9, neighbors));
  PetscCall(CharacteristicSetUp(c));
  /* global and local grid info */
  PetscCall(DMDAGetInfo(da, &dim, &gx, &gy, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetLocalInfo(da, &info));
  is = info.xs;
  ie = info.xs + info.xm;
  js = info.ys;
  je = info.ys + info.ym;
  /* Allocation */
  PetscCall(PetscMalloc1(dim, &interpIndices));
  PetscCall(PetscMalloc1(c->numVelocityComp, &velocityValues));
  PetscCall(PetscMalloc1(c->numVelocityComp, &velocityValuesOld));
  PetscCall(PetscMalloc1(c->numFieldComp, &fieldValues));
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_Solve, NULL, NULL, NULL, NULL));

  /* -----------------------------------------------------------------------
     PART 1, AT t-dt/2
     -----------------------------------------------------------------------*/
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_QueueSetup, NULL, NULL, NULL, NULL));
  /* GET POSITION AT HALF TIME IN THE PAST */
  if (c->velocityInterpLocal) {
    PetscCall(DMGetLocalVector(c->velocityDA, &velocityLocal));
    PetscCall(DMGetLocalVector(c->velocityDA, &velocityLocalOld));
    PetscCall(DMGlobalToLocalBegin(c->velocityDA, c->velocity, INSERT_VALUES, velocityLocal));
    PetscCall(DMGlobalToLocalEnd(c->velocityDA, c->velocity, INSERT_VALUES, velocityLocal));
    PetscCall(DMGlobalToLocalBegin(c->velocityDA, c->velocityOld, INSERT_VALUES, velocityLocalOld));
    PetscCall(DMGlobalToLocalEnd(c->velocityDA, c->velocityOld, INSERT_VALUES, velocityLocalOld));
    PetscCall(DMDAVecGetArray(c->velocityDA, velocityLocal, &velocityArray));
    PetscCall(DMDAVecGetArray(c->velocityDA, velocityLocalOld, &velocityArrayOld));
  }
  PetscCall(PetscInfo(NULL, "Calculating position at t_{n - 1/2}\n"));
  for (Qi.j = js; Qi.j < je; Qi.j++) {
    for (Qi.i = is; Qi.i < ie; Qi.i++) {
      interpIndices[0] = Qi.i;
      interpIndices[1] = Qi.j;
      if (c->velocityInterpLocal) PetscCall(c->velocityInterpLocal(velocityArray, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
      else PetscCall(c->velocityInterp(c->velocity, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
      Qi.x = Qi.i - velocityValues[0] * dt / 2.0;
      Qi.y = Qi.j - velocityValues[1] * dt / 2.0;

      /* Determine whether the position at t - dt/2 is local */
      Qi.proc = DMDAGetNeighborRelative(da, Qi.x, Qi.y);

      /* Check for Periodic boundaries and move all periodic points back onto the domain */
      PetscCall(DMDAMapCoordsToPeriodicDomain(da, &(Qi.x), &(Qi.y)));
      PetscCall(CharacteristicAddPoint(c, &Qi));
    }
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_QueueSetup, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicSendCoordinatesBegin(c));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeLocal, NULL, NULL, NULL, NULL));
  /* Calculate velocity at t_n+1/2 (local values) */
  PetscCall(PetscInfo(NULL, "Calculating local velocities at t_{n - 1/2}\n"));
  for (n = 0; n < c->queueSize; n++) {
    Qi = c->queue[n];
    if (c->neighbors[Qi.proc] == rank) {
      interpIndices[0] = Qi.x;
      interpIndices[1] = Qi.y;
      if (c->velocityInterpLocal) {
        PetscCall(c->velocityInterpLocal(velocityArray, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
        PetscCall(c->velocityInterpLocal(velocityArrayOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx));
      } else {
        PetscCall(c->velocityInterp(c->velocity, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
        PetscCall(c->velocityInterp(c->velocityOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx));
      }
      Qi.x = 0.5 * (velocityValues[0] + velocityValuesOld[0]);
      Qi.y = 0.5 * (velocityValues[1] + velocityValuesOld[1]);
    }
    c->queue[n] = Qi;
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeLocal, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicSendCoordinatesEnd(c));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));

  /* Calculate velocity at t_n+1/2 (fill remote requests) */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeRemote, NULL, NULL, NULL, NULL));
  PetscCall(PetscInfo(NULL, "Calculating %" PetscInt_FMT " remote velocities at t_{n - 1/2}\n", c->queueRemoteSize));
  for (n = 0; n < c->queueRemoteSize; n++) {
    Qi               = c->queueRemote[n];
    interpIndices[0] = Qi.x;
    interpIndices[1] = Qi.y;
    if (c->velocityInterpLocal) {
      PetscCall(c->velocityInterpLocal(velocityArray, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
      PetscCall(c->velocityInterpLocal(velocityArrayOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx));
    } else {
      PetscCall(c->velocityInterp(c->velocity, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
      PetscCall(c->velocityInterp(c->velocityOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx));
    }
    Qi.x              = 0.5 * (velocityValues[0] + velocityValuesOld[0]);
    Qi.y              = 0.5 * (velocityValues[1] + velocityValuesOld[1]);
    c->queueRemote[n] = Qi;
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeRemote, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicGetValuesBegin(c));
  PetscCall(CharacteristicGetValuesEnd(c));
  if (c->velocityInterpLocal) {
    PetscCall(DMDAVecRestoreArray(c->velocityDA, velocityLocal, &velocityArray));
    PetscCall(DMDAVecRestoreArray(c->velocityDA, velocityLocalOld, &velocityArrayOld));
    PetscCall(DMRestoreLocalVector(c->velocityDA, &velocityLocal));
    PetscCall(DMRestoreLocalVector(c->velocityDA, &velocityLocalOld));
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));

  /* -----------------------------------------------------------------------
     PART 2, AT t-dt
     -----------------------------------------------------------------------*/

  /* GET POSITION AT t_n (local values) */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeLocal, NULL, NULL, NULL, NULL));
  PetscCall(PetscInfo(NULL, "Calculating position at t_{n}\n"));
  for (n = 0; n < c->queueSize; n++) {
    Qi   = c->queue[n];
    Qi.x = Qi.i - Qi.x * dt;
    Qi.y = Qi.j - Qi.y * dt;

    /* Determine whether the position at t-dt is local */
    Qi.proc = DMDAGetNeighborRelative(da, Qi.x, Qi.y);

    /* Check for Periodic boundaries and move all periodic points back onto the domain */
    PetscCall(DMDAMapCoordsToPeriodicDomain(da, &(Qi.x), &(Qi.y)));

    c->queue[n] = Qi;
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeLocal, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicSendCoordinatesBegin(c));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));

  /* GET VALUE AT FULL TIME IN THE PAST (LOCAL REQUESTS) */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeLocal, NULL, NULL, NULL, NULL));
  if (c->fieldInterpLocal) {
    PetscCall(DMGetLocalVector(c->fieldDA, &fieldLocal));
    PetscCall(DMGlobalToLocalBegin(c->fieldDA, c->field, INSERT_VALUES, fieldLocal));
    PetscCall(DMGlobalToLocalEnd(c->fieldDA, c->field, INSERT_VALUES, fieldLocal));
    PetscCall(DMDAVecGetArray(c->fieldDA, fieldLocal, &fieldArray));
  }
  PetscCall(PetscInfo(NULL, "Calculating local field at t_{n}\n"));
  for (n = 0; n < c->queueSize; n++) {
    if (c->neighbors[c->queue[n].proc] == rank) {
      interpIndices[0] = c->queue[n].x;
      interpIndices[1] = c->queue[n].y;
      if (c->fieldInterpLocal) PetscCall(c->fieldInterpLocal(fieldArray, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx));
      else PetscCall(c->fieldInterp(c->field, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx));
      for (comp = 0; comp < c->numFieldComp; comp++) c->queue[n].field[comp] = fieldValues[comp];
    }
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeLocal, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicSendCoordinatesEnd(c));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));

  /* GET VALUE AT FULL TIME IN THE PAST (REMOTE REQUESTS) */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeRemote, NULL, NULL, NULL, NULL));
  PetscCall(PetscInfo(NULL, "Calculating %" PetscInt_FMT " remote field points at t_{n}\n", c->queueRemoteSize));
  for (n = 0; n < c->queueRemoteSize; n++) {
    interpIndices[0] = c->queueRemote[n].x;
    interpIndices[1] = c->queueRemote[n].y;

    /* for debugging purposes */
    if (1) { /* hacked bounds test...let's do better */
      PetscScalar im = interpIndices[0];
      PetscScalar jm = interpIndices[1];

      PetscCheck((im >= (PetscScalar)is - 1.) && (im <= (PetscScalar)ie) && (jm >= (PetscScalar)js - 1.) && (jm <= (PetscScalar)je), PETSC_COMM_SELF, PETSC_ERR_LIB, "Nonlocal point: (%g,%g)", (double)PetscAbsScalar(im), (double)PetscAbsScalar(jm));
    }

    if (c->fieldInterpLocal) PetscCall(c->fieldInterpLocal(fieldArray, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx));
    else PetscCall(c->fieldInterp(c->field, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx));
    for (comp = 0; comp < c->numFieldComp; comp++) c->queueRemote[n].field[comp] = fieldValues[comp];
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeRemote, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicGetValuesBegin(c));
  PetscCall(CharacteristicGetValuesEnd(c));
  if (c->fieldInterpLocal) {
    PetscCall(DMDAVecRestoreArray(c->fieldDA, fieldLocal, &fieldArray));
    PetscCall(DMRestoreLocalVector(c->fieldDA, &fieldLocal));
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));

  /* Return field of characteristics at t_n-1 */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_DAUpdate, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetInfo(c->fieldDA, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAVecGetArray(c->fieldDA, solution, &solArray));
  for (n = 0; n < c->queueSize; n++) {
    Qi = c->queue[n];
    for (comp = 0; comp < c->numFieldComp; comp++) solArray[Qi.j][Qi.i * dof + c->fieldComp[comp]] = Qi.field[comp];
  }
  PetscCall(DMDAVecRestoreArray(c->fieldDA, solution, &solArray));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_DAUpdate, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_Solve, NULL, NULL, NULL, NULL));

  /* Cleanup */
  PetscCall(PetscFree(interpIndices));
  PetscCall(PetscFree(velocityValues));
  PetscCall(PetscFree(velocityValuesOld));
  PetscCall(PetscFree(fieldValues));
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicSetNeighbors(Characteristic c, PetscInt numNeighbors, PetscMPIInt neighbors[])
{
  PetscFunctionBegin;
  c->numNeighbors = numNeighbors;
  PetscCall(PetscFree(c->neighbors));
  PetscCall(PetscMalloc1(numNeighbors, &c->neighbors));
  PetscCall(PetscArraycpy(c->neighbors, neighbors, numNeighbors));
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicAddPoint(Characteristic c, CharacteristicPointDA2D *point)
{
  PetscFunctionBegin;
  PetscCheck(c->queueSize < c->queueMax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Exceeded maximum queue size %" PetscInt_FMT, c->queueMax);
  c->queue[c->queueSize++] = *point;
  PetscFunctionReturn(0);
}

int CharacteristicSendCoordinatesBegin(Characteristic c)
{
  PetscMPIInt rank, tag = 121;
  PetscInt    i, n;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)c), &rank));
  PetscCall(CharacteristicHeapSort(c, c->queue, c->queueSize));
  PetscCall(PetscArrayzero(c->needCount, c->numNeighbors));
  for (i = 0; i < c->queueSize; i++) c->needCount[c->queue[i].proc]++;
  c->fillCount[0] = 0;
  for (n = 1; n < c->numNeighbors; n++) PetscCallMPI(MPI_Irecv(&(c->fillCount[n]), 1, MPIU_INT, c->neighbors[n], tag, PetscObjectComm((PetscObject)c), &(c->request[n - 1])));
  for (n = 1; n < c->numNeighbors; n++) PetscCallMPI(MPI_Send(&(c->needCount[n]), 1, MPIU_INT, c->neighbors[n], tag, PetscObjectComm((PetscObject)c)));
  PetscCallMPI(MPI_Waitall(c->numNeighbors - 1, c->request, c->status));
  /* Initialize the remote queue */
  c->queueLocalMax = c->localOffsets[0] = 0;
  c->queueRemoteMax = c->remoteOffsets[0] = 0;
  for (n = 1; n < c->numNeighbors; n++) {
    c->remoteOffsets[n] = c->queueRemoteMax;
    c->queueRemoteMax += c->fillCount[n];
    c->localOffsets[n] = c->queueLocalMax;
    c->queueLocalMax += c->needCount[n];
  }
  /* HACK BEGIN */
  for (n = 1; n < c->numNeighbors; n++) c->localOffsets[n] += c->needCount[0];
  c->needCount[0] = 0;
  /* HACK END */
  if (c->queueRemoteMax) {
    PetscCall(PetscMalloc1(c->queueRemoteMax, &c->queueRemote));
  } else c->queueRemote = NULL;
  c->queueRemoteSize = c->queueRemoteMax;

  /* Send and Receive requests for values at t_n+1/2, giving the coordinates for interpolation */
  for (n = 1; n < c->numNeighbors; n++) {
    PetscCall(PetscInfo(NULL, "Receiving %" PetscInt_FMT " requests for values from proc %d\n", c->fillCount[n], c->neighbors[n]));
    PetscCallMPI(MPI_Irecv(&(c->queueRemote[c->remoteOffsets[n]]), c->fillCount[n], c->itemType, c->neighbors[n], tag, PetscObjectComm((PetscObject)c), &(c->request[n - 1])));
  }
  for (n = 1; n < c->numNeighbors; n++) {
    PetscCall(PetscInfo(NULL, "Sending %" PetscInt_FMT " requests for values from proc %d\n", c->needCount[n], c->neighbors[n]));
    PetscCallMPI(MPI_Send(&(c->queue[c->localOffsets[n]]), c->needCount[n], c->itemType, c->neighbors[n], tag, PetscObjectComm((PetscObject)c)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicSendCoordinatesEnd(Characteristic c)
{
#if 0
  PetscMPIInt rank;
  PetscInt    n;
#endif

  PetscFunctionBegin;
  PetscCallMPI(MPI_Waitall(c->numNeighbors - 1, c->request, c->status));
#if 0
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)c), &rank));
  for (n = 0; n < c->queueRemoteSize; n++) {
    PetscCheck(c->neighbors[c->queueRemote[n].proc] != rank,PETSC_COMM_SELF,PETSC_ERR_PLIB, "This is messed up, n = %d proc = %d", n, c->queueRemote[n].proc);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicGetValuesBegin(Characteristic c)
{
  PetscMPIInt tag = 121;
  PetscInt    n;

  PetscFunctionBegin;
  /* SEND AND RECEIVE FILLED REQUESTS for velocities at t_n+1/2 */
  for (n = 1; n < c->numNeighbors; n++) PetscCallMPI(MPI_Irecv(&(c->queue[c->localOffsets[n]]), c->needCount[n], c->itemType, c->neighbors[n], tag, PetscObjectComm((PetscObject)c), &(c->request[n - 1])));
  for (n = 1; n < c->numNeighbors; n++) PetscCallMPI(MPI_Send(&(c->queueRemote[c->remoteOffsets[n]]), c->fillCount[n], c->itemType, c->neighbors[n], tag, PetscObjectComm((PetscObject)c)));
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicGetValuesEnd(Characteristic c)
{
  PetscFunctionBegin;
  PetscCallMPI(MPI_Waitall(c->numNeighbors - 1, c->request, c->status));
  /* Free queue of requests from other procs */
  PetscCall(PetscFree(c->queueRemote));
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
/*
  Based on code from http://linux.wku.edu/~lamonml/algor/sort/heap.html
*/
PetscErrorCode CharacteristicHeapSort(Characteristic c, Queue queue, PetscInt size)
/*---------------------------------------------------------------------*/
{
  CharacteristicPointDA2D temp;
  PetscInt                n;

  PetscFunctionBegin;
  if (0) { /* Check the order of the queue before sorting */
    PetscCall(PetscInfo(NULL, "Before Heap sort\n"));
    for (n = 0; n < size; n++) PetscCall(PetscInfo(NULL, "%" PetscInt_FMT " %d\n", n, queue[n].proc));
  }

  /* SORTING PHASE */
  for (n = (size / 2) - 1; n >= 0; n--) { PetscCall(CharacteristicSiftDown(c, queue, n, size - 1)); /* Rich had size-1 here, Matt had size*/ }
  for (n = size - 1; n >= 1; n--) {
    temp     = queue[0];
    queue[0] = queue[n];
    queue[n] = temp;
    PetscCall(CharacteristicSiftDown(c, queue, 0, n - 1));
  }
  if (0) { /* Check the order of the queue after sorting */
    PetscCall(PetscInfo(NULL, "Avter  Heap sort\n"));
    for (n = 0; n < size; n++) PetscCall(PetscInfo(NULL, "%" PetscInt_FMT " %d\n", n, queue[n].proc));
  }
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
/*
  Based on code from http://linux.wku.edu/~lamonml/algor/sort/heap.html
*/
PetscErrorCode CharacteristicSiftDown(Characteristic c, Queue queue, PetscInt root, PetscInt bottom)
/*---------------------------------------------------------------------*/
{
  PetscBool               done = PETSC_FALSE;
  PetscInt                maxChild;
  CharacteristicPointDA2D temp;

  PetscFunctionBegin;
  while ((root * 2 <= bottom) && (!done)) {
    if (root * 2 == bottom) maxChild = root * 2;
    else if (queue[root * 2].proc > queue[root * 2 + 1].proc) maxChild = root * 2;
    else maxChild = root * 2 + 1;

    if (queue[root].proc < queue[maxChild].proc) {
      temp            = queue[root];
      queue[root]     = queue[maxChild];
      queue[maxChild] = temp;
      root            = maxChild;
    } else done = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/* [center, left, top-left, top, top-right, right, bottom-right, bottom, bottom-left] */
PetscErrorCode DMDAGetNeighborsRank(DM da, PetscMPIInt neighbors[])
{
  DMBoundaryType bx, by;
  PetscBool      IPeriodic = PETSC_FALSE, JPeriodic = PETSC_FALSE;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscInt     **procs, pi, pj, pim, pip, pjm, pjp, PI, PJ;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMDAGetInfo(da, NULL, NULL, NULL, NULL, &PI, &PJ, NULL, NULL, NULL, &bx, &by, NULL, NULL));

  if (bx == DM_BOUNDARY_PERIODIC) IPeriodic = PETSC_TRUE;
  if (by == DM_BOUNDARY_PERIODIC) JPeriodic = PETSC_TRUE;

  neighbors[0] = rank;
  rank         = 0;
  PetscCall(PetscMalloc1(PJ, &procs));
  for (pj = 0; pj < PJ; pj++) {
    PetscCall(PetscMalloc1(PI, &(procs[pj])));
    for (pi = 0; pi < PI; pi++) {
      procs[pj][pi] = rank;
      rank++;
    }
  }

  pi  = neighbors[0] % PI;
  pj  = neighbors[0] / PI;
  pim = pi - 1;
  if (pim < 0) pim = PI - 1;
  pip = (pi + 1) % PI;
  pjm = pj - 1;
  if (pjm < 0) pjm = PJ - 1;
  pjp = (pj + 1) % PJ;

  neighbors[1] = procs[pj][pim];
  neighbors[2] = procs[pjp][pim];
  neighbors[3] = procs[pjp][pi];
  neighbors[4] = procs[pjp][pip];
  neighbors[5] = procs[pj][pip];
  neighbors[6] = procs[pjm][pip];
  neighbors[7] = procs[pjm][pi];
  neighbors[8] = procs[pjm][pim];

  if (!IPeriodic) {
    if (pi == 0) neighbors[1] = neighbors[2] = neighbors[8] = neighbors[0];
    if (pi == PI - 1) neighbors[4] = neighbors[5] = neighbors[6] = neighbors[0];
  }

  if (!JPeriodic) {
    if (pj == 0) neighbors[6] = neighbors[7] = neighbors[8] = neighbors[0];
    if (pj == PJ - 1) neighbors[2] = neighbors[3] = neighbors[4] = neighbors[0];
  }

  for (pj = 0; pj < PJ; pj++) PetscCall(PetscFree(procs[pj]));
  PetscCall(PetscFree(procs));
  PetscFunctionReturn(0);
}

/*
  SUBDOMAIN NEIGHBORHOOD PROCESS MAP:
    2 | 3 | 4
    __|___|__
    1 | 0 | 5
    __|___|__
    8 | 7 | 6
      |   |
*/
PetscInt DMDAGetNeighborRelative(DM da, PetscReal ir, PetscReal jr)
{
  DMDALocalInfo info;
  PetscReal     is, ie, js, je;

  PetscCall(DMDAGetLocalInfo(da, &info));
  is = (PetscReal)info.xs - 0.5;
  ie = (PetscReal)info.xs + info.xm - 0.5;
  js = (PetscReal)info.ys - 0.5;
  je = (PetscReal)info.ys + info.ym - 0.5;

  if (ir >= is && ir <= ie) { /* center column */
    if (jr >= js && jr <= je) return 0;
    else if (jr < js) return 7;
    else return 3;
  } else if (ir < is) { /* left column */
    if (jr >= js && jr <= je) return 1;
    else if (jr < js) return 8;
    else return 2;
  } else { /* right column */
    if (jr >= js && jr <= je) return 5;
    else if (jr < js) return 6;
    else return 4;
  }
}

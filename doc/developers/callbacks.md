# How the Solvers Handle User Provided Callbacks

The solver objects in PETSc, `KSP` (optionally), `SNES`, and `TS`
require user provided callback functions (and contexts for the
functions) that define the problem to be solved. These functions are
supplied by the user with calls such as `SNESSetFunction(SNES,...)`
and `TSSetRHSFunction(TS,...)`. One would naturally think that the
functions provided would be attached to the appropriate solver object,
that is, that the SNES callbacks would be attached to the `SNES`
object and `TS` callbacks to the `TS` object. This is not the case.
Or possibly one might think the callbacks would be attached to the
`DM` object associated with the solver object. This is also not the
case. Rather, the callback functions are attached to an inner nonpublic
`DMXXX` object (`XXX` is `KSP`, `SNES`, or `TS`) that is
attached to the `DM` that is attached to the `XXX` solver object.
This convoluted design is to support multilevel and multidomain solvers
where different levels and different domains may (or may not) share the
same callback function or callback context. You can control exactly what
`XXX`/`DM` objects share a common `DMXXX` object.

:::{figure} /images/developers/callbacks1.svg
:name: fig_callbacks1

Three levels of KSP/DM share the same DMKSP
:::

In the preceding figure, we depict how three levels of `KSP`
objects share a common `DMKSP` object. The code to access the inner
`DMKSP` object is

```
DM    dm_2;
DMKSP dmksp;
KSPGetDM(ksp_2,&dm_2);
DMGetDMKSP(dm_2,&dmksp);
```

To obtain a new DMKSP object for which you can change the callback
functions (or their contexts) without affecting the original DMKSP, call

```
DM    dm_2;
DMKSP dmksp;
KSPGetDM(ksp_2,&dm_2);
DMGetDMKSPWrite(dm_2,&dmksp_2);
```

This results in the object organization as indicated in the following figure

:::{figure} /images/developers/callbacks2.svg
:name: fig_callbacks2

Two levels of KSP/DM share the same DMKSP; one has its own private copy
:::

The `DMKSP` object is essentially the list of callback functions and
their contexts, for example,

```{literalinclude} /../include/petsc/private/kspimpl.h
:end-at: };
:language: c
:start-at: typedef struct _p_DMKSP
```

```{literalinclude} /../include/petsc/private/kspimpl.h
:end-at: };
:language: c
:start-at: struct _p_DMKSP {
```

We now explore in more detail exactly how the solver calls set by the
user are passed down to the inner `DMKSP` object. For each user level
solver routine for setting a callback a similar routine exists at the
`DM` level. Thus, `XXXSetY(XXX,...)` has a routine
`DMXXXSetY(DM,...)`.

```{literalinclude} /../src/ksp/ksp/interface/itfunc.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode KSPSetComputeOperators(
```

The implementation of `DMXXXSetY(DM,...)` gets a “writable” version of
the `DMXXX` object via `DMGetDMXXXWrite(DM,DMXXX*)` and sets the
function callback and its context into the `DMXXX` object.

```{literalinclude} /../src/ksp/ksp/interface/dmksp.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode DMKSPSetComputeOperators(
```

The routine for `DMGetDMXXXWrite(DM,DMXXX*)` entails a duplication of
the object unless the `DM` associated with the `DMXXX` object is the
original `DM` that the `DMXXX` object was created with. This can be
seen in the following code.

```{literalinclude} /../src/ksp/ksp/interface/dmksp.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode DMGetDMKSPWrite(
```

The routine `DMGetDMXXX(DM,DMXXX*)` has the following form.

```{literalinclude} /../src/ksp/ksp/interface/dmksp.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode DMGetDMKSP(DM dm, DMKSP *kspdm)
```

This routine uses `DMCoarsenHookAdd()` and `DMRefineHookAdd()` to
attach to the `DM` object two functions that are automatically called
when the object is coarsened or refined. The hooks
`DMCoarsenHook_DMXXX()` and `DMRefineHook_DMXXX()` have the same form:

```{literalinclude} /../src/ksp/ksp/interface/dmksp.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: static PetscErrorCode DMCoarsenHook_DMKSP(
```

where

```{literalinclude} /../src/ksp/ksp/interface/dmksp.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode DMCopyDMKSP(
```

ensures that the new `DM` shares the same `DMXXX` as the parent
`DM` and also inherits the hooks if it is refined or coarsened.

If you provide callbacks to a solver *after* the `DM` associated with
a solver has been refined or coarsened, those child `DM`s will not
share a common `DMXXX`.

The `TS` object manages its callback functions in a way similar to
`KSP` and `SNES`, although there are no multilevel `TS`
implementations so in theory the `DMTS` object is currently unneeded.

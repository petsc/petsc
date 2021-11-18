How the Solvers Handle User Provided Callbacks
==============================================

The solver objects in PETSc, ``KSP`` (optionally), ``SNES``, and ``TS``
require user provided callback functions (and contexts for the
functions) that define the problem to be solved. These functions are
supplied by the user with calls such as ``SNESSetFunction(SNES,...)``
and ``TSSetRHSFunction(TS,...)``. One would naturally think that the
functions provided would be attached to the appropriate solver object,
that is, that the SNES callbacks would be attached to the ``SNES``
object and ``TS`` callbacks to the ``TS`` object. This is not the case.
Or possibly one might think the callbacks would be attached to the
``DM`` object associated with the solver object. This is also not the
case. Rather, the callback functions are attached to an inner nonpublic
``DMXXX`` object (``XXX`` is ``KSP``, ``SNES``, or ``TS``) that is
attached to the ``DM`` that is attached to the ``XXX`` solver object.
This convoluted design is to support multilevel and multidomain solvers
where different levels and different domains may (or may not) share the
same callback function or callback context. You can control exactly what
``XXX``/``DM`` objects share a common ``DMXXX`` object.

.. figure:: /images/developers/callbacks1.svg
  :name: fig_callbacks1

  Three levels of KSP/DM share the same DMKSP

In the preceding figure, we depict how three levels of ``KSP``
objects share a common ``DMKSP`` object. The code to access the inner
``DMKSP`` object is

::

      DM    dm_2;
      DMKSP dmksp;
      KSPGetDM(ksp_2,&dm_2);
      DMGetDMKSP(dm_2,&dmksp);

To obtain a new DMKSP object for which you can change the callback
functions (or their contexts) without affecting the original DMKSP, call

::

      DM    dm_2;
      DMKSP dmksp;
      KSPGetDM(ksp_2,&dm_2);
      DMGetDMKSPWrite(dm_2,&dmksp_2);

This results in the object organization as indicated in the following figure

.. figure:: /images/developers/callbacks2.svg
  :name: fig_callbacks2

  Two levels of KSP/DM share the same DMKSP; one has its own private copy


The ``DMKSP`` object is essentially the list of callback functions and
their contexts, for example,

::

    typedef struct _p_DMKSP *DMKSP;
    typedef struct _DMKSPOps *DMKSPOps;
    struct _DMKSPOps {
      PetscErrorCode (*computeoperators)(KSP,Mat,Mat,void*);
      PetscErrorCode (*computerhs)(KSP,Vec,void*);
      PetscErrorCode (*computeinitialguess)(KSP,Vec,void*);
      PetscErrorCode (*destroy)(DMKSP*);
      PetscErrorCode (*duplicate)(DMKSP,DMKSP);
    };

    struct _p_DMKSP {
      PETSCHEADER(struct _DMKSPOps);
      void *operatorsctx;
      void *rhsctx;
      void *initialguessctx;
      void *data;
      DM originaldm;

      void (*fortran_func_pointers[3])(void); /* Store our own function pointers so they are associated with the DMKSP instead of the DM */
    };

We now explore in more detail exactly how the solver calls set by the
user are passed down to the inner ``DMKSP`` object. For each user level
solver routine for setting a callback a similar routine exists at the
``DM`` level. Thus, ``XXXSetY(XXX,...)`` has a routine
``DMXXXSetY(DM,...)``.

::

    PetscErrorCode KSPSetComputeOperators(KSP ksp,PetscErrorCode (*func)(KSP,Mat,Mat,void*),void *ctx)
    {
      PetscErrorCode ierr;
      DM             dm;

      PetscFunctionBegin;
      PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
      ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
      ierr = DMKSPSetComputeOperators(dm,func,ctx);CHKERRQ(ierr);
      if (ksp->setupstage == KSP_SETUP_NEWRHS) ksp->setupstage = KSP_SETUP_NEWMATRIX;
      PetscFunctionReturn(0);
    }

The implementation of ``DMXXXSetY(DM,...)`` gets a “writable” version of
the ``DMXXX`` object via ``DMGetDMXXXWrite(DM,DMXXX*)`` and sets the
function callback and its context into the ``DMXXX`` object.

::

    PetscErrorCode DMKSPSetComputeOperators(DM dm,PetscErrorCode (*func)(KSP,Mat,Mat,void*),void *ctx)
    {
      PetscErrorCode ierr;
      DMKSP          kdm;

      PetscFunctionBegin;
      PetscValidHeaderSpecific(dm,DM_CLASSID,1);
      ierr = DMGetDMKSPWrite(dm,&kdm);CHKERRQ(ierr);
      if (func) kdm->ops->computeoperators = func;
      if (ctx) kdm->operatorsctx = ctx;
      PetscFunctionReturn(0);
    }

The routine for ``DMGetDMXXXWrite(DM,DMXXX*)`` entails a duplication of
the object unless the ``DM`` associated with the ``DMXXX`` object is the
original ``DM`` that the ``DMXXX`` object was created with. This can be
seen in the following code.

::

    PetscErrorCode DMGetDMKSPWrite(DM dm,DMKSP *kspdm)
    {
      PetscErrorCode ierr;
      DMKSP          kdm;

      PetscFunctionBegin;
      PetscValidHeaderSpecific(dm,DM_CLASSID,1);
      ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
      if (!kdm->originaldm) kdm->originaldm = dm;
      if (kdm->originaldm != dm) {  /* Copy on write */
        DMKSP oldkdm = kdm;
        ierr      = PetscInfo(dm,"Copying DMKSP due to write\n");CHKERRQ(ierr);
        ierr      = DMKSPCreate(PetscObjectComm((PetscObject)dm),&kdm);CHKERRQ(ierr);
        ierr      = DMKSPCopy(oldkdm,kdm);CHKERRQ(ierr);
        ierr      = DMKSPDestroy((DMKSP*)&dm->dmksp);CHKERRQ(ierr);
        dm->dmksp = (PetscObject)kdm;
        kdm->originaldm = dm;
      }
      *kspdm = kdm;
      PetscFunctionReturn(0);
    }

The routine ``DMGetDMXXX(DM,DMXXX*)`` has the following form.

::

    PetscErrorCode DMGetDMKSP(DM dm,DMKSP *kspdm)
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      PetscValidHeaderSpecific(dm,DM_CLASSID,1);
      *kspdm = (DMKSP) dm->dmksp;
      if (!*kspdm) {
        ierr      = PetscInfo(dm,"Creating new DMKSP\n");CHKERRQ(ierr);
        ierr      = DMKSPCreate(PetscObjectComm((PetscObject)dm),kspdm);CHKERRQ(ierr);
        dm->dmksp = (PetscObject) *kspdm;
        (*kspdm)->originaldm = dm;
        ierr      = DMCoarsenHookAdd(dm,DMCoarsenHook_DMKSP,NULL,NULL);CHKERRQ(ierr);
        ierr      = DMRefineHookAdd(dm,DMRefineHook_DMKSP,NULL,NULL);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }

This routine uses ``DMCoarsenHookAdd()`` and ``DMRefineHookAdd()`` to
attach to the ``DM`` object two functions that are automatically called
when the object is coarsened or refined. The hooks
``DMCoarsenHook_DMXXX()`` and ``DMRefineHook_DMXXX()`` have the same form:

::

    static PetscErrorCode DMCoarsenHook_DMKSP(DM dm,DM dmc,void *ctx)
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = DMCopyDMKSP(dm,dmc);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }

where

::

    PetscErrorCode DMCopyDMKSP(DM dmsrc,DM dmdest)
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
      PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
      ierr          = DMKSPDestroy((DMKSP*)&dmdest->dmksp);CHKERRQ(ierr);
      dmdest->dmksp = dmsrc->dmksp;
      ierr          = PetscObjectReference(dmdest->dmksp);CHKERRQ(ierr);
      ierr          = DMCoarsenHookAdd(dmdest,DMCoarsenHook_DMKSP,NULL,NULL);CHKERRQ(ierr);
      ierr          = DMRefineHookAdd(dmdest,DMRefineHook_DMKSP,NULL,NULL);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }

ensures that the new ``DM`` shares the same ``DMXXX`` as the parent
``DM`` and also inherits the hooks if it is refined or coarsened.

If you provide callbacks to a solver *after* the ``DM`` associated with
a solver has been refined or coarsened, those child ``DM``\ s will not
share a common ``DMXXX``.

The ``TS`` object manages its callback functions in a way similar to
``KSP`` and ``SNES``, although there are no multilevel ``TS``
implementations so in theory the ``DMTS`` object is currently unneeded.

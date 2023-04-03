
#include <../src/vec/pf/pfimpl.h> /*I "petscpf.h" I*/

/*
        This PF generates a function on the fly and loads it into the running
   program.
*/

static PetscErrorCode PFView_String(void *value, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "String = %s\n", (char *)value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PFDestroy_String(void *value)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PFSetFromOptions_String(PF pf, PetscOptionItems *PetscOptionsObject)
{
  PetscBool flag;
  char      value[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "String function options");
  PetscCall(PetscOptionsString("-pf_string", "Enter the function", "PFStringCreateFunction", "", value, sizeof(value), &flag));
  if (flag) PetscCall(PFStringSetFunction(pf, value));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    PFStringSetFunction - Creates a function from a string

   Collective

  Input Parameters:
+    pf - the function object
-    string - the string that defines the function

  Developer Notes:
  Currently this can be used only ONCE in a running code. It needs to be fixed to generate a new library name for each new function added.

  Requires `PETSC_HAVE_POPEN` `PETSC_USE_SHARED_LIBRARIES` `PETSC_HAVE_DYNAMIC_LIBRARIES` to use

.seealso: `PFSetFromOptions()`
*/
PetscErrorCode PFStringSetFunction(PF pf, const char *string)
{
  char      task[1024], tmp[PETSC_MAX_PATH_LEN], lib[PETSC_MAX_PATH_LEN];
  PetscBool tmpshared, wdshared, keeptmpfiles = PETSC_FALSE;
  MPI_Comm  comm;
  FILE     *fd;
  char     *data;
  PetscErrorCode (*f)(void *, PetscInt, const PetscScalar *, PetscScalar *);

  PetscFunctionBegin;
  PetscCall(PetscObjectChangeTypeName((PetscObject)pf, PFSTRING));
  /* create the new C function and compile it */
  PetscCall(PetscSharedTmp(PetscObjectComm((PetscObject)pf), &tmpshared));
  PetscCall(PetscSharedWorkingDirectory(PetscObjectComm((PetscObject)pf), &wdshared));
  if (tmpshared) { /* do it in /tmp since everyone has one */
    PetscCall(PetscGetTmp(PetscObjectComm((PetscObject)pf), tmp, PETSC_STATIC_ARRAY_LENGTH(tmp)));
    PetscCall(PetscObjectGetComm((PetscObject)pf, &comm));
  } else if (!wdshared) { /* each one does in private /tmp */
    PetscCall(PetscGetTmp(PetscObjectComm((PetscObject)pf), tmp, PETSC_STATIC_ARRAY_LENGTH(tmp)));
    comm = PETSC_COMM_SELF;
  } else { /* do it in current directory */
    PetscCall(PetscStrncpy(tmp, ".", sizeof(tmp)));
    PetscCall(PetscObjectGetComm((PetscObject)pf, &comm));
  }
  PetscCall(PetscOptionsGetBool(((PetscObject)pf)->options, ((PetscObject)pf)->prefix, "-pf_string_keep_files", &keeptmpfiles, NULL));
  PetscCall(PetscSNPrintf(task, PETSC_STATIC_ARRAY_LENGTH(task), "cd %s ; if [ ! -d ${USERNAME} ]; then mkdir ${USERNAME}; fi ; cd ${USERNAME} ; rm -f makefile petscdlib.* ; cp -f ${PETSC_DIR}/src/vec/pf/impls/string/makefile ./makefile ; ${PETSC_MAKE} NIN=%" PetscInt_FMT " NOUT=%" PetscInt_FMT " -f makefile libpetscdlib STRINGFUNCTION=\"%s\"  %s ;  sync\n", tmp, pf->dimin, pf->dimout, string, keeptmpfiles ? "; rm -f makefile petscdlib.c" : ""));

  PetscCall(PetscPOpen(comm, NULL, task, "r", &fd));
  PetscCall(PetscPClose(comm, fd));
  PetscCallMPI(MPI_Barrier(comm));

  /* load the apply function from the dynamic library */
  PetscCall(PetscSNPrintf(lib, PETSC_STATIC_ARRAY_LENGTH(lib), "%s/${USERNAME}/libpetscdlib", tmp));
  PetscCall(PetscDLLibrarySym(comm, NULL, lib, "PFApply_String", (void **)&f));
  PetscCheck(f, PetscObjectComm((PetscObject)pf), PETSC_ERR_ARG_WRONGSTATE, "Cannot find function %s", lib);

  PetscCall(PetscFree(pf->data));
  PetscCall(PetscStrallocpy(string, (char **)&data));
  PetscCall(PFSet(pf, f, NULL, PFView_String, PFDestroy_String, data));
  pf->ops->setfromoptions = PFSetFromOptions_String;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PFCreate_String(PF pf, void *value)
{
  PetscFunctionBegin;
  PetscCall(PFStringSetFunction(pf, (const char *)value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

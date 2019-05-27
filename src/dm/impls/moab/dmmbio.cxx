#include <petsc/private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/
#include <petscdmmoab.h>

static PetscErrorCode DMMoab_GetWriteOptions_Private(PetscInt fsetid, PetscInt numproc, PetscInt dim, MoabWriteMode mode, PetscInt dbglevel, const char* dm_opts, const char* extra_opts, const char** write_opts)
{
  PetscErrorCode ierr;
  char           *wopts;
  char           wopts_par[PETSC_MAX_PATH_LEN];
  char           wopts_parid[PETSC_MAX_PATH_LEN];
  char           wopts_dbg[PETSC_MAX_PATH_LEN];
  PetscFunctionBegin;

  ierr = PetscMalloc1(PETSC_MAX_PATH_LEN, &wopts);CHKERRQ(ierr);
  ierr = PetscMemzero(&wopts_par, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMemzero(&wopts_parid, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMemzero(&wopts_dbg, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);

  // do parallel read unless only one processor
  if (numproc > 1) {
    ierr = PetscSNPrintf(wopts_par, PETSC_MAX_PATH_LEN, "PARALLEL=%s;", MoabWriteModes[mode]);CHKERRQ(ierr);
    if (fsetid >= 0) {
      ierr = PetscSNPrintf(wopts_parid, PETSC_MAX_PATH_LEN, "PARALLEL_COMM=%d;", fsetid);CHKERRQ(ierr);
    }
  }

  if (dbglevel) {
    ierr = PetscSNPrintf(wopts_dbg, PETSC_MAX_PATH_LEN, "CPUTIME;DEBUG_IO=%d;", dbglevel);CHKERRQ(ierr);
  }

  ierr = PetscSNPrintf(wopts, PETSC_MAX_PATH_LEN, "%s%s%s%s%s", wopts_par, wopts_parid, wopts_dbg, (extra_opts ? extra_opts : ""), (dm_opts ? dm_opts : ""));CHKERRQ(ierr);
  *write_opts = wopts;
  PetscFunctionReturn(0);
}


/*@C
  DMMoabOutput - Output the solution vectors that are stored in the DMMoab object as tags
  along with the complete mesh data structure in the native H5M or VTK format. The H5M output file
  can be visualized directly with Paraview (if compiled with appropriate plugin) or converted
  with MOAB/tools/mbconvert to a VTK or Exodus file.

  This routine can also be used for check-pointing purposes to store a complete history of
  the solution along with any other necessary data to restart computations.

  Collective

  Input Parameters:
+ dm     - the discretization manager object containing solution in MOAB tags.
.  filename - the name of the output file: e.g., poisson.h5m
-  usrwriteopts - the parallel write options needed for serializing a MOAB mesh database. Can be NULL.
   Reference (Parallel Mesh Initialization: http://ftp.mcs.anl.gov/pub/fathom/moab-docs/contents.html#fivetwo)

  Level: intermediate

.seealso: DMMoabLoadFromFile(), DMMoabSetGlobalFieldVector()
@*/
PetscErrorCode DMMoabOutput(DM dm, const char* filename, const char* usrwriteopts)
{
  DM_Moab         *dmmoab;
  const char      *writeopts;
  PetscBool       isftype;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dmmoab = (DM_Moab*)(dm)->data;

  ierr = PetscStrendswith(filename, "h5m", &isftype);CHKERRQ(ierr);

  /* add mesh loading options specific to the DM */
  if (isftype) {
#ifdef MOAB_HAVE_MPI
    ierr = DMMoab_GetWriteOptions_Private(dmmoab->pcomm->get_id(), dmmoab->pcomm->size(), dmmoab->dim, dmmoab->write_mode,
                                          dmmoab->rw_dbglevel, dmmoab->extra_write_options, usrwriteopts, &writeopts);CHKERRQ(ierr);
#else
    ierr = DMMoab_GetWriteOptions_Private(0, 1, dmmoab->dim, dmmoab->write_mode,
                                          dmmoab->rw_dbglevel, dmmoab->extra_write_options, usrwriteopts, &writeopts);CHKERRQ(ierr);
#endif
    PetscInfo2(dm, "Writing file %s with options: %s\n", filename, writeopts);
  }
  else {
    writeopts = NULL;
  }

  /* output file, using parallel write */
  merr = dmmoab->mbiface->write_file(filename, NULL, writeopts, &dmmoab->fileset, 1); MBERRVM(dmmoab->mbiface, "Writing output of DMMoab failed.", merr);
  ierr = PetscFree(writeopts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


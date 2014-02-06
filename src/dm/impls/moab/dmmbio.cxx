#include <petsc-private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/
#include <petscdmmoab.h>

#undef __FUNCT__
#define __FUNCT__ "DMMoab_GetWriteOptions_Private"
static PetscErrorCode DMMoab_GetWriteOptions_Private(PetscInt fsetid, PetscInt numproc, PetscInt dim, MoabWriteMode mode, PetscInt dbglevel, const char* dm_opts, const char* extra_opts, const char** write_opts)
{
  PetscErrorCode ierr;
  char           wopts[PETSC_MAX_PATH_LEN];
  char           wopts_par[PETSC_MAX_PATH_LEN];
  char           wopts_parid[PETSC_MAX_PATH_LEN];
  char           wopts_dbg[PETSC_MAX_PATH_LEN];
  PetscFunctionBegin;

  ierr = PetscMemzero(&wopts,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMemzero(&wopts_par,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMemzero(&wopts_parid,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMemzero(&wopts_dbg,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);

  // do parallel read unless only one processor
  if (numproc > 1) {
    ierr = PetscSNPrintf(wopts_par, sizeof(wopts_par), "PARALLEL=%s;",MoabWriteModes[mode]);CHKERRQ(ierr);
    if (fsetid >= 0) {
      ierr = PetscSNPrintf(wopts_parid, sizeof(wopts_parid), "PARALLEL_COMM=%d;",fsetid);CHKERRQ(ierr);
    }
  }

  if (dbglevel) {
    ierr = PetscSNPrintf(wopts_dbg, sizeof(wopts_dbg), "CPUTIME;DEBUG_IO=%d;",dbglevel);CHKERRQ(ierr);
  }

  ierr = PetscSNPrintf(wopts, sizeof(wopts), "%s%s%s%s%s",wopts_par,wopts_parid,wopts_dbg,(extra_opts?extra_opts:""),(dm_opts?dm_opts:""));CHKERRQ(ierr);
  
  *write_opts=wopts;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabOutput"
/*@C
  DMMoabOutput - Output the solution vectors that are stored in the DMMoab object as tags 
  along with the complete mesh data structure in the native H5M format. This output file
  can be visualized directly with Paraview (if compiled with appropriate plugin) or converted
  with tools/mbconvert to a VTK or Exodus file.

  This routine can also be used for check-pointing purposes to store a complete history of 
  the solution along with any other necessary data to restart computations.

  Not Collective

  Input Parameters:
+ dm     - the discretization manager object containing solution in MOAB tags.
.  filename - the name of the output file: e.g., poisson.h5m
-  usrwriteopts - the parallel write options needed for serializing a MOAB mesh database. Can be NULL.
   Reference (Parallel Mesh Initialization: http://www.mcs.anl.gov/~fathom/moab-docs/html/contents.html#fivetwo)

  Level: intermediate

.keywords: discretization manager, set, component solution

.seealso: DMMoabLoadFromFile(), DMMoabSetGlobalFieldVector()
@*/
PetscErrorCode DMMoabOutput(DM dm,const char* filename,const char* usrwriteopts)
{
  DM_Moab         *dmmoab;
  const char      *writeopts;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  /* add mesh loading options specific to the DM */
  ierr = DMMoab_GetWriteOptions_Private(dmmoab->pcomm->get_id(), dmmoab->pcomm->size(), dmmoab->dim, dmmoab->write_mode,
                                          dmmoab->rw_dbglevel, dmmoab->extra_write_options, usrwriteopts, &writeopts);CHKERRQ(ierr);
  PetscInfo2(dm, "Writing file %s with options: %s\n",filename,writeopts);

  /* output file, using parallel write */
  merr = dmmoab->mbiface->write_file(filename, NULL, writeopts, &dmmoab->fileset, 1);MBERRVM(dmmoab->mbiface,"Writing output of DMMoab failed.",merr);
  PetscFunctionReturn(0);
}


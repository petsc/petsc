#include <petsc-private/dmmbimpl.h> /*I  "petscdm.h"   I*/
#include <petscdmmoab.h>

#undef __FUNCT__
#define __FUNCT__ "DMMoab_GetWriteOptions_Private"
static PetscErrorCode DMMoab_GetWriteOptions_Private(PetscInt fsetid, PetscInt numproc, PetscInt dim, MoabWriteMode mode, PetscInt dbglevel, const char* extra_opts, const char** write_opts)
{
  std::ostringstream str;

  PetscFunctionBegin;

  // do parallel read unless only one processor
  if (numproc > 1) {
    str << "PARALLEL=" << mode << ";";
    if (fsetid>=0) str << "PARALLEL_COMM=" << fsetid << ";";
  }

  if (dbglevel)
    str << "CPUTIME;DEBUG_IO=" << dbglevel << ";";

  if (extra_opts)
    str << extra_opts ;

  *write_opts = str.str().c_str();
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
  DM_Moab        *dmmoab;
  PetscInt       dbglevel=0;
  const char *writeopts;

  PetscErrorCode ierr;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  PetscBarrier((PetscObject)dm);

  /* TODO: Use command-line options to control by_rank, verbosity, MoabReadMode and extra options */
  ierr  = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for reading/writing MOAB based meshes from file", "DMMoab");
  ierr  = PetscOptionsInt("-dmmb_rw_dbg", "The verbosity level for reading and writing MOAB meshes", "dmmbutil.cxx", dbglevel, &dbglevel, NULL);CHKERRQ(ierr);
  ierr  = PetscOptionsEnd();

  /* add mesh loading options specific to the DM */
  ierr = DMMoab_GetWriteOptions_Private(dmmoab->pcomm->get_id(), dmmoab->pcomm->size(), dmmoab->dim, MOAB_PARWOPTS_WRITE_PART, dbglevel, usrwriteopts, &writeopts);CHKERRQ(ierr);
  PetscInfo2(dm, "Writing file %s with options: %s\n",filename,writeopts);

  /* output file, using parallel write */
  merr = dmmoab->mbiface->write_file(filename, NULL, writeopts, &dmmoab->fileset, 1);MBERRVM(dmmoab->mbiface,"Writing output of DMMoab failed.",merr);
  PetscFunctionReturn(0);
}


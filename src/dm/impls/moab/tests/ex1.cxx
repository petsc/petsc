static char help[] = "Simple MOAB example\n\n";

#include <petscdmmoab.h>
#include "moab/ScdInterface.hpp"

typedef struct {
  DM            dm; /* DM implementation using the MOAB interface */
  PetscLogEvent createMeshEvent;
  /* Domain and mesh definition */
  PetscInt dim;
  char     filename[PETSC_MAX_PATH_LEN];
  char     tagname[PETSC_MAX_PATH_LEN];
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(PetscStrncpy(options->filename, "", sizeof(options->filename)));
  PetscCall(PetscStrncpy(options->tagname, "petsc_tag", sizeof(options->tagname)));
  options->dim = -1;

  PetscOptionsBegin(comm, "", "MOAB example options", "DMMOAB");
  PetscCall(PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex1.cxx", options->dim, &options->dim, NULL, PETSC_DECIDE, 3));
  PetscCall(PetscOptionsString("-filename", "The file containing the mesh", "ex1.cxx", options->filename, options->filename, sizeof(options->filename), NULL));
  PetscCall(PetscOptionsString("-tagname", "The tag name from which to create a vector", "ex1.cxx", options->tagname, options->tagname, sizeof(options->tagname), &flg));
  PetscOptionsEnd();

  PetscCall(PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  moab::Interface *iface    = NULL;
  moab::Tag        tag      = NULL;
  moab::Tag        ltog_tag = NULL;
  moab::Range      range;
  PetscInt         tagsize;
  moab::ErrorCode  merr;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(user->createMeshEvent, 0, 0, 0, 0));
  PetscCall(DMMoabCreateMoab(comm, iface, &ltog_tag, &range, dm));
  std::cout << "Created DMMoab using DMMoabCreateMoab." << std::endl;
  PetscCall(DMMoabGetInterface(*dm, &iface));

  // load file and get entities of requested or max dimension
  if (strlen(user->filename) > 0) {
    merr = iface->load_file(user->filename);
    MBERRNM(merr);
    std::cout << "Read mesh from file " << user->filename << std::endl;
  } else {
    // make a simple structured mesh
    moab::ScdInterface *scdi;
    merr = iface->query_interface(scdi);

    moab::ScdBox *box;
    merr = scdi->construct_box(moab::HomCoord(0, 0, 0), moab::HomCoord(5, 5, 5), NULL, 0, box);
    MBERRNM(merr);
    user->dim = 3;
    merr      = iface->set_dimension(user->dim);
    MBERRNM(merr);
    std::cout << "Created structured 5x5x5 mesh." << std::endl;
  }
  if (-1 == user->dim) {
    moab::Range tmp_range;
    merr = iface->get_entities_by_handle(0, tmp_range);
    MBERRNM(merr);
    if (tmp_range.empty()) MBERRNM(moab::MB_FAILURE);
    user->dim = iface->dimension_from_handle(*tmp_range.rbegin());
  }
  merr = iface->get_entities_by_dimension(0, user->dim, range);
  MBERRNM(merr);
  PetscCall(DMMoabSetLocalVertices(*dm, &range));

  // get the requested tag and create if necessary
  std::cout << "Creating tag with name: " << user->tagname << ";\n";
  merr = iface->tag_get_handle(user->tagname, 1, moab::MB_TYPE_DOUBLE, tag, moab::MB_TAG_CREAT | moab::MB_TAG_DENSE);
  MBERRNM(merr);
  {
    // initialize new tag with gids
    std::vector<double> tag_vals(range.size());
    merr = iface->tag_get_data(ltog_tag, range, tag_vals.data());
    MBERRNM(merr); // read them into ints
    double *dval = tag_vals.data();
    int    *ival = reinterpret_cast<int *>(dval); // "stretch" them into doubles, from the end
    for (int i = tag_vals.size() - 1; i >= 0; i--) dval[i] = ival[i];
    merr = iface->tag_set_data(tag, range, tag_vals.data());
    MBERRNM(merr); // write them into doubles
  }
  merr = iface->tag_get_length(tag, tagsize);
  MBERRNM(merr);

  PetscCall(DMSetUp(*dm));

  // create the dmmoab and initialize its data
  PetscCall(PetscObjectSetName((PetscObject)*dm, "MOAB mesh"));
  PetscCall(PetscLogEventEnd(user->createMeshEvent, 0, 0, 0, 0));
  user->dm = *dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx           user; /* user-defined work context */
  moab::ErrorCode  merr;
  Vec              vec;
  moab::Interface *mbImpl  = NULL;
  moab::Tag        datatag = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &user.dm)); /* create the MOAB dm and the mesh */

  PetscCall(DMMoabGetInterface(user.dm, &mbImpl));
  merr = mbImpl->tag_get_handle(user.tagname, datatag);
  MBERRNM(merr);
  PetscCall(DMMoabCreateVector(user.dm, datatag, NULL, PETSC_TRUE, PETSC_FALSE, &vec)); /* create a vec from user-input tag */

  std::cout << "Created VecMoab from existing tag." << std::endl;
  PetscCall(VecDestroy(&vec));
  std::cout << "Destroyed VecMoab." << std::endl;
  PetscCall(DMDestroy(&user.dm));
  std::cout << "Destroyed DMMoab." << std::endl;
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     build:
       requires: moab

     test:

TEST*/

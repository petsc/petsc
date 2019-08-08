static char help[] = "Simple MOAB example\n\n";

#include <petscdmmoab.h>
#include "moab/ScdInterface.hpp"

typedef struct {
  DM            dm;                /* DM implementation using the MOAB interface */
  PetscLogEvent createMeshEvent;
  /* Domain and mesh definition */
  PetscInt dim;
  char filename[PETSC_MAX_PATH_LEN];
  char tagname[PETSC_MAX_PATH_LEN];
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscStrcpy(options->filename, "");CHKERRQ(ierr);
  ierr = PetscStrcpy(options->tagname, "petsc_tag");CHKERRQ(ierr);
  options->dim = -1;

  ierr = PetscOptionsBegin(comm, "", "MOAB example options", "DMMOAB");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex1.cxx", options->dim, &options->dim, NULL,PETSC_DECIDE,3);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The file containing the mesh", "ex1.cxx", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-tagname", "The tag name from which to create a vector", "ex1.cxx", options->tagname, options->tagname, sizeof(options->tagname), &flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",          DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  moab::Interface *iface=NULL;
  moab::Tag tag=NULL;
  moab::Tag ltog_tag=NULL;
  moab::Range range;
  PetscInt tagsize;
  moab::ErrorCode merr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMMoabCreateMoab(comm, iface, &ltog_tag, &range, dm);CHKERRQ(ierr);
  std::cout << "Created DMMoab using DMMoabCreateMoab." << std::endl;
  ierr = DMMoabGetInterface(*dm, &iface);CHKERRQ(ierr);

    // load file and get entities of requested or max dimension
  if (strlen(user->filename) > 0) {
    merr = iface->load_file(user->filename);MBERRNM(merr);
    std::cout << "Read mesh from file " << user->filename << std::endl;
  }
  else {
      // make a simple structured mesh
    moab::ScdInterface *scdi;
    merr = iface->query_interface(scdi);

    moab::ScdBox *box;
    merr = scdi->construct_box (moab::HomCoord(0,0,0), moab::HomCoord(5,5,5), NULL, 0, box);MBERRNM(merr);
    user->dim = 3;
    merr = iface->set_dimension(user->dim);MBERRNM(merr);
    std::cout << "Created structured 5x5x5 mesh." << std::endl;
  }
  if (-1 == user->dim) {
    moab::Range tmp_range;
    merr = iface->get_entities_by_handle(0, tmp_range);MBERRNM(merr);
    if (tmp_range.empty()) {
      MBERRNM(moab::MB_FAILURE);
    }
    user->dim = iface->dimension_from_handle(*tmp_range.rbegin());
  }
  merr = iface->get_entities_by_dimension(0, user->dim, range);MBERRNM(merr);
  ierr = DMMoabSetLocalVertices(*dm, &range);CHKERRQ(ierr);

    // get the requested tag and create if necessary
  std::cout << "Creating tag with name: " << user->tagname << ";\n";
  merr = iface->tag_get_handle(user->tagname, 1, moab::MB_TYPE_DOUBLE, tag, moab::MB_TAG_CREAT | moab::MB_TAG_DENSE);MBERRNM(merr);
  {
      // initialize new tag with gids
    std::vector<double> tag_vals(range.size());
    merr = iface->tag_get_data(ltog_tag, range, tag_vals.data());MBERRNM(merr); // read them into ints
    double *dval = tag_vals.data(); int *ival = reinterpret_cast<int*>(dval); // "stretch" them into doubles, from the end
    for (int i = tag_vals.size()-1; i >= 0; i--) dval[i] = ival[i];
    merr = iface->tag_set_data(tag, range, tag_vals.data());MBERRNM(merr); // write them into doubles
  }
  merr = iface->tag_get_length(tag, tagsize);MBERRNM(merr);

  ierr = DMSetUp(*dm);CHKERRQ(ierr);

    // create the dmmoab and initialize its data
  ierr = PetscObjectSetName((PetscObject) *dm, "MOAB mesh");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx            user;                 /* user-defined work context */
  moab::ErrorCode   merr;
  PetscErrorCode    ierr;
  Vec               vec;
  moab::Interface*  mbImpl=NULL;
  moab::Tag         datatag=NULL;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);

  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr); /* create the MOAB dm and the mesh */

  ierr = DMMoabGetInterface(user.dm, &mbImpl);CHKERRQ(ierr);
  merr = mbImpl->tag_get_handle(user.tagname, datatag);MBERRNM(merr);
  ierr = DMMoabCreateVector(user.dm, datatag, NULL, PETSC_TRUE, PETSC_FALSE,&vec);CHKERRQ(ierr); /* create a vec from user-input tag */

  std::cout << "Created VecMoab from existing tag." << std::endl;
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  std::cout << "Destroyed VecMoab." << std::endl;
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  std::cout << "Destroyed DMMoab." << std::endl;
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     build:
       requires: moab

     test:

TEST*/

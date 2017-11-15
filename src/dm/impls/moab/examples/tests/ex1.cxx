static char help[] = "Simple MOAB example\n\n";

#include <petscdmmoab.h>
#include <iostream>
#include "moab/Interface.hpp"
#include "moab/ScdInterface.hpp"
#include "MBTagConventions.hpp"

class AppCtx {
public:
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscLogEvent createMeshEvent;
  /* Domain and mesh definition */
  PetscInt dim;
  char filename[PETSC_MAX_PATH_LEN];
  char tagname[PETSC_MAX_PATH_LEN];
  moab::Interface *iface;
  moab::ParallelComm *pcomm;
  moab::Tag tag;
  moab::Tag ltog_tag;
  moab::Range range;
  PetscInt tagsize;

  AppCtx()
          : dm(NULL), dim(-1), iface(NULL), pcomm(NULL), tag(0), ltog_tag(0), tagsize(0)
      {strcpy(filename, ""); strcpy(tagname, "");}

};

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  strcpy(options->filename, "");
  strcpy(options->tagname, "");
  options->dim = -1;

  ierr = PetscOptionsBegin(comm, "", "MOAB example options", "DMMOAB");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The file containing the mesh", "ex1.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-tagname", "The tag name from which to create a vector", "ex1.c", options->tagname, options->tagname, sizeof(options->tagname), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",          DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMMoabCreateMoab(comm, user->iface, user->pcomm, user->ltog_tag, &user->range, dm);CHKERRQ(ierr);
  std::cout << "Created DMMoab using DMMoabCreateDMAndInstance." << std::endl;
  ierr = DMMoabGetInterface(*dm, &user->iface);CHKERRQ(ierr);

    // load file and get entities of requested or max dimension
  moab::ErrorCode merr;
  if (strlen(user->filename) > 0) {
    merr = user->iface->load_file(user->filename);MBERRNM(merr);
    std::cout << "Read mesh from file " << user->filename << std::endl;
  }
  else {
      // make a simple structured mesh
    moab::ScdInterface *scdi;
    merr = user->iface->query_interface(scdi);
    moab::ScdBox *box;
    merr = scdi->construct_box (moab::HomCoord(0,0,0), moab::HomCoord(2,2,2), NULL, 0, box);MBERRNM(merr);
    user->dim = 3;
    std::cout << "Created structured 2x2x2 mesh." << std::endl;
  }
  if (-1 == user->dim) {
    moab::Range tmp_range;
    merr = user->iface->get_entities_by_handle(0, tmp_range);MBERRNM(merr);
    if (tmp_range.empty()) {
      MBERRNM(moab::MB_FAILURE);
    }
    user->dim = user->iface->dimension_from_handle(*tmp_range.rbegin());
  }
  merr = user->iface->get_entities_by_dimension(0, user->dim, user->range);MBERRNM(merr);
  ierr = DMMoabSetRange(*dm, user->range);CHKERRQ(ierr);

    // get the requested tag if a name was input
  if (strlen(user->tagname)) {
    merr = user->iface->tag_get_handle(user->tagname, user->tag);MBERRNM(merr);
    moab::DataType ttype;
    merr = user->iface->tag_get_data_type(user->tag, ttype);MBERRNM(merr);
    if (ttype != moab::MB_TYPE_DOUBLE) {
      printf("Tag type must be double!.\n");
      return 1;
    }
  }
  else {
      // make a new tag
    merr = user->iface->tag_get_handle("petsc_tag", 1, moab::MB_TYPE_DOUBLE, user->tag, moab::MB_TAG_CREAT | moab::MB_TAG_DENSE); MBERRNM(merr);
      // initialize new tag with gids
    std::vector<double> tag_vals(user->range.size());
    moab::Tag gid_tag;
    merr = user->iface->tag_get_handle("GLOBAL_ID", gid_tag);MBERRNM(merr);
    merr = user->iface->tag_get_data(gid_tag, user->range, tag_vals.data());MBERRNM(merr); // read them into ints
    double *dval = tag_vals.data(); int *ival = reinterpret_cast<int*>(dval); // "stretch" them into doubles, from the end
    for (int i = tag_vals.size()-1; i >= 0; i--) dval[i] = ival[i];
    merr = user->iface->tag_set_data(user->tag, user->range, tag_vals.data());MBERRNM(merr); // write them into doubles
  }
  merr = user->iface->tag_get_length(user->tag, user->tagsize);MBERRNM(merr);

    // create the dmmoab and initialize its data
  ierr = PetscObjectSetName((PetscObject) *dm, "MOAB mesh");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;
  Vec            vec;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);

  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr); /* create the MOAB dm and the mesh */
  ierr = DMMoabCreateVector(user.dm, user.tag, 1, user.range, PETSC_TRUE, PETSC_FALSE,
                              &vec);CHKERRQ(ierr); /* create a vec from user-input tag */
  std::cout << "Created VecMoab from existing tag." << std::endl;
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  std::cout << "Destroyed VecMoab." << std::endl;
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  std::cout << "Destroyed DMMoab." << std::endl;
  ierr = PetscFinalize();
  return 0;
}

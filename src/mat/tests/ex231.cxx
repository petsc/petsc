static char help[] = "A test for MatAssembly that heavily relies on PetscSortIntWithArrayPair\n";

/*
   The characteristic of the array (about 4M in length) to sort in this test is that it has
   many duplicated values that already clustered together (around 95 duplicates per unique integer).

   It was gotten from a petsc performance bug report from the Moose project. One can use
   it for future performance study.

   Contributed-by: Fande Kong <fdkong.jd@gmail.com>, John Peterson <jwpeterson@gmail.com>
 */

// PETSc includes
#include <petscmat.h>

// C++ includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <set>

int main (int argc, char** argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size, rank;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  const unsigned int n_dofs = 26559;
  unsigned int   first_local_index;
  unsigned int   last_local_index;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (size > 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example is for <=2 procs");

  ierr = PetscOptionsGetString(NULL,NULL,"-f0",file[0],sizeof(file[0]),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate dof indices file for rank 0 with -f0 option");
  if (size == 2) {
    ierr = PetscOptionsGetString(NULL,NULL,"-f1",file[1],sizeof(file[1]),&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate dof indices file for rank 1 with -f1 option");
  }

  if (size == 1) {
    first_local_index = 0;
    last_local_index  = 26559;
  } else {
    if (rank == 0) {
      first_local_index = 0;
      last_local_index  = 13911;
    } else {
      first_local_index = 13911;
      last_local_index  = 26559;
    }
  }

  // Read element dof indices from files
  std::vector<std::vector<std::vector<PetscInt> > > elem_dof_indices(size);
  for (PetscInt proc_id = 0; proc_id < size; ++proc_id) {
    std::string line;
    std::ifstream dof_file(file[proc_id]);
    if (dof_file.good()) {
      while (std::getline (dof_file,line)) {
        std::vector<PetscInt> dof_indices;
        std::stringstream sstream(line);
        std::string token;
        while (std::getline(sstream, token, ' ')) {dof_indices.push_back(std::atoi(token.c_str()));}
        elem_dof_indices[proc_id].push_back(dof_indices);
      }
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Could not open file %s",file[proc_id]);
  }

  // Debugging: Verify we read in elem_dof_indices correctly
  // for (unsigned int i=0; i<elem_dof_indices.size(); ++i)
  //   {
  //     for (unsigned int j=0; j<elem_dof_indices[i].size(); ++j)
  //       {
  //         for (unsigned int k=0; k<elem_dof_indices[i][j].size(); ++k)
  //           std::cout << elem_dof_indices[i][j][k] << " ";
  //         std::cout << std::endl;
  //       }
  //     std::cout << std::endl;
  //   }

  // Set up the (global) sparsity pattern
  std::vector< std::set< unsigned int > > sparsity(n_dofs);
  for (PetscInt proc_id = 0; proc_id < size; ++proc_id)
    for (unsigned int k = 0; k < elem_dof_indices[proc_id].size(); k++) {
      std::vector<PetscInt>& dof_indices = elem_dof_indices[proc_id][k];
      for (unsigned int i = 0; i < dof_indices.size(); ++i)
        for (unsigned int j = 0; j < dof_indices.size(); ++j)
          sparsity[dof_indices[i]].insert(dof_indices[j]);
    }

  // Determine the local nonzeros on this processor
  const unsigned int n_local_dofs = last_local_index - first_local_index;
  std::vector<PetscInt> n_nz(n_local_dofs);
  std::vector<PetscInt> n_oz(n_local_dofs);

  for (unsigned int i = 0; i < n_local_dofs; ++i) {
    for (std::set<unsigned int>::iterator iter = sparsity[i+first_local_index].begin(); iter != sparsity[i+first_local_index].end(); iter++) {
      unsigned int dof = *iter;
      if ((dof >= first_local_index) && (dof < last_local_index)) n_nz[i]++;
      else n_oz[i]++;
    }
  }

  // Debugging: print number of on/off diagonal nonzeros
  // for (unsigned int i=0; i<n_nz.size(); ++i)
  //   std::cout << n_nz[i] << " ";
  // std::cout << std::endl;

  // for (unsigned int i=0; i<n_oz.size(); ++i)
  //   std::cout << n_oz[i] << " ";
  // std::cout << std::endl;

  // Compute and print max number of on- and off-diagonal nonzeros.
  PetscInt n_nz_max = *std::max_element(n_nz.begin(), n_nz.end());
  PetscInt n_oz_max = *std::max_element(n_oz.begin(), n_oz.end());

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Max on-diagonal non-zeros: = %d\n", n_nz_max);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Max off-diagonal non-zeros: = %d\n", n_oz_max);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  // Initialize the matrix similar to what we do in the PetscMatrix
  // ctor and init() routines.
  Mat mat;
  ierr = MatCreate(PETSC_COMM_WORLD, &mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat, n_local_dofs, n_local_dofs, n_dofs, n_dofs);CHKERRQ(ierr);
  ierr = MatSetBlockSize(mat, 1);CHKERRQ(ierr);
  ierr = MatSetType(mat, MATAIJ);CHKERRQ(ierr); // Automatically chooses seqaij or mpiaij
  ierr = MatSeqAIJSetPreallocation(mat, 0, &n_nz[0]);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat, 0, &n_nz[0], 0, &n_oz[0]);CHKERRQ(ierr);
  ierr = MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);

  // Local "element" loop
  for (unsigned int k = 0; k < elem_dof_indices[rank].size(); k++) {
    std::vector<PetscInt>& dof_indices = elem_dof_indices[rank][k];
    // DenseMatrix< Number >  zero_mat( dof_indices.size(), dof_indices.size());
    // B.add_matrix( zero_mat, dof_indices);
    std::vector<PetscScalar> ones(dof_indices.size() * dof_indices.size(), 1.);
    ierr = MatSetValues(mat, dof_indices.size(), &dof_indices[0], dof_indices.size(), &dof_indices[0], &ones[0], ADD_VALUES);CHKERRQ(ierr);
  }

  // Matrix assembly
  ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // Clean up
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
/*TEST
   build:
      requires: !defined(PETSC_HAVE_SUN_CXX)

   test:
      nsize: 2
      args: -f0 ${DATAFILESPATH}/meshes/moose_dof_indices_np_2_proc_0.txt -f1 ${DATAFILESPATH}/meshes/moose_dof_indices_np_2_proc_1.txt
      # Skip the test for Sun C++ compiler because of its warnings/errors in petscmat.h
      requires: datafilespath

TEST*/

static char help[] = "Test using nested field splits with DMStag()\n\n";

#include <petscdm.h>
#include <petscdmstag.h>
#include <petscksp.h>

static PetscErrorCode AssembleSystem(DM dm, Mat A, Vec b)
{
    PetscInt      start[3], n[3], n_extra[3];
    DMStagStencil row[11];
    PetscScalar   val[11];

    PetscFunctionBeginUser;
    PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&n_extra[0],&n_extra[1],&n_extra[2]));

    // Corner diagonal entries 10-14
    for (PetscInt c=0; c<4; ++c) {
      row[c].loc = DMSTAG_BACK_DOWN_LEFT;
      row[c].c = c;
      val[c] = 10.0 + c;
    }

    // Element entries 20
    row[4].loc = DMSTAG_ELEMENT;
    row[4].c = 0;
    val[4] = 20.0;

    // Face entries 30-32
    row[5].loc = DMSTAG_BACK;
    row[5].c = 0;
    val[5] = 30.0;

    row[6].loc = DMSTAG_LEFT;
    row[6].c = 0;
    val[6] = 32.0;

    row[7].loc = DMSTAG_DOWN;
    row[7].c = 0;
    val[7] = 31.0;

    // Edge entries 40-42
    row[8].loc = DMSTAG_BACK_DOWN;
    row[8].c = 0;
    val[8] = 40.0;

    row[9].loc = DMSTAG_BACK_LEFT;
    row[9].c = 0;
    val[9] = 41.0;

    row[10].loc = DMSTAG_DOWN_LEFT;
    row[10].c = 0;
    val[10] = 42.0;

    for (PetscInt k=start[2]; k<start[2]+n[2]+n_extra[2]; ++k){
      for (PetscInt j=start[1]; j<start[1]+n[1]+n_extra[1]; ++j){
        for (PetscInt i=start[0]; i<start[0]+n[0]+n_extra[0]; ++i){
          for (PetscInt e=0; e<11; ++e) {
            row[e].i = i;
            row[e].j = j;
            row[e].k = k;
            PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row[e],1,&row[e],&val[e],INSERT_VALUES));
          }
        }
      }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatGetDiagonal(A,b)); // Get the diagonal, so x should be a constant 1.0
    PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  DM       dm;
  KSP      ksp;
  PC       pc;
  Mat      A;
  Vec      b, x;

  PetscInt dof[4] = {4, 1, 1, 1};
  PetscInt N[3] = {2, 3, 2};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* Create DM */
  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,N[0],N[1],N[2],PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof[0],dof[1],dof[2],dof[3],DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],&N[2]));
  PetscCall(DMSetUp(dm));

  /* Create System */
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));
  PetscCall(DMCreateMatrix(dm,&A));
  PetscCall(DMCreateGlobalVector(dm,&b));
  PetscCall(AssembleSystem(dm,A,b));
  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecSet(x,0.0));

  /* Create Linear Solver */
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm),&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));

  /* Set Up Preconditioner */
  {
    IS            is[2];
    DMStagStencil stencil_not_element[10], stencil_element[1];

    const char *name[2] = {"not_element", "element"};

    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCFIELDSPLIT));

    // First split is everything except elements (intentionally not provided in canonical order)
    for (PetscInt c=0; c<4; ++c) {
      stencil_not_element[c].loc = DMSTAG_BACK_DOWN_LEFT;
      stencil_not_element[c].c = c;
    }
    stencil_not_element[4].loc = DMSTAG_LEFT;
    stencil_not_element[4].c = 0;
    stencil_not_element[5].loc = DMSTAG_BACK;
    stencil_not_element[5].c = 0;
    stencil_not_element[6].loc = DMSTAG_DOWN;
    stencil_not_element[6].c = 0;
    stencil_not_element[7].loc = DMSTAG_BACK_DOWN;
    stencil_not_element[7].c = 0;
    stencil_not_element[8].loc = DMSTAG_BACK_LEFT;
    stencil_not_element[8].c = 0;
    stencil_not_element[9].loc = DMSTAG_DOWN_LEFT;
    stencil_not_element[9].c = 0;

    // Second split is elements
    stencil_element[0].loc = DMSTAG_ELEMENT;
    stencil_element[0].c = 0;

    PetscCall(DMStagCreateISFromStencils(dm,10,stencil_not_element,&is[0]));
    PetscCall(DMStagCreateISFromStencils(dm,1,stencil_element,&is[1]));

    for (PetscInt i=0; i<2; ++i) {
      PetscCall(PCFieldSplitSetIS(pc,name[i],is[i]));
    }

    for (PetscInt i=0; i<2; ++i) {
      PetscCall(ISDestroy(&is[i]));
    }
  }

  /* Logic below modifies the PC directly, so this is the last chance to change the solver
     from the command line */
  PetscCall(KSPSetFromOptions(ksp));

  /* If the fieldsplit PC wasn't overridden, further split */
  {
    PCType    pc_type;
    PetscBool is_fieldsplit;

    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCGetType(pc,&pc_type));
    PetscCall(PetscStrcmp(pc_type,PCFIELDSPLIT,&is_fieldsplit));
    if (is_fieldsplit) {
      PC pc_not_element,pc_not_vertex_first_three,pc_face_and_edge;

      {
        DM            dm_not_element;
        IS            is[2];
        KSP           *sub_ksp;
        PetscInt      n_splits;
        DMStagStencil stencil_vertex_first_three[3], stencil_not_vertex_first_three[7];
        const char    *name[2] = {"vertex_first_three", "not_vertex_first_three"};

        PetscCall(PCSetUp(pc)); // Set up the Fieldsplit PC
        PetscCall(PCFieldSplitGetSubKSP(pc,&n_splits,&sub_ksp));
        PetscAssert(n_splits == 2,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Expected a Fieldsplit PC with two fields");
        PetscCall(KSPGetPC(sub_ksp[0],&pc_not_element)); // Select first sub-KSP
        PetscCall(PCSetType(pc_not_element,PCFIELDSPLIT));
        PetscCall(PetscFree(sub_ksp));

        // A compatible DM for the second top-level split
        PetscCall(DMStagCreateCompatibleDMStag(dm,4,1,1,0,&dm_not_element));

        // First split within not_element is vertex_first_three
        for (PetscInt c=0; c<3; ++c) {
          stencil_vertex_first_three[c].loc = DMSTAG_BACK_DOWN_LEFT;
          stencil_vertex_first_three[c].c = c;
        }

        // Second split within not_element is everything else
        stencil_not_vertex_first_three[0].loc = DMSTAG_BACK_DOWN_LEFT;
        stencil_not_vertex_first_three[0].c = 3;
        stencil_not_vertex_first_three[1].loc = DMSTAG_LEFT;
        stencil_not_vertex_first_three[1].c = 0;
        stencil_not_vertex_first_three[2].loc = DMSTAG_BACK;
        stencil_not_vertex_first_three[2].c = 0;
        stencil_not_vertex_first_three[3].loc = DMSTAG_DOWN;
        stencil_not_vertex_first_three[3].c = 0;
        stencil_not_vertex_first_three[4].loc = DMSTAG_BACK_DOWN;
        stencil_not_vertex_first_three[4].c = 0;
        stencil_not_vertex_first_three[5].loc = DMSTAG_BACK_LEFT;
        stencil_not_vertex_first_three[5].c = 0;
        stencil_not_vertex_first_three[6].loc = DMSTAG_DOWN_LEFT;
        stencil_not_vertex_first_three[6].c = 0;

        PetscCall(DMStagCreateISFromStencils(dm_not_element,3,stencil_vertex_first_three,&is[0]));
        PetscCall(DMStagCreateISFromStencils(dm_not_element,7,stencil_not_vertex_first_three,&is[1]));

        for (PetscInt i=0; i<2; ++i) {
          PetscCall(PCFieldSplitSetIS(pc_not_element,name[i],is[i]));
        }

        for (PetscInt i=0; i<2; ++i) {
          PetscCall(ISDestroy(&is[i]));
        }
        PetscCall(DMDestroy(&dm_not_element));
      }

      // Further split the second split of the first split
      {
        DM            dm_not_vertex_first_three;
        PetscInt      n_splits;
        IS            is[2];
        KSP           *sub_ksp;
        DMStagStencil stencil_vertex_fourth[1],stencil_face_and_edge[6];
        const char    *name[2] = {"vertex_fourth", "face_and_edge"};

        PetscCall(PCSetUp(pc_not_element)); // Set up the Fieldsplit PC
        PetscCall(PCFieldSplitGetSubKSP(pc_not_element,&n_splits,&sub_ksp));
        PetscAssert(n_splits == 2,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Expected a Fieldsplit PC with two fields");
        PetscCall(KSPGetPC(sub_ksp[1],&pc_not_vertex_first_three)); // Select second sub-KSP
        PetscCall(PCSetType(pc_not_vertex_first_three,PCFIELDSPLIT));
        PetscCall(PetscFree(sub_ksp));

        PetscCall(DMStagCreateCompatibleDMStag(dm,1,1,1,0,&dm_not_vertex_first_three));

        // First split is 4th vertex entry
        stencil_vertex_fourth[0].loc = DMSTAG_BACK_DOWN_LEFT;
        stencil_vertex_fourth[0].c = 3;

        // Second split is faces and edges
        stencil_face_and_edge[0].loc = DMSTAG_LEFT;
        stencil_face_and_edge[0].c = 0;
        stencil_face_and_edge[1].loc = DMSTAG_BACK;
        stencil_face_and_edge[1].c = 0;
        stencil_face_and_edge[2].loc = DMSTAG_DOWN;
        stencil_face_and_edge[2].c = 0;
        stencil_face_and_edge[3].loc = DMSTAG_BACK_DOWN;
        stencil_face_and_edge[3].c = 0;
        stencil_face_and_edge[4].loc = DMSTAG_BACK_LEFT;
        stencil_face_and_edge[4].c = 0;
        stencil_face_and_edge[5].loc = DMSTAG_DOWN_LEFT;
        stencil_face_and_edge[5].c = 0;

        PetscCall(DMStagCreateISFromStencils(dm_not_vertex_first_three,1,stencil_vertex_fourth,&is[0]));
        PetscCall(DMStagCreateISFromStencils(dm_not_vertex_first_three,6,stencil_face_and_edge,&is[1]));

        for (PetscInt i=0; i<2; ++i) {
          PetscCall(PCFieldSplitSetIS(pc_not_vertex_first_three,name[i],is[i]));
        }

        for (PetscInt i=0; i<2; ++i) {
          PetscCall(ISDestroy(&is[i]));
        }
        PetscCall(DMDestroy(&dm_not_vertex_first_three));
      }

      // Further split the second split of the second split of the first split
      {
        DM            dm_face_and_edge;
        PetscInt      n_splits;
        IS            is[2];
        KSP           *sub_ksp;
        DMStagStencil stencil_face[3],stencil_edge[3];
        const char    *name[2] = {"face", "edge"};

        PetscCall(PCSetUp(pc_not_vertex_first_three)); // Set up the Fieldsplit PC
        PetscCall(PCFieldSplitGetSubKSP(pc_not_vertex_first_three,&n_splits,&sub_ksp));
        PetscAssert(n_splits == 2,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Expected a Fieldsplit PC with two fields");
        PetscCall(KSPGetPC(sub_ksp[1],&pc_face_and_edge)); // Select second sub-KSP
        PetscCall(PCSetType(pc_face_and_edge,PCFIELDSPLIT));
        PetscCall(PetscFree(sub_ksp));

        PetscCall(DMStagCreateCompatibleDMStag(dm,0,1,1,0,&dm_face_and_edge));

        // First split is faces
        stencil_face[0].loc = DMSTAG_LEFT;
        stencil_face[0].c = 0;
        stencil_face[1].loc = DMSTAG_BACK;
        stencil_face[1].c = 0;
        stencil_face[2].loc = DMSTAG_DOWN;
        stencil_face[2].c = 0;

        // Second split is edges
        stencil_edge[0].loc = DMSTAG_BACK_DOWN;
        stencil_edge[0].c = 0;
        stencil_edge[1].loc = DMSTAG_BACK_LEFT;
        stencil_edge[1].c = 0;
        stencil_edge[2].loc = DMSTAG_DOWN_LEFT;
        stencil_edge[2].c = 0;

        PetscCall(DMStagCreateISFromStencils(dm_face_and_edge,3,stencil_face,&is[0]));
        PetscCall(DMStagCreateISFromStencils(dm_face_and_edge,3,stencil_edge,&is[1]));

        for (PetscInt i=0; i<2; ++i) {
          PetscCall(PCFieldSplitSetIS(pc_face_and_edge,name[i],is[i]));
        }

        for (PetscInt i=0; i<2; ++i) {
          PetscCall(ISDestroy(&is[i]));
        }
        PetscCall(DMDestroy(&dm_face_and_edge));
      }
    }
  }

  /* Solve */
  PetscCall(KSPSolve(ksp,b,x));

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Clean Up */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 8
      args: -fieldsplit_element_ksp_max_it 1 -fieldsplit_element_ksp_type richardson -fieldsplit_element_pc_type none -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_fieldsplit_edge_ksp_max_it 1 -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_fieldsplit_edge_ksp_type richardson -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_fieldsplit_edge_pc_type none -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_fieldsplit_face_ksp_max_it 1 -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_fieldsplit_face_ksp_type richardson -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_fieldsplit_face_pc_type none -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_ksp_max_it 1 -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_ksp_type richardson -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_pc_fieldsplit_type additive -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_face_and_edge_pc_type fieldsplit -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_vertex_fourth_ksp_max_it 1 -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_vertex_fourth_ksp_type richardson -fieldsplit_not_element_fieldsplit_not_vertex_first_three_fieldsplit_vertex_fourth_pc_type none -fieldsplit_not_element_fieldsplit_not_vertex_first_three_ksp_max_it 1 -fieldsplit_not_element_fieldsplit_not_vertex_first_three_ksp_type richardson -fieldsplit_not_element_fieldsplit_not_vertex_first_three_pc_fieldsplit_type additive -fieldsplit_not_element_fieldsplit_not_vertex_first_three_pc_type fieldsplit -fieldsplit_not_element_fieldsplit_vertex_first_three_ksp_max_it 1 -fieldsplit_not_element_fieldsplit_vertex_first_three_ksp_type richardson -fieldsplit_not_element_fieldsplit_vertex_first_three_pc_type none -fieldsplit_not_element_ksp_max_it 1 -fieldsplit_not_element_ksp_type richardson -fieldsplit_not_element_pc_fieldsplit_type additive -fieldsplit_not_element_pc_type fieldsplit -ksp_converged_reason -ksp_type preonly

TEST*/

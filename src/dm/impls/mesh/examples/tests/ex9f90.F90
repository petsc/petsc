Program test2f90
#include "finclude/petscdef.h"
   Use petsc
   Implicit NONE

   Type(DM)                               :: dmBody,dmFS
   PetscBool                              :: inflag
   Character(len=256)                     :: infilename,IOBuffer
   PetscErrorCode                         :: ierr
   PetscInt                               :: dof=1
   PetscInt                               :: my_num_cells,my_num_faces,cell,face
   Integer                                :: rank
   Type(SectionReal)                      :: sec1
   Type(Mat)                              :: mat1
   PetscReal, Dimension(:), Pointer       :: m_array
   PetscInt                               :: conesize
   PetscInt                               :: i,j,k,c,d
   PetscReal                              :: zero = 0


   Call PetscInitialize(PETSC_NULL_CHARACTER,ierr); CHKERRQ(ierr)
   Call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)

   Call PetscOptionsGetString(PETSC_NULL_CHARACTER, '-i',infilename,inflag,ierr)
   CHKERRQ(ierr)
   If (.NOT. inflag) Then
      Call PetscPrintf(PETSC_COMM_WORLD,"No file name given\n",iErr);CHKERRQ(ierr)
      Call PetscFinalize(iErr)
      STOP
   End If

   !!!
   !!!   Reads a mesh
   !!!
   Call DMMeshCreateExodusNG(PETSC_COMM_WORLD,infilename,dmBody,dmFS,ierr)
   CHKERRQ(ierr)
   Call DMMeshGetStratumSize(dmBody,"height",0,my_num_cells,ierr);CHKERRQ(ierr)
   Call DMMeshGetStratumSize(dmFS,"height",0,my_num_faces,ierr);CHKERRQ(ierr)

   Call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-dof',dof,inflag,ierr);CHKERRQ(ierr)
   Call DMMeshSetMaxDof(dmBody,dof,ierr);CHKERRQ(ierr)
   Call DMMeshSetMaxDof(dmFS,dof,ierr);CHKERRQ(ierr)

   !!!
   !!! Testing Vertex Sections
   !!!
   Call DMMeshGetVertexSectionReal(dmBody,"sec1",dof,sec1,ierr);CHKERRQ(ierr)
   Call DMMeshCreateMatrix(dmBody,sec1,MATMPIAIJ,mat1,ierr);CHKERRQ(ierr)

   Do cell = 0,my_num_cells-1
      Call DMMeshGetConeSize(dmBody,cell,conesize,ierr);CHKERRQ(ierr)
      Allocate(m_array(conesize**2*dof**2))
      k = 1
      Do i = 1,conesize
         Do c = 1,dof
            Do j = 1,conesize
               Do d = 1,dof
                  m_array(k) = 1.
                  k = k+1
               End Do
            End Do
         End Do
      End Do
      Call DMMeshAssembleMatrix(mat1,dmBody,sec1,cell,m_array,ADD_VALUES,ierr)
      CHKERRQ(ierr)
      DeAllocate(m_array)
   End Do
   Call MatAssemblyBegin(mat1,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
   Call MatAssemblyEnd(mat1,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
   Call PetscPrintf(PETSC_COMM_WORLD,"mat1 on dmBody\n",ierr);CHKERRQ(ierr)
   Call MatView(mat1,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRQ(ierr)

   Call MatZeroEntries(mat1,ierr);CHKERRQ(ierr)
   Do face = 0,my_num_faces-1
      Call DMMeshGetConeSize(dmFS,face,conesize,ierr);CHKERRQ(ierr)
      Allocate(m_array(conesize**2*dof**2))
      k = 1
      Do i = 1,conesize
         Do c = 1,dof
            m_array(k) = 1+100*(c-1)
            k = k+1
         End Do
      End Do
      Call DMMeshAssembleMatrix(mat1,dmBody,sec1,cell,m_array,ADD_VALUES,ierr)
      CHKERRQ(ierr)
      DeAllocate(m_array)
   End Do
   Call MatAssemblyBegin(mat1,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
   Call MatAssemblyEnd(mat1,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
   Call PetscPrintf(PETSC_COMM_WORLD,"mat1 on dmBody\n",ierr);CHKERRQ(ierr)
   Call MatView(mat1,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRQ(ierr)

   Call SectionRealDestroy(sec1);CHKERRQ(ierr)
   Call MatDestroy(mat1);CHKERRQ(ierr)
   Call DMDestroy(dmBody,ierr);CHKERRQ(ierr)
   Call DMDestroy(dmFS,ierr);CHKERRQ(ierr)
   Call PetscFinalize(iErr)
End Program test2f90
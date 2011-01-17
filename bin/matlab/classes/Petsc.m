classdef Petsc < handle
%
%     PETSc MATLAB Interface Help
%
%   PetscInitialize({'-option1','value1','-option2'});
%   PetscFinalize; 
%
%   Petsc.INSERT_VALUES, Petsc.ADD_VALUES  --- Options for setting values into Vecs and Mats
%   Petsc.DECIDE,Petsc.DEFAULT,Petsc.DETERMINE --- Use instead of some integer arguments
%
%   PetscOptionsView;  % show current options
%   PetscOptionsSetValue('-optionname','value');
%   PetscOptionsClearValue('-optionname');         % remove from the database
%
%   PetscObjectsView; % show all current PETSc objects, like MATLAB who
%   A = PetscObjectsGetObject('name'); % return MATLAB pointer to PETSc object of given name
%
%   PetscCHKERRQ(ierr); % check if an error code is non-zero and set MATLAB error
%   PETSC_COMM_SELF;    % returns current MPI_COMM_SELF communicator, not needed by users
%
%   Each of the following have their own help
%      PetscObject - Abstract base class of all PETSc classes
%      PetscIS     - Index set class
%      PetscVec    - Vector class
%      PetscMat    - Matrix class
%      PetscPC     - Preconditioner class, not ususally used directly
%      PetscKSP    - Linear solver class
%      PetscSNES   - Nonlinear solver class
%      PetscTS     - ODE integrator class
%      PetscDM     - Manages interface between mesh data and the solvers
%
%  The script demo.m in bin/matlab/classes/examples/tutorials demonstrates several example 
%     usages of PETSc from MATLAB

%  Notes: You can call PetscInitialize() multiple times so long as you call PetscFinalize() between each call
%         The interface currently works only for sequential (one processor) runs, for 
%            a good hacker it should be relatively easy to make it parallel.
%         All PETSc MATLAB functions that end with Internal.m are used by PETSc and should not be called 
%            directly by users.
%
  properties (Constant)
    INSERT_VALUES=1;
    ADD_VALUES=2;
    DECIDE=-1;
    DETERMINE=-1;
    DEFAULT=-2;
    
    COPY_VALUES=0;
  
    FILE_MODE_READ=0;
    FILE_MODE_WRITE=1;
    FILE_MODE_APPEND=2;
    FILE_MODE_UPDATE=3;
    FILE_MODE_APPEND_UPDATE=4;
  end
end

 

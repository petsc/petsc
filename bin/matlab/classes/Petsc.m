%
%     PETSc MATLAB Interface Help
%
%   PetscInitialize({'-option1','value1','-option2'});
%   PetscFinalize; 
%
%   PetscOptionsView;  % show current options
%   PetscOptionsSetValue('-optionname','value');
%   PetscOptionsClearValue('-optionname');         % remove from the database
%
%   PetscCHKERRQ(ierr); % check if an error code is non-zero and set Matlab error
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
%  Notes: You can call PetscInitialize() multiple times so long as you call PetscFinalize() between each call
%         The interface currently works only for sequential (one processor) runs, for 
%         a good hacker it should be relatively easy to make it parallel.

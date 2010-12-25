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
%   PetscObjectsView; % show all current PETSc objects, like Matlab who
%   A = PetscObjectsGetObject('name'); % return Matlab pointer to PETSc object of given name
%
%   If v is a PetscVec then a = v(:) returns a Matlab array of the vector
%       and v(:) = a; assigns the array values in a into the vector. 
%       v(1:3) = [2.0 2. 3.0]; also work
%
%   If A is a PetscMat then a = A(:,:) returns the Matlab version of the sparse matrix
%       and A(:,:) = a; assigns the sparse matrix values into the PETScMat
%       you CANNOT yet use syntax like A(1,2) = 1.0
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
%            a good hacker it should be relatively easy to make it parallel.
%         All PETSc Matlab functions that end with Internal.m are used by PETSc and should not be called 
%            directly by users.

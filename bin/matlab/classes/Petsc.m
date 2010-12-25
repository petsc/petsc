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
%   PetscObjectsView; % show all current PETSc objects, like MATLAB who
%   A = PetscObjectsGetObject('name'); % return MATLAB pointer to PETSc object of given name
%
%   If v is a PetscVec then a = v(:) returns a MATLAB array of the vector
%       and v(:) = a; assigns the array values in a into the vector. 
%       v(1:3) = [2.0 2. 3.0]; also work
%
%   If A is a PetscMat then a = A(:,:) returns the MATLAB version of the sparse matrix
%       and A(:,:) = a; assigns the sparse matrix values into the PETScMat
%       you CANNOT yet use syntax like A(1,2) = 1.0
%
%   Indexing into PETSc Vecs and Mats from Matlab starts with index of 1, NOT 0 like 
%     everywhere else in PETSc, but Shri felt MATLAB users could not handle 0.
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

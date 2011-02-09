function err = PetscInitialize(args,argfile,arghelp)
%
%  PETSc must be configured with --with-shared-libraries --with-matlab-engine --with-matlab [--download-c-blas-lapack]
%
%  The option --download-c-blas-lapack must be used if using 64bit MATLAB on LINUX or 64bit MATLAB and the --download-ml external package on Apple Mac OS X.
%
%  You can build with or without MPI, but cannot run on more than one process
%
%  There is currently no MPI in the API, the MPI_Comm is not in any of the 
%  argument lists but otherwise the argument lists try to mimic the C binding
%
%  Add ${PETSC_DIR}/bin/matlab/classes to your MATLAB path
%
%  In MATLAB use help Petsc to get started using PETSc from MATLAB
%
%

if ~libisloaded('libpetsc')
  PETSC_DIR = getenv('PETSC_DIR');
  PETSC_ARCH = getenv('PETSC_ARCH');
  if (length(PETSC_DIR) == 0) 
    disp('Must have environmental variable PETSC_DIR set')
  end
  if (length(PETSC_ARCH) == 0) 
    disp('Must have environmental variable PETSC_ARCH set')
  end
  loadlibrary([PETSC_DIR '/' PETSC_ARCH '/lib/' 'libpetsc'], [PETSC_DIR '/bin/matlab/classes/matlabheader.h']);
end

if (nargin == 0)
  args = '';
end
if (nargin < 2) 
  argfile = '';
end
if (nargin < 3) 
  arghelp = '';
end
if (ischar(args)) 
  args = {args};
end

% append any options in the options variable
global options
if (length(options) > 0)
  args = cellcat(args,options)
  disp('Using additional options')
  disp(options)
end

% first argument should be program name, use matlab for this
arg = cellcat('matlab',args);
%
% If the user forgot to PetscFinalize() we do it for them, before restarting PETSc
%
init = calllib('libpetsc', 'PetscInitializedMatlab');
if (init) 
  err = calllib('libpetsc', 'PetscFinalize');PetscCHKERRQ(err);
end
err = calllib('libpetsc', 'PetscInitializeMatlab', length(arg), arg,argfile,arghelp);PetscCHKERRQ(err);



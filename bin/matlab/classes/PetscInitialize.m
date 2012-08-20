function err = PetscInitialize(args,argfile,arghelp)
%
%  PETSc must be configured with --with-shared-libraries --with-matlab-engine --with-matlab 
%
%   Note some 64 bit MATLAB versions will crash on BLAS/LAPACK calls. Some ways of handling this are to 
%       1) use   -download-f2cblaslapack  or 
%       2) use --known-64-bit-blas-indices --with-blas-lapack-dir=/Applications/MATLAB_R2011b.app/
%          the path above should point to the directory above the bin directory where the MATLAB command is
%
%  You can build with or without MPI, but cannot run on more than one process
%
%  There is currently no MPI in the API, the MPI_Comm is not in any of the 
%  argument lists but otherwise the argument lists try to mimic the C binding
%
%  Add ${PETSC_DIR}/bin/matlab/classes to your MATLAB path
%
%  In MATLAB use help PETSc to get started using PETSc from MATLAB
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
  args = [args,options];
  disp('Using additional options')
  disp(options)
end

% first argument should be program name, use matlab for this
arg = ['matlab',args];
%
% If the user forgot to PetscFinalize() we do it for them, before restarting PETSc
%
init = 0;
err = calllib('libpetsc', 'PetscInitialized',init);
if (init) 
  err = calllib('libpetsc', 'PetscFinalize');PetscCHKERRQ(err);
end
err = calllib('libpetsc', 'PetscInitializeNoPointers', length(arg), arg,argfile,arghelp);PetscCHKERRQ(err);



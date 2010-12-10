function err = PetscInitialize(args,argfile,arghelp)
%
%  PETSc must be configured with --with-shared-libraries --with-mpi=0 --with-matlab-engine --with-matlab
%
%  You currently must run matlab -nodesktop to get any output from PETSc
%
%  There is currently no MPI in the API, the MPI_Comm is not in any of the 
%  argument lists but otherwise the argument lists try to mimic the C binding
%
%
if libisloaded('libpetsc')
  unloadlibrary('libpetsc');
end
PETSC_DIR = getenv('PETSC_DIR');
PETSC_ARCH = getenv('PETSC_ARCH');
if (length(PETSC_DIR) == 0) 
  disp('Must have environmental variable PETSC_DIR set')
end
if (length(PETSC_ARCH) == 0) 
  disp('Must have environmental variable PETSC_ARCH set')
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
if (ischar(options))
  options = {options};
end
if (iscell(options))
  l = length(args);
  for i=1:length(options)
    args{i+l} = options{i};
  end
  disp('Using additional options')
  disp(options)
end

% first argument should be program name, use matlab for this
arg = cell(1,length(args)+1);
arg{1} = 'matlab';
for i=1:length(args)
  arg{i+1} = args{i};
end
loadlibrary([PETSC_DIR '/' PETSC_ARCH '/lib/' 'libpetsc'], [PETSC_DIR '/bin/matlab/classes/matlabheader.h']);
err = calllib('libpetsc', 'PetscInitializeNonPointers', length(arg), arg,argfile,arghelp);PetscCHKERRQ(err);



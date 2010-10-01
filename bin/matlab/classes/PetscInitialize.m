function err = PetscInitialize(args)
%
%  args is currently ignored
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
loadlibrary([PETSC_DIR '/' PETSC_ARCH '/lib/' 'libpetsc'], [PETSC_DIR '/bin/matlab/classes/matlabheader.h']);
err = calllib('libpetsc', 'PetscInitialize', 0, 0,'','');


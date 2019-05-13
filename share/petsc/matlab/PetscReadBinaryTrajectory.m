function [varargout] = PetscBinaryReadTrajectory(inarg)
%
%   [varargout] = PetscBinaryReadTrajectory(inarg)
%
%  Read in the trajectory information saved in a folder of PETSc binary file
%  Emit as Matlab struct
%
%  Examples: A = PetscBinaryReadTrajectory('myfolder'); read from myfolder.
%            A = PetscBinaryReadTrajectory(); read from folder 'TS-data' or 'Visualization-data' if they exist, TS-data has the priority.
%

if nargin < 1
  if exist('TS-data','dir')
    inarg = 'TS-data';
  elseif exist('Visualization-data','dir')
    inarg = 'Visualization-data';
  else
    error('Can not find the folder of trajectory files!');
  end
end

indices = 'int32';
precision = 'float64';
maxsteps = 10000;

t = zeros(1,maxsteps);

foundnames = 0;
fullname = fullfile(inarg,'variablenames');
if exist(fullname,'file') == 2
  fd = PetscOpenFile(fullname);
  n     = read(fd,1,indices);
  sizes = read(fd,n,indices);
  names = {'  '};
  for i=1:n,
    names{i} = deblank(char(read(fd,sizes(i),'uchar')))';
  end
  foundnames = 1;
end

for stepnum=1:maxsteps
  filename = sprintf('TS-%06d.bin',stepnum-1);
  fullname = fullfile(inarg,filename);
  if exist(fullname,'file') ~= 2
    steps = stepnum-1;
    break;
  end
  fd = PetscOpenFile(fullname);
  header = double(read(fd,1,indices));

  if isempty(header)
    steps = stepnum-1;
    break;
  end

  if  header == 1211214 % Petsc Vec Object
    %% Read state vector
    m = double(read(fd,1,indices));
    if (stepnum == 1)
      x = zeros(m,maxsteps);
    end
    v = read(fd,m,precision);
    x(:,stepnum) = v;

    %% Read time
    t(stepnum) = read(fd,1,precision);
  end
  % close the reader if we opened it
  close(fd);
end

if steps > 1
  varargout{1} = t(1:steps);
  varargout{2} = x(:,1:steps);
  if foundnames == 1
    varargout{3} = names;
  end
end





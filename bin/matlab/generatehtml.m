%
%   Generates html versions of all MATLAB files
%
opts.outputDir = '.';
opts.evalCode  = false;

d = dir('*.m');
l = size(d);
l = l(1);
for i=1:l
  n = d(i).name;
  publish(n,opts);
  n = n(1:(length(n)-2));
  nhtml = [n '.html'];
  nmhtml = [n '.m.html'];
  movefile(nhtml,nmhtml);
end

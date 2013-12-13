% mfqnls_driver.m
% Runs mfqnls on a generic function called specified in nls_f.m
% Requires:
%      mnhnlsv0  (directory containing mnhnls)
%      nls_f.m  (for calling the fucntion)
% Jorge More, Jason Sarich, and Stefan Wild September 2008
%
% mfqnls solves problems of the form:
%    min_x  fval(x) = sum_{i=1}^n f_i(x)^2
% where
%   x    [dbl] is n-dimensional input
%   f(x) [dbl] is m-dimensional output (evaluated in nls_f.m)

global m nfev Fvals Xhist Fhist Deltahist delta  % Global vars used in nls_f
path(path,'/home/sarich/working/ptho/mfqnls/gqt'); 

% 1. Things the user must set:
n = 3;                      % [int] Number of variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxfev = 200;               % [int] Maximum Number of Evaluations
m = 214;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xs = [0.15;0.008;0.01];
%delta = max(norm(Xs,inf),1);% [dbl] Initial trust-region-radius/step-size
delta = 0.001;
% 2. Advanced options:
npmax = (n+1)*(n+2)/2;     % [int] Max # interpolation pts [(n+1)(n+2)/2]
mtype = 1;                  % [int] Flag for type of nls solver [1]
tolx = 1e-10;               % [dbl] Stopping criterion [1e-10]

% 3. Initializations:
nfev = 0;                   % [int] Counter for the number of evaluations
Fvals = zeros(maxfev,1);    % [dbl] Vector of the values of fval(x)
Xhist = zeros(maxfev,n);    % [dbl] Matrix whose rows are the x's evaluated
Fhist = zeros(maxfev,m);    % [dbl] Matrix whose rows are the f components
Deltahist = zeros(maxfev,1);

ts = cputime;               % Start a timer
diary('mfqnls_diary');      % Start the diary log

% 4. Call mnhnls (no additional output requested):
%showme_xs = Xs'
[X,F,flag]=MFQnls('nls_f',Xs',n,npmax,maxfev,tolx,delta,m,mtype);

% 5.Summary information 
% Info to screen:
tf = cputime - ts;
disp(sprintf('Number of function evaluations    %25i \n',nfev));
disp(sprintf('Total execution time              %25.2e \n',tf));

% Info to file
file_data = 'H_output_nls';         % Filename for output
save(file_data,'Fvals','Xhist','Fhist','Deltahist');

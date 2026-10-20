close all; clear all; home;
addpath('model');

m = Model.fromFile('model.model', 'linear', true);
% m = model('model.model', 'linear', true);
p = eval('param');
m = assign(m, p);

m = solve(m);
m = sstate(m);

sschk = chksstate(m);

[~,~,~] = mkdir('results');
[~,~,~] = mkdir(['results\' mfilename]);
save(['results\' mfilename '\model.mat'],'m');
disp([mfilename ' Done!']);
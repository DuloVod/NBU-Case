close all; clear all; home;
[~,~,~] = mkdir(['results\' mfilename]);

%% Read model
tmp = load('results\a02_read_model\model.mat');
m = tmp.m;
m1 = m; m2 = m; m3 = m;

%% Executive definitions

% CHANGE IN PARAMETERS
% Either play with ER regime ...
m1.alpha = 1;
m2.alpha = 1;
m3.alpha = 1;

m1.beta = 0;
m2.beta = 0.6;
m3.beta = 1;

% ... or with CB's control over the interest rate
% m1.alpha = 0;
% m2.alpha = 0.6;
% m3.alpha = 1;
% 
% m1.beta = 0.6;
% m2.beta = 0.6;
% m3.beta = 0.6;

% LISTS OF SHOCKS & RESPONSES
list_shocks = {'res_dl_cpi', 'res_l_gdp_gap', 'res_ir_tlr'};
list_sizes  = {1             , 1              , 1           };

list_responses = {'l_gdp_gap', 'd4l_cpi', 'ir_pol', 'l_er', 'l_rer_gap', 'rir_pol_gap'};

sbpl = [2,3];

% SHOCK SIZE & HORIZON
horizon = 1:20;
plot_horizon = 0:20; 

%% And we continue
m1 = solve(m1); m1 = sstate(m1);
m2 = solve(m2); m2 = sstate(m2);
m3 = solve(m3); m3 = sstate(m3); 

%% For the sake of reporting
comd = zerodb(m, horizon);

%% IRF
for i=1:numel(list_shocks)
    db1 = srf(m1, horizon, 'select=', list_shocks(i), 'size', list_sizes{i});
    db2 = srf(m2, horizon, 'select=', list_shocks(i), 'size', list_sizes{i});
    db3 = srf(m3, horizon, 'select=', list_shocks(i), 'size', list_sizes{i});
    
    fig = figure('Position', [100, 100, 900, 600]);
    
    for j=1:numel(list_responses)
        
        subplot(sbpl(1), sbpl(2), j);
        plot(plot_horizon, db1.(list_responses{j}), 'linewidth', 2);
%         plot(plot_horizon, [db1.(list_responses{j}) db2.(list_responses{j}) db3.(list_responses{j})], 'linewidth', 2);
        grid;
        title(char(comment(comd.(list_responses{j}))), 'FontSize', 13);
        
        if j == 1
%             legend({'beta = 0', 'beta = 0.6', 'beta = 1'}, 'location', 'best', 'FontSize', 12);
% % % % % % %             legend({'alpha = 0', 'alpha = 0.6', 'alpha = 1'}, 'location', 'best', 'FontSize', 12);
        end
        
    end

%     saveas(fig, ['results\' mfilename '\beta_' char(comment(comd.(list_shocks{i}))) '.png'])
    saveas(fig, ['results\' mfilename '\' char(comment(comd.(list_shocks{i}))) '.png'])
% % % % % % %     saveas(fig, ['results\' mfilename '\alpha_' char(comment(comd.(list_shocks{i}))) '.png'])
    
end

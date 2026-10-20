close all; clear all; clc;
[~,~,~] = mkdir('results');

%% Load all data
dm = databank.fromCSV('data\monthly.csv');
dq = databank.fromCSV('data\quarterly.csv');
dq_ext  = databank.fromCSV('data\foreign.csv');

%% Monthly
% Rewrite CPI m-o-m to levels
idx = ["su_m"]; % indices for CPIs
begin = 1; % value in 2000m1 to have the same level
for i = 1:length(idx)
    var_name = strcat("dm.D_CPI_", idx(i)); 
    eval(sprintf("%s = cumprod([1+%s/100]);", var_name, var_name)); % from m-o-m to level 
    eval(sprintf(strcat("%s = %s/%s(mm(2000,1))*%d;"), var_name, var_name, var_name, begin));
    dm.(strcat("CPI_", idx(i))) = dm.(strcat("D_CPI_", idx(i)));
    dm = rmfield(dm, strcat("D_CPI_", idx(i)));
end

clear idx begin i var_name; % clear tmp variables

% Seasonal adjustment
% dm = dbbatch(dm,'$1_sf_m', 'x12(dm.$0,''output'',''SF'')','nameFilter','(.*)_su_m'); % if you need seasonal factors
dm = dbbatch(dm,'$1_m', 'x12(dm.$0)','nameFilter','(.*)_su_m');

%% Quarterly
% Seasonal adjustment
% dq = dbbatch(dq,'$1_sf','x12(dq.$0,''output'',''SF'')','nameFilter','(.*)_su'); % if you need seasonal factors
dq = dbbatch(dq,'$1', 'x12(dq.$0)','nameFilter','(.*)_su');

% Convert monthly to quarterly
name = ["CPI", "IR_POLICY", "ER"];
for i = 1:length(name)
    var_name_1 = strcat("dq.", name(i));
    var_name_2 = strcat("dm.", name(i),"_m");
    eval(sprintf("%s = convert(%s,'Q',inf,'method','nanmean');", var_name_1, var_name_2));
end

clear i name var_name_1 var_name_2;

%% External
% Seasonal adjustment (not needed currently)
dq_ext = dbbatch(dq_ext,'$1', 'x12(dq_ext.$0)','nameFilter','(.*)_su');

% Overlay dq and dq_ext
dq = dbextend(dq, dq_ext);

qseries = fieldnames(dq);
dq = rmfield(dq, qseries(contains(qseries, '_su')));

%% Transformations
% Make logs
exceptions = {'IR_EA', 'IR_POLICY'};
dq = make_logs(dq, exceptions); % The function is introduced later

clear exceptions qseries;

% HP filter for external
% dq.L_GDP_EA_TND = hpf(dq.L_GDP_EA); % if you need trend
dq.L_GDP_EA_GAP = hpf2(dq.L_GDP_EA);

%% Save resulted databases
[~,~,~] = mkdir(['results\' mfilename]);
save(['results\' mfilename '\read_data.mat'], 'dq');
disp([mfilename ' Done!']);

%% Function
function s = make_logs(data, except)
    list = fieldnames(data);
    s = data;
    for i = 1:length(list)
    varname = list{i};
        if isempty(strmatch(varname, except, 'exact'))
            s.(['L_' varname])     = 100*log(s.(varname));
            s.(['DL_' varname])    = 4*diff(s.(['L_' varname]), -1); % q-o-q annualized
            s.(['D4L_' varname])   = diff(s.(['L_' varname]), -4);   % y-o-y
        end
    end
end
        
        
        

% Standard MATLAB table import. IRIS mapping depends on your own model.
qpm = readtable(fullfile('model','quarterly_training.csv'), 'TextType','string');
assert(width(qpm)==8, 'Expected period + 7 variables');
assert(all(all(~ismissing(qpm(:,2:end)))), 'Missing core observation');

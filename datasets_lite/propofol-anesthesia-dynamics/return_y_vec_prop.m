function [y_vec] = return_y_vec_prop(t,signal,metric,width,num_bins)
%This function takes the t, signal, width of time window, and metric to use (mean or median) and returns
%the vector of indices summarized for each bin using the metric

%Inputs:
%t - vector of timepoints
%signal - vector of actual index at the values in t
%metric - either 'mean' or 'median'
%width - duration of time window to use, in seconds
%num_bins - the number of total time windows to fill (remaining signal at
%the end will be unused)

%Outputs:
%y_vec - signal summarized for each time window (e.g. mean of each
%10 second window)

y_vec = ones(num_bins,1)*NaN;

%Get the label of each time point to which bin
if t(1)>2
    inds = ceil((t-t(1))/width);
else
    inds = ceil(t/width);
end

for i = 1:num_bins
    chunk = real(signal(inds == i));
    if strcmp(metric,'mean')
        y_vec(i) = nanmean(chunk);
    else 
        y_vec(i) = nanmedian(chunk);
    end
end

end
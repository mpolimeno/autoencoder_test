using DelimitedFiles
using Plots

rows_to_skip = 49
# load base_run data
case4_time_series_base_case = readdlm("PATH/TO/case4_original_input/Results/probes.dat")
case4_time_series_base_case = case4_time_series_base_case[rows_to_skip:end,1:3]
case4_time_series_base_case = Float64.(case4_time_series_base_case)

# input data
for ii=1:8
    case4_time_series_input = readdlm("PATH/TO/case4_HOS-NWT_"*string(ii)*"_inp_output/Results/probes.dat")
    case4_time_series_input = case4_time_series_input[rows_to_skip:end,1:3]
    case4_time_series_input = Float64.(case4_time_series_input)

    plot_file = plot(case4_time_series_base_case[:,1],case4_time_series_base_case[:,3],label="Original")
    plot!(case4_time_series_input[:,1],case4_time_series_input[:,3],label="Input "*string(ii))
    savefig(plot_file,"time_series_base_vs_"*string(ii)*".pdf")
end

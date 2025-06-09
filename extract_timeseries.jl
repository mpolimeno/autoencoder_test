using DelimitedFiles

function extract_timeseries(input::Matrix, output_filepath::String)
    # Extract the first column (time) and the second column (eta) from the input matrix.
    # Note: The 'input' matrix passed from the main loop already has time as column 1
    # and eta as column 2, due to the hcat(time, eta) operation.
    extracted_time = input[:, 1]
    extracted_eta = input[:, 2]

    # To save these two arrays into two columns in the output file,
    # we first horizontally concatenate them using hcat.
    # Then, we use writedlm, which is designed for writing delimited data (like matrices)
    # to a file. The third argument specifies the delimiter (here, a space ' ').
    # This will automatically create or overwrite the file as needed for each iteration.
    writedlm(output_filepath, hcat(extracted_time, extracted_eta), ' ')
end

# Define the number of rows to skip at the beginning of the input file.
rows_to_skip = 49

# Initialize empty variables to hold the time and eta data and the output filename.
# These are declared globally as they are modified within the loop.
time_and_eta = []
output_file = ""

# Loop through 8 different cases (from 1 to 8).
for ii in 1:8
    # Construct the input file path using string concatenation.
    # Assumes the files are named like "case4_HOS-NWT_1_inp_output/Results/probes.dat", etc.
    input_filepath = "PATH/TO/case4_HOS-NWT_" * string(ii) * "_inp_output/Results/probes.dat"

    # Read the data from the specified input file.
    # readdlm reads the data into a matrix.
    case4_time_series_base_case = readdlm(input_filepath)

    # Slice the matrix: skip the initial rows_to_skip rows and select columns 1 to 3.
    case4_time_series_base_case = case4_time_series_base_case[rows_to_skip:end, 1:3]

    # Convert all elements in the sliced matrix to Float64.
    case4_time_series_base_case = Float64.(case4_time_series_base_case)

    # Extract the first column (time) and the third column (eta) from the processed data.
    time = case4_time_series_base_case[:, 1]
    eta = case4_time_series_base_case[:, 3]

    # Horizontally concatenate the time and eta vectors into a new matrix.
    # This matrix will have two columns: time and eta.
    global time_and_eta = hcat(time, eta)

    # Construct the output file name for the current iteration.
    global output_file = "timeseries_" * string(ii) * ".txt"

    # Call the extract_timeseries function to process and save the data.
    # This function expects a matrix with time in the first column and eta in the second.
    extract_timeseries(time_and_eta, output_file)
end

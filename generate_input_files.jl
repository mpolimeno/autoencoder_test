# This Julia script reads an input text file, modifies specific lines,
# and generates multiple output .inp files with incrementally changed parameters.


using Random # Import the Random module for random number generation
using Statistics
using Distributions
"""
    update_parameter_value(line::String, new_value::Float64)

Parses a line from the input file, extracts the numerical value, and replaces it
with `new_value`, ensuring the 'd0' suffix is maintained for floating-point precision.

Arguments:
- `line::String`: The original line from the input file.
- `new_value::Float64`: The new numerical value to insert.

Returns:
- `String`: The modified line with the new value, or the original line if no match is found.
"""
function update_parameter_value(line::String, new_value::Float64)::String
    # Regular expression to capture the prefix (parameter name, variable name, and '::')
    # and the numerical value with its 'd0' or 'D0' suffix.
    # Group 1: Everything before the value (e.g., "Significant wave height (m)  :: Hs               :: ")
    # Group 2: The numerical part of the value (e.g., "0.05")
    # Group 3: The 'd0' or 'D0' suffix
    m = match(r"^(.*?::\s*[\w_]+\s*::\s*)([\d\.]+)([dD]0)\s*$", line)

    if m !== nothing
        prefix = m.captures[1]
        suffix = m.captures[3] # Capture the 'd0' or 'D0' part

        # Format the new value to a string and append the original 'd0' suffix.
        # Using Printf.@sprintf for consistent formatting if needed, but string() is simpler here.
        new_value_str = string(new_value) * suffix
        return prefix * new_value_str
    else
        # If the line does not match the expected parameter format, return it unchanged.
        # This prevents accidental modification of headers or other non-parameter lines.
        return line
    end
end

function main()
    input_filename = "./base_run/case4_HOS-NWT_0.inp"
    output_base_filename = "case4_HOS-NWT_"
    num_files_to_generate = 8

    # Check if the input file exists
    if !isfile(input_filename)
        println(stderr, "Error: Input file '$input_filename' not found.")
        return
    end

    # Read all lines from the input file into memory
    lines = readlines(input_filename)

    # Define initial values for the parameters to be changed
    # These correspond to the values on lines 23, 24, and 25 in the original file
    # These will be the starting points for the random increments
    current_Hs = 0.148   # Significant wave height (m)
    current_Tp = 1.6971    # Peak period (s)

    println("Generating $num_files_to_generate .inp files with random increments (can be negative)...")

    # Loop to generate each of the 10 files
    for i in 1:num_files_to_generate
        # Create a deep copy of the original lines for each new file.
        # This ensures that modifications for one file don't affect the next.
        modified_lines = deepcopy(lines)

        # Generate random increments for Hs and Tp, allowing negative values
        # Hs increment range: [-std(wave_height_exp), std(wave_height_exp)] rand(Uniform(-0.0367,0.0367),1)
        # this has turned out to sometimes produce waveheights that HOS-WT cannot handle. So I am going to half it or so
        random_Hs_vec = rand(Uniform(-0.0123,0.0123),1)
        random_increment_Hs = random_Hs_vec[1]
        # Tp increment range: [-df, df]
        df = 1.0/141.42
        random_Tp_vec = rand(Uniform(-df,df),1)  
        random_increment_Tp = random_Tp_vec[1]

        # Update Hs and Tp by adding the random increments
        current_Hs = 0.148 + random_increment_Hs
        current_Tp = 1.6971 + random_increment_Tp

        # Ensure Hs and Tp never go negative
        current_Hs = max(0.0, current_Hs)
        current_Tp = max(0.0, current_Tp)

        # Generate a random value for gamma directly within its specified range [1.0, 7.0]
        current_gamma = 1.0 + (7.0 - 1.0) * rand()


        # Update the values on the specific lines
        # Line 23: Significant wave height (m) :: Hs
        if length(modified_lines) >= 23
            modified_lines[23] = update_parameter_value(modified_lines[23], current_Hs)
        else
            println(stderr, "Warning: Line 23 not found in input file for iteration $i. Skipping modification.")
        end

        # Line 24: Peak period (s) :: Tp
        if length(modified_lines) >= 24
            modified_lines[24] = update_parameter_value(modified_lines[24], current_Tp)
        else
            println(stderr, "Warning: Line 24 not found in input file for iteration $i. Skipping modification.")
        end

        # Line 25: Shape factor (Jonswap) :: gamma
        if length(modified_lines) >= 25
            modified_lines[25] = update_parameter_value(modified_lines[25], current_gamma)
        else
            println(stderr, "Warning: Line 25 not found in input file for iteration $i. Skipping modification.")
        end

        # Construct the output filename (e.g., "case4_HOS-NWT_1.inp")
        output_filename = output_base_filename * string(i) * ".inp"

        # Write the modified content to the new .inp file
        open(output_filename, "w") do f
            for line in modified_lines
                write(f, line * "\n") # Add newline character back for each line
            end
        end

        println("Successfully generated: $output_filename (Hs=$(current_Hs)d0, Tp=$(current_Tp)d0, gamma=$(current_gamma)d0)")
    end

    println("File generation complete.")
end

# Call the main function to execute the script
main()

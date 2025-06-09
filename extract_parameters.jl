function extract_and_save_data(input_filepath::String, output_filepath::String)
    extracted_values = String[]
    
    open(input_filepath, "r") do io
        lines = readlines(io)
        
        # Extract data from lines 23, 24, and 25 (1-indexed)
        for i in 23:25
            if i <= length(lines) # Ensure line exists to prevent out-of-bounds error
                line = lines[i]
                parts = split(line, "::")
                if length(parts) == 3
                    cleaned_value = replace(strip(parts[3]), "d0" => "")
                    push!(extracted_values, cleaned_value)
                end
            end
        end
    end
    
    # Save the extracted content to the output file
    open(output_filepath, "w") do io
        for val in extracted_values
            write(io, val * "\n") # Write each value on a new line
        end
    end
    
    println("Extracted data saved to '$output_filepath'")
end

# Define the input and output file paths
input_file = ""
output_file = ""
for ii in 1:8
    global input_file = "PATH/TO/case4_HOS-NWT_"*string(ii)*"_inp_output/case4_HOS-NWT_"*string(ii)*".inp"
    global output_file = "parameters_"*string(ii)*".txt"
    # Run the extraction and saving function
    extract_and_save_data(input_file, output_file)
end
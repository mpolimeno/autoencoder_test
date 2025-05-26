using DelimitedFiles
using Dates

# --- Configuration ---
const EXECUTABLE_PATH = "PATH/TO/HOS/EXECUTABLE" # Adjust this path

# Dynamically generate input file names
const NUM_INPUT_FILES = 8 # Define how many input files you want
const INPUT_FILES = ["case4_HOS-NWT_$(ii).inp" for ii in 1:NUM_INPUT_FILES]

# Define the name of the additional file to be copied into each output directory
const ADDITIONAL_FILE_TO_COPY = "probe_2.inp" # <-- NEW CONSTANT

# Define the base directory for all outputs
const OUTPUT_BASE_DIR = "run_outputs_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

# --- Main Driver Logic ---

function run_all_processes_concurrently_with_outputs(executable::String, input_files::Vector{String}, base_output_dir::String)
    # Executable existence and permission check
    if !isfile(executable)
        @warn "Executable '$executable' not found. Please ensure the path is correct."
        return
    elseif !isexecutable(executable)
        @warn "Executable '$executable' is not executable. Please ensure it has execute permissions (e.g., `chmod +x $executable`)."
        return
    end

    # Create the main base output directory
    mkpath(base_output_dir)
    println("All process outputs will be saved under: $(abspath(base_output_dir))")
    println("Starting concurrent execution of processes...")

    processes = [] # To store the Process objects

    for input_file_name in input_files
        # Assuming original input files are in the current working directory of the Julia script
        original_input_filepath = joinpath(pwd(), input_file_name)

        # Create a unique output directory for this specific run
        input_basename = replace(basename(input_file_name), "." => "_")
        output_subdir = joinpath(base_output_dir, "$(input_basename)_output")

        mkpath(output_subdir)
        println("Prepared output directory for $(input_file_name): $(output_subdir)")

        # --- Copy the main input file into the output_subdir ---
        destination_input_filepath = joinpath(output_subdir, input_file_name)
        if isfile(original_input_filepath)
            cp(original_input_filepath, destination_input_filepath, force=true)
            println("Copied input file '$(original_input_filepath)' to '$(destination_input_filepath)'")
        else
            @error "Original input file '$(original_input_filepath)' not found. Cannot proceed with this input. Skipping."
            continue # Skip this input file and move to the next one
        end

        # --- NEW: Copy the additional "probe_2.inp" file ---
        original_probes_filepath = joinpath(pwd(), ADDITIONAL_FILE_TO_COPY)
        destination_probes_filepath = joinpath(output_subdir, ADDITIONAL_FILE_TO_COPY)

        if isfile(original_probes_filepath)
            cp(original_probes_filepath, destination_probes_filepath, force=true)
            println("Copied additional file '$(original_probes_filepath)' to '$(destination_probes_filepath)'")
        else
            @error "Additional file '$(original_probes_filepath)' not found. Skipping copy for this run. Executable might fail."
            # Decide if you want to `continue` here or let the executable potentially fail if this file is critical.
            # For now, it will proceed without the file.
        end
        # ----------------------------------------------------

        # Redirect stdout and stderr to files within the output directory
        stdout_log_path = joinpath(output_subdir, "stdout.log")
        stderr_log_path = joinpath(output_subdir, "stderr.log")

        stdout_handle = open(stdout_log_path, "w")
        stderr_handle = open(stderr_log_path, "w")

        # Construct the command. The executable will find the copied input file
        # because its working directory is `output_subdir`.
        command_with_dir = Cmd(`$executable $input_file_name`, dir=output_subdir)

        println("Launching: $command_with_dir (output to $stdout_log_path, errors to $stderr_log_path)")

        proc = run(pipeline(command_with_dir, stdout=stdout_handle, stderr=stderr_handle),
                   wait=false)

        push!(processes, (proc, stdout_handle, stderr_handle, input_file_name, output_subdir))
    end

    println("\nAll processes launched. Waiting for them to complete...\n")

    # Wait for all launched processes to finish and close log files
    for (i, (proc, stdout_handle, stderr_handle, input_file_name, output_subdir)) in enumerate(processes)
        try
            wait(proc)
            close(stdout_handle)
            close(stderr_handle)

            if success(proc)
                println("Process $(i) for input '$(input_file_name)' completed successfully. Output in $(output_subdir)")
            else
                println("Process $(i) for input '$(input_file_name)' FAILED with exit code $(proc.exitcode). Check logs in $(output_subdir)")
            end
        catch e
            @error "An error occurred while waiting for process $(i) for input '$(input_file_name)': $e"
            close(stdout_handle)
            close(stderr_handle)
        end
    end

    println("\nAll processes have finished.")
end

# --- Run the driver code ---
run_all_processes_concurrently_with_outputs(EXECUTABLE_PATH, INPUT_FILES, OUTPUT_BASE_DIR)

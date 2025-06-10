using Random
using LinearAlgebra
using Statistics
using CUDA # Changed from Metal
using Flux
using Flux: train!, DataLoader, trainable, gradient, setup
using Flux.Optimisers: update!
import ProgressMeter
using Flux.Losses: mae, mse
using MLUtils # For train/test split and batching
using DelimitedFiles

# --- Custom StandardScaler Implementation ---
mutable struct StandardScaler
    mean::Union{Vector{Float32}, Nothing}
    std::Union{Vector{Float32}, Nothing}
    StandardScaler() = new(nothing, nothing) # Constructor for initial state
end

function fit!(scaler::StandardScaler, X::AbstractMatrix{Float32})
    # Calculate mean and standard deviation along columns (features)
    scaler.mean = mean(X, dims=1)[:] # dims=1 preserves 1st dim, [:] flattens to vector
    scaler.std = std(X, dims=1)[:]
    # Handle the case where std is zero (e.g., constant feature)
    # Use a small epsilon instead of 1f0 for numerical stability
    scaler.std[scaler.std .== 0] .= eps(Float32) # Use a small epsilon to avoid division by zero
    return scaler
end

function transform!(scaler::StandardScaler, X::AbstractMatrix{Float32})
    if isnothing(scaler.mean) || isnothing(scaler.std)
        error("StandardScaler not fitted. Call fit! first.")
    end
    # Ensure broadcasting works correctly. Expand dims for mean/std if they are vectors
    # to match the shape of X for element-wise operation.
    return (X .- permutedims(scaler.mean)) ./ permutedims(scaler.std)
end

function fit_transform!(scaler::StandardScaler, X::AbstractMatrix{Float32})
    fit!(scaler, X)
    return transform!(scaler, X)
end


# --- Configuration ---
const number_of_samples = 100
const batch_size = 50
const num_epochs = 100
const learning_rate = 0.001
const hidden_layer_1_size = 512
const hidden_layer_2_size = 256
const hidden_layer_3_size = 128

# --- Data Generation ---
function process_data_padded_tensors(number_of_samples::Int, pad_value::Float32=0.0f0)
    # --- Step 1: First Pass to Find Maximum Dimensions for both X and Y ---
    max_x_rows = 0
    max_x_cols = 0
    max_y_rows = 0
    max_y_cols = 0
    
    # Store the raw data temporarily to avoid reading files twice
    raw_x_data = Vector{Matrix{Float64}}(undef, number_of_samples)
    raw_y_data = Vector{Matrix{Float64}}(undef, number_of_samples)

    println("--- First Pass: Determining Max Dimensions for X and Y ---")
    for ii in 1:number_of_samples
        # For X_generated (timeseries data)
        input_timeseries_path = "timeseries_" * string(ii) * ".txt"
        try
            time_and_eta = readdlm(input_timeseries_path, Float64)
            raw_x_data[ii] = time_and_eta
            rows_x, cols_x = size(time_and_eta)
            if rows_x > max_x_rows
                max_x_rows = rows_x
            end
            if cols_x > max_x_cols
                max_x_cols = cols_x
            end
        catch e
            error("Error reading $input_timeseries_path: $e. Please ensure all timeseries files exist and are readable.")
        end

        # For y_labels (parameters data)
        input_parameters_path = "parameters_" * string(ii) * ".txt"
        try
            parameters = readdlm(input_parameters_path, Float64)
            raw_y_data[ii] = parameters
            rows_y, cols_y = size(parameters)
            if rows_y > max_y_rows
                max_y_rows = rows_y
            end
            if cols_y > max_y_cols
                max_y_cols = cols_y
            end
        catch e
            error("Error reading $input_parameters_path: $e. Please ensure all parameters files exist and are readable.")
        end
        
        println("  Sample $ii: X_size $(size(raw_x_data[ii])), Y_size $(size(raw_y_data[ii]))")
    end
    println("--- Max dimensions found: X=($max_x_rows, $max_x_cols), Y=($max_y_rows, $max_y_cols) ---")

    # --- Step 2: Pre-allocate the final tensors with maximum dimensions ---
    X_generated = fill(pad_value, max_x_rows, max_x_cols, number_of_samples)
    y_labels = fill(pad_value, max_y_rows, max_y_cols, number_of_samples)
    
    println("\n--- Second Pass: Padding and Populating Tensors ---")
    for ii in 1:number_of_samples
        # Populate X_generated
        current_x_matrix = raw_x_data[ii]
        rows_x, cols_x = size(current_x_matrix)
        X_generated[1:rows_x, 1:cols_x, ii] = current_x_matrix
        println("  Padded Sample $ii (X original size $(size(current_x_matrix))) into slice of size $(size(X_generated[:,:,ii]))")

        # Populate y_labels
        current_y_matrix = raw_y_data[ii]
        rows_y, cols_y = size(current_y_matrix)
        y_labels[1:rows_y, 1:cols_y, ii] = current_y_matrix
        println("  Padded Sample $ii (Y original size $(size(current_y_matrix))) into slice of size $(size(y_labels[:,:,ii]))")
    end

    println("\nFinal Shape of X_generated: $(size(X_generated))")
    println("Final Shape of y_labels: $(size(y_labels))")

    return X_generated, y_labels
end

# Make sure your timeseries_*.txt and parameters_*.txt files are available in the run directory
X_final, Y_final = process_data_padded_tensors(number_of_samples, 0.0f0) # Using 0.0 as padding value

println("Shape of X: $(size(X_final))")
println("Shape of labels: $(size(Y_final))")

X_generated = X_final
y_labels = Y_final

# --- Scaling per sample using custom StandardScaler ---
X_scaled_per_sample = similar(X_generated)
for i in 1:number_of_samples
    sample_data = X_generated[:, :, i]
    scaler = StandardScaler() # Create a new scaler for each sample
    X_scaled_per_sample[:, :, i] .= fit_transform!(scaler, sample_data)
end

# Reshape for the FFNN: (features, samples)
# Flux expects (features, batch_size) for input
X_reshaped = reshape(X_scaled_per_sample, :, number_of_samples)
y_reshaped = dropdims(y_labels, dims=2) # Assuming y_labels has a singleton second dimension

# --- Train/Test Split using MLUtils.splitobs ---
(X_train, y_train), (X_test, y_test) = splitobs((X_reshaped, y_reshaped), at=0.8, shuffle=true)


# --- Move to GPU (if available) ---
# Check if CUDA is functional (NVIDIA GPUs)
const device = CUDA.functional() ? gpu : cpu
println("Using device: $device")
if device == cpu
    @warn "CUDA is not functional. Running on CPU. Performance may be significantly slower."
end

X_train_gpu = device(X_train)
y_train_gpu = device(y_train)
X_test_gpu = device(X_test)
y_test_gpu = device(y_test)

# --- DataLoader ---
train_loader = DataLoader((X_train_gpu, y_train_gpu), batchsize=batch_size, shuffle=true)
test_loader = DataLoader((X_test_gpu, y_test_gpu), batchsize=batch_size, shuffle=false)

# --- Model Definition ---
input_dim = size(X_train_gpu, 1) # First dimension is features
output_dim = size(y_train_gpu, 1) # (3) - based on your example

simple_ffnn = Chain(
    Dense(input_dim, hidden_layer_1_size, relu),
    Dense(hidden_layer_1_size, hidden_layer_2_size, leakyrelu),
    Dense(hidden_layer_2_size, hidden_layer_3_size, leakyrelu),
    Dense(hidden_layer_3_size, output_dim)
) |> device # Move model to GPU

# --- Loss Function and Optimizer ---
# Flux's `mae` is Mean Absolute Error, equivalent to PyTorch's L1Loss
criterion(y_hat, y) = mae(y_hat, y)

optimizer = Adam(learning_rate)

# --- Initialize Optimizer State ---
optimizer_state = Flux.setup(optimizer, simple_ffnn)

# --- Training Loop ---
println("\n--- Starting Training ---")
for epoch in 1:num_epochs
    epoch_loss = 0.0
    # Use fully qualified names for ProgressMeter functions
    progress_bar = ProgressMeter.Progress(length(train_loader); desc="Epoch $(epoch)/$(num_epochs)", color=:blue) 

    for (batch_X, batch_y) in train_loader
        # Calculate gradients using the explicit form
        grads = gradient(m -> criterion(m(batch_X), batch_y), simple_ffnn)
        
        # Apply gradients using the optimizer state and the model directly
        update!(optimizer_state, simple_ffnn, grads[1]) 
        
        epoch_loss += criterion(simple_ffnn(batch_X), batch_y)
        ProgressMeter.next!(progress_bar) # No `showvalues` argument needed, just advance
    end
    avg_epoch_loss = epoch_loss / length(train_loader)
    println("Epoch [$(epoch)/$(num_epochs)], Average Training Loss: $(round(avg_epoch_loss, digits=4))")
end

# --- Evaluation ---
println("\n--- Starting Evaluation ---")
Flux.testmode!(simple_ffnn) # Set model to evaluation mode

all_predictions_cpu = [] # Store on CPU
all_targets_cpu = []      # Store on CPU

progress_bar_test = ProgressMeter.Progress(length(test_loader); desc="Predicting", color=:green)
for (batch_X_test, batch_y_test) in test_loader
    predictions = simple_ffnn(batch_X_test)
    push!(all_predictions_cpu, cpu(predictions)) # Move predictions back to CPU
    push!(all_targets_cpu, cpu(batch_y_test))
    ProgressMeter.next!(progress_bar_test)
end

# Concatenate results
final_predictions = hcat(all_predictions_cpu...) # (output_dim, total_samples)
final_targets = hcat(all_targets_cpu...)          # (output_dim, total_samples)

# --- Calculate Metrics (Julia-native) ---
test_mae = mae(final_predictions, final_targets)

# MSE (Mean Squared Error)
test_mse = mse(final_predictions, final_targets)
test_rmse = sqrt(test_mse)

# R-squared (R2 Score)
# R2 = 1 - (SS_res / SS_tot)
# SS_res = sum((y_true - y_pred)^2)
# SS_tot = sum((y_true - mean(y_true))^2)
mean_targets = mean(final_targets, dims=2) # Mean for each output feature
ss_total = sum((final_targets .- mean_targets).^2)
ss_residual = sum((final_targets - final_predictions).^2)

# Handle the case where ss_total is zero (e.g., all target values are the same)
test_r2 = if ss_total == 0
    (ss_residual == 0) ? 1.0 : NaN # If residual is also zero, perfect fit. Else, undefined.
else
    1 - (ss_residual / ss_total)
end

println("\n==> Test Set Performance Metrics:")
println("      -> Mean Absolute Error (MAE): $(round(test_mae, digits=4))")
println("      -> Mean Squared Error (MSE): $(round(test_mse, digits=4))")
println("      -> Root Mean Squared Error (RMSE): $(round(test_rmse, digits=4))")
println("      -> R-squared (R2 Score): $(round(test_r2, digits=4))")

### **Printing Learned Network Parameters**

println("Learned Network Parameters")
for (i, layer) in enumerate(simple_ffnn.layers)
    if typeof(layer) <: Dense
        println("Layer $(i):")
        println("  Weights (first 5x5 sub-matrix, or full if smaller):")
        # Move weights to CPU for printing to avoid GPU memory issues
        weight_matrix = cpu(layer.weight) 
        rows, cols = size(weight_matrix)
        # Display a sub-matrix or the full matrix if it's small
        display(weight_matrix[1:min(5, rows), 1:min(5, cols)])
        
        println("  Biases (first 10 elements, or full if smaller):")
        # Move biases to CPU for printing
        bias_vector = cpu(layer.bias) 
        display(bias_vector[1:min(10, length(bias_vector))])
        println("-"^30)
    end
end
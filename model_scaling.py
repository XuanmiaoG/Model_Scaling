import os
import shutil
import pandas as pd

def format_value(value):
    return int(value) if value == int(value) else value

def main():
    trained_model_dir = "trained_models/ResNet18/"
    scaled_model_dir = "scaled_models/ResNet18/"
    shared_weights_path = os.path.join(trained_model_dir, "ResNet18_shared_weights.pth")

    if not os.path.exists(scaled_model_dir):
        os.makedirs(scaled_model_dir)

    results_csv_path = "best_accuracy_epoch_results.csv"
    results_df = pd.read_csv(results_csv_path)

    scaling_gradient = 0.05

    selected_models = []
    selected_models_data = []

    # List all files in the trained models directory
    trained_files = os.listdir(trained_model_dir)
    print("Files in trained models directory:", trained_files)

    for index, row in results_df.iterrows():
        learning_rate = format_value(row['learning_rate'])
        pruning_amount = format_value(row['pruning_amount'])
        width_scaling_factor = format_value(row['width_scaling_factor'])
        depth_scaling_factor = format_value(row['depth_scaling_factor'])
        average_test_time = row['average_test_time']

        model_file_name = f"ResNet18_lr{learning_rate}_pa{pruning_amount}_wsf{width_scaling_factor}_dsf{depth_scaling_factor}_avgt{average_test_time}.pth"
        model_path = os.path.join(trained_model_dir, model_file_name)
        
        if model_file_name not in trained_files:
            print(f"File {model_file_name} does not exist.")
            continue

        if selected_models:
            previous_model_data = selected_models_data[-1]
            previous_flops = previous_model_data['average_test_time']
            current_flops = row['average_test_time']

            if current_flops - previous_flops < scaling_gradient * previous_flops:
                continue

        selected_models.append(model_file_name)
        selected_models_data.append(row)

    for model_file_name in selected_models:
        src_path = os.path.join(trained_model_dir, model_file_name)
        dst_path = os.path.join(scaled_model_dir, model_file_name)
        shutil.copy(src_path, dst_path)
        print(f"Copied {model_file_name} to {dst_path}")

    # Save selected models data to a new CSV file
    selected_models_df = pd.DataFrame(selected_models_data)
    selected_models_csv_path = "selected_model_variants.csv"
    selected_models_df.to_csv(selected_models_csv_path, index=False)
    print(f"Selected models data saved to {selected_models_csv_path}")

if __name__ == "__main__":
    main()

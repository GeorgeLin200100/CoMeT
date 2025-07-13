import matplotlib.pyplot as plt

def extract_numbers_from_file1(file_path):
    bank_access_count = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('@&'):
                numbers = line.split()
                if len(numbers) >= 5:
                    bank_access_count.append(int(numbers[4]))
    return bank_access_count

def extract_numbers_from_file2(file_path, target_columns):
    temperature_dict = {col: [] for col in target_columns}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) < 2:
            return temperature_dict
        headers = lines[0].split()
        target_indices = {col: headers.index(col) for col in target_columns if col in headers}
        for line in lines[1:]:
            numbers = line.split()
            for col, idx in target_indices.items():
                if len(numbers) > idx:
                    temperature_dict[col].append(float(numbers[idx]))
    return temperature_dict

def plot_data(bank_access_count, temperature_dict):
    # 检查长度一致性
    min_len = min([len(bank_access_count)] + [len(v) for v in temperature_dict.values()])
    if not all(len(v) == min_len for v in temperature_dict.values()) or len(bank_access_count) != min_len:
        print("Error: The lengths of the arrays are not consistent.")
        print(len(bank_access_count))
        for k, v in temperature_dict.items():
            print(f"{k}: {len(v)}")
        return
    time = list(range(1, min_len + 1))
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot bank access count on the left y-axis
    ax1.plot(time, bank_access_count[:min_len], label='bank access count per ms', color='b', linewidth=2)
    ax1.set_xlabel('Time (ms)', fontsize=20)
    ax1.set_ylabel('Bank Access Count Per MS', color='b', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)

    # Create a second y-axis for temperature
    ax2 = ax1.twinx()
    colors = ['r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, (col, temp) in enumerate(temperature_dict.items()):
        ax2.plot(time, temp[:min_len], label=f'temperature({col})', color=colors[i % len(colors)], linewidth=1.5)
    ax2.set_ylabel('Temperature(℃)', color='r', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=20)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=16, bbox_to_anchor=(0.9,0.1))

    # Title
    plt.title('Bank Access Count and Temperature over Time', fontsize=22)
    plt.savefig("./D.png", dpi=300, bbox_inches="tight")
    plt.show()
    

# Replace 'file1.txt' and 'file2.txt' with your actual file paths
file1_path = 'trace2.log'
file2_path = 'combined_temperature.trace'

target_columns = ['B_1', 'B_17', 'B_33', 'B_49', 'B_65', 'B_81', 'B_97', 'B_113']
bank_access_count = extract_numbers_from_file1(file1_path)
temperature_dict = extract_numbers_from_file2(file2_path, target_columns)

plot_data(bank_access_count, temperature_dict)


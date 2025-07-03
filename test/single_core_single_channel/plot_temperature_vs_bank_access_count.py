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

def extract_numbers_from_file2(file_path, target_column):
    temperature = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) < 2:
            return temperature
        headers = lines[0].split()
        if target_column not in headers:
            return temperature
        target_index = headers.index(target_column)
        for line in lines[1:]:
            numbers = line.split()
            if len(numbers) > target_index:
                temperature.append(float(numbers[target_index]))
    return temperature

def plot_data(bank_access_count, temperature):
    if len(bank_access_count) != len(temperature):
        print("Error: The lengths of the two arrays are not consistent.")
        print(len(bank_access_count))
        print(len(temperature))
        return
    time = list(range(1, len(bank_access_count) + 1))
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, bank_access_count, label='bank access count')
    # plt.plot(time, temperature, label='temperature')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.title('Bank Access Count and Temperature over Time')
    # plt.show()
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot bank access count on the left y-axis
    ax1.plot(time, bank_access_count, label='bank access count per ms', color='b')
    ax1.set_xlabel('Time (ms)', fontsize=20)
    ax1.set_ylabel('Bank Access Count Per MS', color='b', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)

    # Create a second y-axis for temperature
    ax2 = ax1.twinx()
    ax2.plot(time, temperature, label='temperature(\u2103))', color='r')
    ax2.set_ylabel('Temperature(\u2103)', color='r', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=20)

    # Add legends
    # ax1.legend(loc='lower right', fontsize=20)
    # ax2.legend(loc='lower right', fontsize=20)
    fig.legend(loc='lower right', fontsize=20, bbox_to_anchor=(0.9,0.1))

    # Title
    plt.title('Bank Access Count and Temperature over Time', fontsize=22)
    plt.savefig("./c.png", dpi=300, bbox_inches="tight")
    plt.show()
    

# Replace 'file1.txt' and 'file2.txt' with your actual file paths
file1_path = 'trace2.log'
file2_path = 'combined_temperature.trace'

bank_access_count = extract_numbers_from_file1(file1_path)
temperature = extract_numbers_from_file2(file2_path, 'B_1')

plot_data(bank_access_count, temperature)


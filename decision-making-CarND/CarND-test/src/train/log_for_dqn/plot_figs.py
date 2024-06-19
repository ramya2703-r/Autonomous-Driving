import os
import numpy as np
import matplotlib.pyplot as plt

log_dir = "/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/log"
log_files = [log_file for log_file in os.listdir(log_dir) if log_file.endswith(".txt")]

log_files_len = len(log_files)

for i in range(log_files_len - 1):
    swapped = False
    for j in range(0, log_files_len - i - 1):
        idx1 = int(log_files[j].split('episode')[1].split('.txt')[0])
        idx2 = int(log_files[j+1].split('episode')[1].split('.txt')[0])
        if idx1 > idx2:
            log_files[j], log_files[j+1] = log_files[j+1], log_files[j]

def parse_log(log_file):
    count_left = 0
    count_right = 0
    count_collision = 0

    with open(os.path.join(log_dir, log_file), 'r') as file:
        log_contents = file.read()

    for line in log_contents.splitlines():
        if 'change_lane_left' in line:
            count_left += 1
        elif 'change_lane_right' in line:
            count_right += 1
        elif 'collision' in line:
            count_collision += 1
    return count_left, count_collision, count_right

episode_left_right_collision = []

for log_file in log_files:
    left, collision, right = parse_log(log_file=log_file)
    episode_left_right_collision.append([left, collision, right])

# Convert data to numpy array
data = np.array(episode_left_right_collision)

# Transpose the data to separate rows and columns
data_transposed = data.T

# Create a figure and axis
fig, ax = plt.subplots()


some = [ "left lane", "collision", "right lane"  ]
# Iterate over each row and plot it as a line
for i, row in enumerate(data_transposed):
    ax.plot(row, label=some[i])

# Set labels and title
ax.set_xlabel('Episodes')
ax.set_ylabel('Count')
ax.set_title('Vehicle Actions Overview')
ax.legend()

# Show the plot
plt.show()
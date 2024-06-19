import matplotlib.pyplot as plt

# Sample log data as provided
# log_data = """
# connected ...
# Listening to port 4567
# Connected!!!
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# collision:1
# change_lane_left execute
# change_lane_right execute
# collision:2
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# """

# episode 100
# log_data = """
# Listening to port 4567
# Connected!!!
# change_lane_left execute

# collision:1
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute

# collision:2
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute

# collision:3
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute

# collision:4
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute
# change_lane_right execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute

# collision:5
# change_lane_left execute
# change_lane_right execute
# change_lane_left execute
# change_lane_left execute
# change_lane_right execute"""


log_data = """change_lane_left execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_left execute

collision:1
change_lane_left execute

collision:2
change_lane_right execute

collision:3
change_lane_right execute

collision:4
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_left execute

collision:5
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_left execute

collision:6
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_left execute

collision:7
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_right execute

collision:8
change_lane_right execute

collision:9
change_lane_left execute
change_lane_left execute
change_lane_right execute

collision:10
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_right execute

collision:11
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_right execute
change_lane_right execute
change_lane_right execute
change_lane_left execute
change_lane_right execute

collision:12
change_lane_right execute
change_lane_left execute

collision:13
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_left execute
change_lane_right execute
"""

def parse_log(data):
    # Initialize counters
    left_count = 0
    right_count = 0
    collisions = 0

    # Iterate through each line in the log data
    for line in data.splitlines():
        if 'change_lane_right execute' in line:
            right_count += 1
        elif 'change_lane_left execute' in line:
            left_count += 1
        elif 'collision:' in line:
            collisions += int(line.split(':')[-1])

    return left_count, right_count, collisions

def plot_data(left_count, right_count, collisions):
    # Creating figure and axis objects
    fig, ax = plt.subplots()

    # Data for plotting
    actions = ['Left Lane Change', 'Right Lane Change', 'Collisions']
    counts = [left_count, right_count, collisions]

    # Creating the bar chart
    ax.bar(actions, counts, color=['blue', 'green', 'red'])

    # Adding labels and title
    ax.set_xlabel('Actions')
    ax.set_ylabel('Count')
    ax.set_title('Vehicle Actions Overview')

    # Show the plot
    plt.show()

# Parsing the log data
left_count, right_count, collisions = parse_log(log_data)

# Plotting the data
plot_data(left_count, right_count, collisions)

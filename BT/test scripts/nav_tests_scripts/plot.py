# Re-importing necessary libraries to ensure plot generation works
import matplotlib.pyplot as plt

# Data for the horizontal histogram
values = [0.39505264, 0.57510966, 0.02983765]
labels = ["empty floor", "vial on the floor", "glove on the floor"]

# Create a horizontal bar chart
plt.figure(figsize=(8, 4))
plt.barh(labels, values, color="skyblue", edgecolor="black")
plt.xlim(0, 1)
plt.xlabel("Prediction Values", fontsize=14)
plt.ylabel("Labels", fontsize=14)
plt.title("VLN based detection", fontsize=16)

# Set font size for tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Display the plot
plt.show()

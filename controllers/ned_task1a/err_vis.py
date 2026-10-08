import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("robot_data_1e-3.csv")
errors = df["error"].dropna()

# Print largest errors
print("\nTop 10 errors:")
print(errors.nlargest(10).to_string(index=False))

# Plot distribution
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50)

plt.axvline(
    errors.median(),
    linestyle="--",
    label=f"Median: {errors.median():.2f} mm"
)
plt.axvline(
    errors.quantile(0.95),
    linestyle="--",
    label=f"95th percentile: {errors.quantile(0.95):.2f} mm"
)

plt.xlabel("Error (mm)")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.legend()
plt.tight_layout()
plt.show()
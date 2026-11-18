
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Neural Network Lab
# Experiment: How Learning Rate Affects Training
# ==========================================

# 1. Create the dataset: y = 2x + 1
X = np.arange(1, 51, dtype=float)
y = 2 * X + 1

# Normalize inputs
X_normalized = (X - X.mean()) / X.std()

# Experiment settings
learning_rates = [0.0001, 0.01, 0.1, 1.0]
epochs = 100

results = {}

# 2. Train a linear model with each learning rate
for lr in learning_rates:
    weight = 0.0
    bias = 0.0
    loss_history = []
    unstable = False

    for epoch in range(epochs):
        predictions = weight * X_normalized + bias
        errors = predictions - y
        loss = np.mean(errors ** 2)

        # Stop if training becomes unstable
        if not np.isfinite(loss) or loss > 1e12:
            unstable = True
            break

        loss_history.append(loss)

        # Calculate gradients
        dw = 2 * np.mean(errors * X_normalized)
        db = 2 * np.mean(errors)

        # Update parameters
        weight -= lr * dw
        bias -= lr * db

    # Measure the final loss if training remained stable
    if not unstable:
        predictions = weight * X_normalized + bias
        final_loss = np.mean((predictions - y) ** 2)
        if not np.isfinite(final_loss) or final_loss > 1e12:
            unstable = True
    else:
        final_loss = float("inf")

    results[lr] = {
        "weight": weight,
        "bias": bias,
        "loss_history": loss_history,
        "final_loss": final_loss,
        "unstable": unstable,
    }

# 3. Display results
print("\nLEARNING RATE EXPERIMENT")
print("=" * 45)

for lr, result in results.items():
    print(f"\nLearning rate: {lr}")

    if result["unstable"]:
        print("Training status: UNSTABLE")
        print("Loss became too large or invalid.")
    else:
        print("Training status: STABLE")
        print(f"Final weight: {result['weight']:.4f}")
        print(f"Final bias:   {result['bias']:.4f}")
        print(f"Final loss:   {result['final_loss']:.8f}")

# 4. Plot loss curves
plt.figure(figsize=(10, 6))

for lr, result in results.items():
    losses = result["loss_history"]

    if losses:
        plt.plot(
            range(1, len(losses) + 1),
            losses,
            label=f"Learning rate = {lr}",
        )

plt.title("Learning Rate vs Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (Loss)")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_rate_comparison.png", dpi=150)
plt.show()

# 5. Plot predictions for stable models
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Actual data", s=25)

for lr, result in results.items():
    if not result["unstable"]:
        predictions = (
            result["weight"] * X_normalized + result["bias"]
        )

        plt.plot(X, predictions, label=f"LR = {lr}")

plt.title("Predictions at Different Learning Rates")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_rate_predictions.png", dpi=150)
plt.show()

print("\nExperiment complete!")
print("Saved graphs:")
print("- learning_rate_comparison.png")
print("- learning_rate_predictions.png")

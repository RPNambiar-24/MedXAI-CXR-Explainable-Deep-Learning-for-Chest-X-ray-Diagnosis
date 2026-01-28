import matplotlib.pyplot as plt

# Loss values from your training logs
baseline_loss = [61.663, 34.470, 29.178, 24.834, 23.165]
resnet_loss   = [37.159, 10.152, 4.943, 3.113, 0.900]

epochs = range(1, 6)

plt.figure(figsize=(8,5))
plt.plot(epochs, baseline_loss, marker='o', label='Baseline CNN')
plt.plot(epochs, resnet_loss, marker='s', label='ResNet50')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()

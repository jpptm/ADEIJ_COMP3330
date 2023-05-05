from sklearn.metrics import confusion_matrix

def confusion_matrix(model):
  # Get predictions for the test set
  test_preds = []
  test_labels = []
  with torch.no_grad():
      for inputs, targets in tqdm(
          test_loader,
          position=1,
          total=len(test_loader),
          leave=False,
          desc="Testing",
      ):
          # Cast tensors to device
          inputs, targets = inputs.to(device), targets.to(device)

          # Get model predictions
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)

          # Save predictions and labels
          test_preds.extend(predicted.cpu().numpy())
          test_labels.extend(targets.cpu().numpy())

  # Compute confusion matrix
  cm = confusion_matrix(test_labels, test_preds)

  # Plot confusion matrix
  plt.figure()
  plot_confusion_matrix(cm, classes=class_names)
  plt.show()
# Third-party modules
import torch

def train_fitness_finder_from_plm_embeddings_nn(model, train_loader, val_loader, criterion, optimiser, max_epochs: int = 100, patience: int = 10, device: str = "cpu"):

    model.to(device)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in range(max_epochs):

        # Training phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:

            inputs = batch['sequence_representation'].float().to(device)  # The 320-length vector embeddings
            fitness_values = batch['fitness_value'].float().to(device)   # The target fitness values

            # Zero the parameter gradients
            optimiser.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), fitness_values)

            # Backward pass and optimization
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        # Log training loss
        if (epoch + 1) % 1 == 0: print(f"Epoch [{epoch + 1}/{max_epochs}], Training Loss: {running_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                inputs = batch['sequence_representation'].float().to(device)
                fitness_values = batch['fitness_value'].float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), fitness_values)
                val_loss += loss.item()

        # Log validation loss
        if (epoch + 1) % 1 == 0: print(f"Epoch [{epoch + 1}/{max_epochs}], Validation Loss: {val_loss / len(val_loader)}")

        # Early stopping check
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model = model.state_dict()  # Save the best model

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model is not None:

            model.load_state_dict(best_model)

    return model

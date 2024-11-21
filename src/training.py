# Third-party modules
import torch
from torch.utils.data import DataLoader

# Local modules
from config_loader import config
from visuals import plot_loss

def train_energy_and_fitness_finder_from_plm_embeddings_nn(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimiser,
    results_path,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu"):

    model.to(device)

    best_energy_val_loss = float('inf')
    best_fitness_val_loss = float('inf')
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None
    
    training_loss_list = []
    validation_loss_list = []

    for epoch in range(max_epochs):

        model.train()
        running_loss = 0.0

        for batch in train_loader:

            inputs = batch["sequence_representation"].float().to(device)
            energy_values = batch["energy_value"].float().to(device)
            fitness_values = batch["fitness_value"].float().to(device)

            energy_mask = batch["energy_mask"].bool().to(device)
            fitness_mask = batch["fitness_mask"].bool().to(device)

            optimiser.zero_grad()
            outputs = model(inputs)

            energy_predictions = outputs[:, 0].squeeze()
            fitness_predictions = outputs[:, 1].squeeze()

            energy_loss = 0
            fitness_loss = 0

            if energy_predictions.masked_select(energy_mask).nelement() > 0:

                energy_loss = criterion(energy_predictions.masked_select(energy_mask), energy_values.masked_select(energy_mask))

            if fitness_predictions.masked_select(fitness_mask).nelement() > 0:

                fitness_loss = criterion(fitness_predictions.masked_select(fitness_mask), fitness_values.masked_select(fitness_mask))

            loss = energy_loss + fitness_loss
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        # Log training loss
        if (epoch + 1) % 1 == 0:
            
            print(f"Epoch [{epoch + 1}/{max_epochs}], Training Loss: {running_loss / len(train_loader)}")
        
        training_loss_list.append(running_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                inputs = batch['sequence_representation'].float().to(device)
                energy_values = batch["energy_value"].float().to(device)
                fitness_values = batch['fitness_value'].float().to(device)

                energy_mask = batch["energy_mask"].bool().to(device)
                fitness_mask = batch["fitness_mask"].bool().to(device)

                outputs = model(inputs)

                energy_predictions = outputs[:, 0].squeeze()
                fitness_predictions = outputs[:, 1].squeeze()

                energy_loss = 0
                fitness_loss = 0

                if energy_predictions.masked_select(energy_mask).nelement() > 0:

                    energy_loss = criterion(energy_predictions.masked_select(energy_mask), energy_values.masked_select(energy_mask))

                if fitness_predictions.masked_select(fitness_mask).nelement() > 0:

                    fitness_loss = criterion(fitness_predictions.masked_select(fitness_mask), fitness_values.masked_select(fitness_mask))

                loss = energy_loss + fitness_loss
                val_loss += loss.item()

        # Log validation loss
        if (epoch + 1) % 1 == 0:
            
            print(f"Epoch [{epoch + 1}/{max_epochs}], Validation Loss: {val_loss / len(val_loader)}")

        validation_loss_list.append(val_loss / len(val_loader))

        # Early stopping check
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model = model.state_dict()  # Save the best model

        else:

            epochs_without_improvement += 1

        if (epochs_without_improvement >= patience) and (epoch > config["TRAINING_PARAMETERS"]["MIN_EPOCHS"]):

            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model is not None:

        model.load_state_dict(best_model)
    
    plot_loss(training_loss_list, validation_loss_list, results_path)

    return model

def train_fitness_finder_from_plm_embeddings_nn(model, train_loader: DataLoader, val_loader: DataLoader, criterion, optimiser, max_epochs: int = 100, patience: int = 10, device: str = "cpu"):

    model.to(device)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in range(max_epochs):

        # Training phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:

            inputs = batch["sequence_representation"].float().to(device)
            fitness_values = batch["fitness_value"].float().to(device)

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
            best_model = model.state_dict()

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model is not None:

            model.load_state_dict(best_model)

    return model

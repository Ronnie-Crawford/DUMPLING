# Third-party modules
import torch
from torch.utils.data import DataLoader

# Local modules
from config_loader import config
from visuals import plot_loss

def handle_training_models(
    downstream_model_type,
    model,
    dataloaders,
    output_features,
    criterion,
    optimiser,
    results_path,
    min_epochs,
    max_epochs,
    patience,
    device
):
    
    trained_model = None
    
    match downstream_model_type:
        
        case "FFNN":
            
            trained_model = train_ffnn_from_embeddings(
                model,
                dataloaders,
                output_features,
                criterion,
                optimiser,
                results_path,
                min_epochs,
                max_epochs,
                patience,
                device
            )
        
        case "LSTM_UNIDIRECTIONAL" | "LSTM_BIDIRECTIONAL" | "GRU_UNIDIRECTIONAL" | "GRU_BIDIRECTIONAL":
            
            trained_model = train_rnn_from_embeddings(
                model,
                dataloaders,
                output_features,
                criterion,
                optimiser,
                results_path,
                min_epochs,
                max_epochs,
                patience,
                device
            )
        
    return trained_model

def train_ffnn_from_embeddings(
    model,
    dataloaders: dict,
    output_features: list,
    criterion,
    optimiser,
    results_path,
    min_epochs: int,
    max_epochs: int,
    patience: int,
    device: str
):
    
    model.to(device)
    
    if dataloaders["train"] is None:
        
        raise Exception("Currently do not support running without training.")
    
    epochs_without_improvement = 0
    best_model = None
    training_loss_list = []
    validation_loss_list = []
    best_training_loss = float("inf")
    best_validation_loss = float("inf")

    for epoch in range(max_epochs):

        # Prepare to start training
        model.train()
        training_loss = 0.0
        
        # Iterate through batches in training data loader
        for batch in dataloaders["train"]:
            
            inputs = batch["sequence_representation"].float().to(device)
            values = {}
            masks = {}
            
            for output_feature in output_features:
                
                values[output_feature] = batch[f"{output_feature}_value"].float().to(device)
                masks[output_feature] = batch[f"{output_feature}_mask"].bool().to(device)
            
            optimiser.zero_grad()
            outputs = model(inputs)
            
            predictions = {}
            losses = {}
            
            for index, output_feature in enumerate(output_features):
                
                predictions[output_feature] = outputs[:, index].view(-1)
                losses[output_feature] = torch.tensor(0.0, device = device)
                
                if predictions[output_feature].masked_select(masks[output_feature]).nelement() > 0:
                    
                    losses[output_feature] = criterion(
                        predictions[output_feature].masked_select(masks[output_feature]),
                        values[output_feature].masked_select(masks[output_feature])
                        )
                    
            total_loss = sum(losses.values())
            total_loss.backward()
            optimiser.step()
            training_loss += total_loss.item()
            
        # Log training loss
        if (epoch + 1) % 1 == 0:
        
            print(f"Epoch [{epoch + 1}/{max_epochs}], Training Loss: {training_loss / len(dataloaders['train'])}")
    
        training_loss_list.append(training_loss / len(dataloaders["train"]))
        
        # Validation phase of trianing step
        if dataloaders["validation"] is None:
        
            # Early stopping check
            average_training_loss = training_loss / len(dataloaders["train"])

            if average_training_loss < best_training_loss:

                best_training_loss = average_training_loss
                epochs_without_improvement = 0
                best_model = model.state_dict()

            else:

                epochs_without_improvement += 1

            if (epochs_without_improvement >= patience) and (epoch > min_epochs):

                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        else:
            
            model.eval()
            validation_loss = 0.0
            
            with torch.no_grad():
                
                for batch in dataloaders["validation"]:

                    inputs = batch["sequence_representation"].float().to(device)
                    output_feature_values = {}
                    output_feature_masks = {}
                    
                    for output_feature in output_features:
            
                        output_feature_values[output_feature] = batch[f"{output_feature}_value"].float().to(device)
                        output_feature_masks[output_feature] = batch[f"{output_feature}_mask"].bool().to(device)
                    
                    outputs = model(inputs)

                    output_feature_predictions = {}
                    output_feature_losses = {}

                    for index, output_feature in enumerate(output_features):
            
                        output_feature_predictions[output_feature] = outputs[:, index].squeeze()
                        output_feature_losses[output_feature] = 0
                        
                        if output_feature_predictions[output_feature].masked_select(output_feature_masks[output_feature]).nelement() > 0:
                
                            output_feature_losses[output_feature] = criterion(
                                output_feature_predictions[output_feature].masked_select(output_feature_masks[output_feature]),
                                output_feature_values[output_feature].masked_select(output_feature_masks[output_feature])
                                )
                            
                    loss = sum(output_feature_losses.values())
                    validation_loss += loss.item()
                        
            # Log validation loss
            if (epoch + 1) % 1 == 0:
                
                print(f"Epoch [{epoch + 1}/{max_epochs}], Validation Loss: {validation_loss / len(dataloaders['validation'])}")

            validation_loss_list.append(validation_loss / len(dataloaders["validation"]))
            
            # Early stopping check
            average_validation_loss = validation_loss / len(dataloaders["validation"])

            if average_validation_loss < best_validation_loss:

                best_validation_loss = average_validation_loss
                epochs_without_improvement = 0
                best_model = model.state_dict()

            else:

                epochs_without_improvement += 1

            if (epochs_without_improvement >= patience) and (epoch > min_epochs):

                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
    if best_model is not None:

        model.load_state_dict(best_model)

    plot_loss(training_loss_list, validation_loss_list, results_path)

    return model

def train_rnn_from_embeddings(
    model,
    dataloaders: dict,
    output_features: list,
    criterion,
    optimiser,
    results_path,
    min_epochs: int,
    max_epochs: int,
    patience: int,
    device: str
):
    model.to(device)
    
    if dataloaders["train"] is None:
        
        raise Exception("Currently do not support running without training.")
    
    epochs_without_improvement = 0
    best_model = None
    training_loss_list = []
    validation_loss_list = []
    best_training_loss = float("inf")
    best_validation_loss = float("inf")

    for epoch in range(max_epochs):

        # Prepare to start training
        model.train()
        training_loss = 0.0
        
        # Iterate through batches in training data loader
        for batch in dataloaders["train"]:
            
            inputs = batch["sequence_representation"].float().to(device)  # Shape: (batch_size, seq_length, embedding_dim)
            lengths = batch["length"].to(device)
            values = {}
            masks = {}
            
            for output_feature in output_features:
                
                values[output_feature] = batch[f"{output_feature}_value"].float().to(device)
                masks[output_feature] = batch[f"{output_feature}_mask"].bool().to(device)
            
            optimiser.zero_grad()
            outputs = model(inputs, lengths)
            
            predictions = {}
            losses = {}
            
            for index, output_feature in enumerate(output_features):
                
                predictions[output_feature] = outputs[:, index].view(-1)
                losses[output_feature] = torch.tensor(0.0, device=device)
                
                if predictions[output_feature].masked_select(masks[output_feature]).nelement() > 0:
                    
                    losses[output_feature] = criterion(
                        predictions[output_feature].masked_select(masks[output_feature]),
                        values[output_feature].masked_select(masks[output_feature])
                    )
                    
            total_loss = sum(losses.values())
            total_loss.backward()
            optimiser.step()
            training_loss += total_loss.item()
            
        # Log training loss
        if (epoch + 1) % 1 == 0:
            
            print(f"Epoch [{epoch + 1}/{max_epochs}], Training Loss: {training_loss / len(dataloaders['train'])}")
    
        training_loss_list.append(training_loss / len(dataloaders["train"]))
        
        # Validation phase
        if dataloaders["validation"] is None:
            
            # Early stopping check
            average_training_loss = training_loss / len(dataloaders["train"])

            if average_training_loss < best_training_loss:
                
                best_training_loss = average_training_loss
                epochs_without_improvement = 0
                best_model = model.state_dict()

            else:
                
                epochs_without_improvement += 1

            if (epochs_without_improvement >= patience) and (epoch > min_epochs):
                
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
        else:
            
            model.eval()
            validation_loss = 0.0
            
            with torch.no_grad():
                
                for batch in dataloaders["validation"]:

                    inputs = batch["sequence_representation"].float().to(device)
                    lengths = batch["length"].to(device)
                    values = {}
                    masks = {}
                    
                    for output_feature in output_features:
                        
                        values[output_feature] = batch[f"{output_feature}_value"].float().to(device)
                        masks[output_feature] = batch[f"{output_feature}_mask"].bool().to(device)
                    
                    outputs = model(inputs, lengths)

                    predictions = {}
                    losses = {}

                    for index, output_feature in enumerate(output_features):
                        
                        predictions[output_feature] = outputs[:, index].view(-1)
                        losses[output_feature] = torch.tensor(0.0, device=device)
                        
                        if predictions[output_feature].masked_select(masks[output_feature]).nelement() > 0:
                            
                            losses[output_feature] = criterion(
                                predictions[output_feature].masked_select(masks[output_feature]),
                                values[output_feature].masked_select(masks[output_feature])
                            )
                            
                    total_loss = sum(losses.values())
                    validation_loss += total_loss.item()
                        
            # Log validation loss
            if (epoch + 1) % 1 == 0:
                
                print(f"Epoch [{epoch + 1}/{max_epochs}], Validation Loss: {validation_loss / len(dataloaders['validation'])}")

            validation_loss_list.append(validation_loss / len(dataloaders["validation"]))
            
            # Early stopping check
            average_validation_loss = validation_loss / len(dataloaders["validation"])

            if average_validation_loss < best_validation_loss:
                
                best_validation_loss = average_validation_loss
                epochs_without_improvement = 0
                best_model = model.state_dict()

            else:
                
                epochs_without_improvement += 1

            if (epochs_without_improvement >= patience) and (epoch > min_epochs):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
    if best_model is not None:
        
        model.load_state_dict(best_model)

    plot_loss(training_loss_list, validation_loss_list, results_path)

    return model


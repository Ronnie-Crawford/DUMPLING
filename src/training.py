# Standard module
from pathlib import Path
import gc

# Third-party modules
import torch

def handle_training_models(
    dataloaders,
    model,
    criterion_dict,
    optimiser,
    output_features,
    embedding_size,
    downstream_model_types,
    min_epochs,
    max_epochs,
    patience,
    device,
    results_path
):

    trained_model = None

    match downstream_model_types[0]:

        case "FFNN":

            trained_model = train_ffnn_from_embeddings(
                dataloaders,
                model,
                criterion_dict,
                optimiser,
                output_features,
                results_path,
                min_epochs,
                max_epochs,
                patience,
                device
            )

    save_trained_model(trained_model, results_path, embedding_size)

    return trained_model

def train_ffnn_from_embeddings(
    dataloaders: dict,
    model,
    criterion_dict,
    optimiser,
    output_features: list,
    results_path,
    min_epochs: int,
    max_epochs: int,
    patience: int,
    device: str
):

    """
    Runs each batch though the model, finds the loss and backpropogates.
    """

    model.to(device)

    if dataloaders["TRAIN"] is None:

        raise Exception("No data selected for training.")

    #print(dataloaders["TRAIN"][0])

    epochs_without_improvement = 0
    best_model = None
    training_loss_list = []
    validation_loss_list = []
    best_training_loss = float("inf")
    best_validation_loss = float("inf")

    for epoch in range(max_epochs):

        # Prepare to start training
        model.train()
        training_loss = torch.tensor(0.0, device = device)

        # Iterate through batches in training data loader
        for batch in dataloaders["TRAIN"]:

            inputs = batch["sequence_embedding"].float().to(device)
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

                    losses[output_feature] = criterion_dict[output_feature](
                        predictions[output_feature].masked_select(masks[output_feature]),
                        values[output_feature].masked_select(masks[output_feature])
                        )

            total_loss = torch.stack(list(losses.values())).sum()
            total_loss.backward()
            optimiser.step()
            training_loss += total_loss.item()

        # Record training loss
        if (epoch + 1) % 1 == 0:

            print(f"Epoch [{epoch + 1}/{max_epochs}], Training Loss: {training_loss / len(dataloaders['TRAIN'])}")

        training_loss_list.append(training_loss / len(dataloaders["TRAIN"]))

        # Validation phase of trianing step
        if dataloaders.get("VALIDATION", None) is None:

            # Early stopping check
            average_training_loss = training_loss / len(dataloaders["TRAIN"])

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
            validation_loss = torch.tensor(0.0, device = device)

            with torch.no_grad():

                for batch in dataloaders["VALIDATION"]:

                    inputs = batch["sequence_embedding"].float().to(device)
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

                            output_feature_losses[output_feature] = criterion_dict[output_feature](
                                output_feature_predictions[output_feature].masked_select(output_feature_masks[output_feature]),
                                output_feature_values[output_feature].masked_select(output_feature_masks[output_feature])
                                )

                    loss = torch.stack(list(output_feature_losses.values())).sum()
                    validation_loss += loss.item()

            # Log validation loss
            if (epoch + 1) % 1 == 0:

                print(f"Epoch [{epoch + 1}/{max_epochs}], Validation Loss: {validation_loss / len(dataloaders['VALIDATION'])}")

            validation_loss_list.append(validation_loss / len(dataloaders["VALIDATION"]))

            # Early stopping check
            average_validation_loss = validation_loss / len(dataloaders["VALIDATION"])

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

    if "VALIDATION" in dataloaders: del dataloaders["VALIDATION"]
    if "TRAINING" in dataloaders: del dataloaders["TRAIN"]
    gc.collect()

    return model

def save_trained_model(trained_model, results_path, embedding_size):

    checkpoint = {
        "model_state_dict": trained_model.state_dict(),
        "input_dimensions": embedding_size
    }

    torch.save(trained_model.state_dict(), results_path / "checkpoint.pt")

def load_trained_model(model, checkpoint_directory, device):

    checkpoint_path = Path(checkpoint_directory) / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model

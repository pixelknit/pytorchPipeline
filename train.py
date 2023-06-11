import os
import torch
import data_setup, engine, TINYVGG, utils

from torchvision import transforms

#HYPER
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 50
LEARNING_RATE = 0.001

train_dir = "/Users/felipepesantez/Documents/datasets/Fast Food Classification V2/Train"
test_dir = "/Users/felipepesantez/Documents/datasets/Fast Food Classification V2/Test"

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #transforms
    data_trasform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])

    #dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
                train_dir=train_dir,
                test_dir=test_dir,
                transform=data_trasform,
                batch_size=BATCH_SIZE
            )

    #model
    model = TINYVGG.TinyVGG(
                input_shape=3,
                hidden_units=HIDDEN_UNITS,
                output_shape=len(class_names)
            ).to(device)

    #loss and optim
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


    #training loop
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    #save
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="modular_test01.pth")

    print("Done processing!")

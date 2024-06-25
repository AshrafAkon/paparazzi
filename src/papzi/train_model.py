import time

from tqdm import tqdm

from papzi.constants import num_epochs, MODEL_PATH, CHECKPOINT_PATH, BATCH_SIZE
from papzi.trainer import Trainer
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime


def main():

    # Data transformations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/face_{}".format(timestamp))
    trainer = Trainer(writer, 250)

    # Training loop

    # if model already exists then we would like to do
    # more epochs with the same data
    if CHECKPOINT_PATH.exists() and CHECKPOINT_PATH.is_file():
        # the final model is only saved after all epoch is done
        # so checkpoint_model should be the newest model if exists
        tqdm.write("loading from checkpoint")
        trainer.load(CHECKPOINT_PATH)

    elif MODEL_PATH.exists() and MODEL_PATH.is_file():
        tqdm.write("loading trained model")
        trainer.load(MODEL_PATH)

    total = (
        (len(trainer.train_dataset) + len(trainer.val_dataset))
        * num_epochs
        // BATCH_SIZE
    )
    with tqdm(total=total, mininterval=2, colour="green") as tq:

        training_start = time.perf_counter()
        tq.set_postfix(trainer.tq_postfix)
        priv_val_acc = 0.0
        for epoch in range(
            trainer.previous_epoch, trainer.previous_epoch + num_epochs
        ):
            tq.set_description(f"Epoch: {epoch} training")
            trainer.train(tq, epoch)
            tq.set_description(f"Epoch: {epoch} validating")
            val_acc = trainer.evaluate(tq)

            if val_acc > priv_val_acc:
                trainer.save(CHECKPOINT_PATH, epoch)
                priv_val_acc = val_acc
            tq.set_postfix(trainer.tq_postfix)
            writer.add_scalar(
                "validation/accuracy",
                val_acc,
                epoch + 1,
            )
            writer.flush()
    writer.flush()

    tqdm.write(f"total training time: {time.perf_counter() - training_start}")
    trainer.save(MODEL_PATH, epoch)
    tqdm.write("model saved")


if __name__ == "__main__":
    main()

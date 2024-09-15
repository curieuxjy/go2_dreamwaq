from params_proto import ParamsProto, Proto, Flag
import os


class Args(ParamsProto):
    """ CLI Args for the program
    Try:
        python3 example.py --help
    And it should print out the help strings
    """
    seed = Proto(1, help="random seed")
    D_lr = 5e-4
    G_lr = 1e-4
    Q_lr = 1e-4
    T_lr = 1e-4
    plot_interval = 10
    verbose = Flag("the verbose flag")


if __name__ == '__main__':
    import scipy
    import numpy as np
    from ml_logger import logger, LOGGER_USER

    # Print the environment variables to ensure they are set
    print(f"ML_LOGGER_ROOT: {os.getenv('ML_LOGGER_ROOT')}")
    print(f"ML_LOGGER_USER: {os.getenv('ML_LOGGER_USER')}")

    # Configure the logger
    logger.configure(prefix=f"{LOGGER_USER}/scratch")

    # Confirm logging setup
    print(f"Logging configured with prefix: {logger.prefix}")

    # Log parameters
    logger.log_params(Args=vars(Args))

    # Upload this script for reference
    logger.upload_file(__file__)

    # Simulate training loop with logging
    for epoch in range(10):
        logger.log(step=epoch, D_loss=0.2, G_loss=0.1, mutual_information=0.01)
        logger.log_key_value(epoch, 'some string key', 0.0012)
        logger.flush()  # Ensure logs are flushed after each step

    # Ensure all logs are flushed at the end
    logger.flush()

    # Save images
    face = scipy.datasets.face()
    face_bw = scipy.datasets.face(gray=True)
    logger.save_image(face, "figures/face_rgb.png")
    logger.save_image(face_bw, "figures/face_bw.png")

    # Save videos
    logger.save_video([face] * 5, "videos/face.mp4")

    print("Logging complete.")

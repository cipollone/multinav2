"""Misc helpers."""

import os
import shutil


# Output base dir
output_base = "outputs"


def prepare_directories(
    env_name, resuming=False, args=None, no_create=False
):
    """Prepare the directories where weights and logs are saved.

    Just to know the output paths, call this function with `no_create=True`.

    :param env_name: the actual paths are a composition of `output_base` and
        `env_name`.
    :param resuming: if true, the directories are not deleted.
    :param args: argsparse.Namespace of arguments. If given, this is saved
        to 'args' file inside the log directory.
    :param no_create: do not touch the files; just return the current paths
        (implies resuming).
    :return: two paths, respectively for models and logs.
    """

    if no_create:
        resuming = True

    # Choose diretories
    models_path = os.path.join(output_base, env_name, "models")
    logs_path = os.path.join(output_base, env_name, "logs")
    dirs = (models_path, logs_path)

    # Delete old ones
    if not resuming:
        if any(os.path.exists(d) for d in dirs):
            print(
                "Old logs and models will be deleted. Continue (Y/n)? ",
                end="")
            c = input()
            if c not in ("y", "Y", ""):
                quit()

        # New
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

    # Logs and models for the same run are saved in
    #   directories with increasing numbers
    i = 0
    while (
        os.path.exists(os.path.join(logs_path, str(i))) or
        os.path.exists(os.path.join(models_path, str(i)))
    ):
        i += 1

    # Should i return the current?
    if no_create:
        last_model_path = os.path.join(models_path, str(i-1))
        last_log_path = os.path.join(logs_path, str(i-1))
        if (
            i == 0 or not os.path.exists(last_log_path) or
            not os.path.exists(last_model_path)
        ):
            raise RuntimeError("Dirs should be created first")
        return (last_model_path, last_log_path)

    # New dirs
    model_path = os.path.join(models_path, str(i))
    log_path = os.path.join(logs_path, str(i))
    os.mkdir(model_path)
    os.mkdir(log_path)

    # Save arguments
    if args is not None:
        raise NotImplementedError("Feature still not used.")

    return (model_path, log_path)

# Celebrity Face Recognition Project

Welcome to the Celebrity Face Recognition Project! This repository contains the code for an advanced face recognition system trained on ResNet50 using data from 65 celebrities. The project includes a GUI application, training scripts, and data scrapping functionalities.

## Table of Contents

- [Install Dependencies](#install-dependencies)
- [App GUI](#app-gui)
- [Training](#training)
- [Data Sources](#data-sources)

## Install Dependencies

To install the dependencies for this project, you have two options:

1. **Using Poetry** (recommended): 

    Poetry is required to install the dependencies using the `pyproject.toml` file. You can find Poetry and its installation instructions [here](https://python-poetry.org/docs/).

    ```sh
    poetry install
    ```

2. **Using Pip**:

    You can also install the dependencies with pip from the requirements file.

    ```sh
    pip install -r requirements.txt
    ```

## App GUI

The main GUI for the application can be found in the following file:

```sh
src/papzi_gui/app.py
```

## Training

The training code and data are organized as follows:

- **Training Script**: 
  - Located at: `src/papzi/train_model.py`
- **Training Data**: 
  - Located at: `src/papzi/data/train`
- **Validation Data**: 
  - Located at: `src/papzi/data/validation`
- **Trained Model**: 
  - Located at: `src/papzi/data/models/trained.pth`
- **Scrapper**:
  - Located at: `src/papzi/scrapper/scrap.py`

## Data Sources

Movie credits were downloaded from:

[![The Movie Database](https://www.themoviedb.org/assets/2/v4/logos/v2/blue_long_1-8ba2ac31f354005783fab473602c34c3f4fd207150182061e425d366e4f34596.svg)](https://www.themoviedb.org/)

---

Feel free to explore the repository and make sure to follow the instructions for setting up the dependencies and running the different components of the project. If you have any questions or issues, please open an issue or reach out to the project maintainers.
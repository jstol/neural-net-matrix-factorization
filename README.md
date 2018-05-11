# Neural Network Matrix Factorization (NNMF)
Completed in 2016 for a course taught by David Duvenaud at the University of Toronto, [CSC 2541](http://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html) ("Differentiable Inference and Generative Models").

Tensorflow prototypes of:
* Dziugaite and Roy's "Neural Network Matrix Factorization" (NNMF) model (https://arxiv.org/abs/1511.06443).
* A proposed extension which makes use of stochastic variational inference to learn an approximate posterior distribution over the latent space ("SVINNMF").

See paper ("Matrix Factorization with Neural Networks and Stochastic Variational Inference") here: https://www.cs.toronto.edu/~jstolee/projects/matrix_factorization_neural.pdf.

## Dependencies
This project was written to be compatible with Python 2.7. See `requirements.txt` for third party dependencies.

## Scripts
The `scripts/` folder contains the following Python scripts:
- `split_data.py`: for splitting up the data set into train, test and validation sets.
- `main.py`: for training, hyperparameter selection (random search) and testing.
- `predict.py`: a CLI program for prediction with a trained model.

Each of the scripts can be invoked with the `--help` flag for more information.

## Data
The [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) was used for this project - see the `data/ml-100k/` folder. (The 1M Dataset was also used, which can be found [here](https://grouplens.org/datasets/movielens/1M/)).
